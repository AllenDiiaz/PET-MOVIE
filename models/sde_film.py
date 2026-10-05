# LKMUNet-SDE-FiLM
# ================================================================
#   1) Sigmoid on the decoder output (output ∈ [0, 1])
#   2) Drift clamped to [-3, 3]; the latent at the end of the SDE
#      integration, z(1), is returned and regularized in the loss by mean(z(1)²)
#   3) FiLM: gamma = tanh(·), beta = 0.1·tanh(·); gamma and beta are returned
#      for the consistency loss 1 - cos(gamma, beta)
#   4) Fixed-step Euler solver, 8 steps
# ================================================================
import torch, torch.nn as nn, torch.nn.functional as F, torchsde
from typing import Tuple
from models.lkmunet import LKMUNet

# ---------- Utilities ---------------------------------------------
def patch_inplace_relu(mod: nn.Module):
    for n, c in mod.named_children():
        if isinstance(c, (nn.ReLU, nn.LeakyReLU)) and getattr(c, "inplace", False):
            setattr(mod, n, c.__class__(inplace=False))
        else:
            patch_inplace_relu(c)

# ---------- SDE ---------------------------------------------------
class SpatialSDEFunc(nn.Module):
    noise_type, sde_type = "diagonal", "ito"
    def __init__(self, C):
        super().__init__(); self.C = C
        self.mu = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, 3, 1, 1, bias=False)
        )
        self.sigma = nn.Conv2d(C, C, 1, bias=False)
        self.gamma = self.beta = None
    def set_film(self, g, b): self.gamma, self.beta = g, b

    def f(self, t, x):
        B, D = x.shape; H = W = int((D // self.C) ** .5)
        feat = x.view(B, self.C, H, W)
        out  = self.mu(feat)
        if self.gamma is not None:
            out = self.gamma * out + self.beta
        out  = torch.clamp(out, min=-3., max=3.)          # clamp drift to [-3, 3]
        return out.flatten(1)

    def g(self, t, x):
        B, D = x.shape; H = W = int((D // self.C) ** .5)
        return self.sigma(x.view(B, self.C, H, W)).flatten(1)

class SDEBlock(nn.Module):
    def __init__(self, func, *, steps=8, tol=1e-2, adjoint=True):
        super().__init__(); self.func, self.steps, self.tol, self.adjoint = func, steps, tol, adjoint
    def forward(self, x_map, int_t, g=None, b=None):
        self.func.set_film(g, b)
        B, C, H, W = x_map.shape
        x_vec = x_map.flatten(1)
        solver = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint
        out = solver(self.func, x_vec, int_t,
                     dt=1./self.steps, method="euler",
                     rtol=self.tol, atol=self.tol)
        return out[-1].view(B, C, H, W)                    # z(1)

# ---------- Main model --------------------------------------------
class LKMUNetSDE_FiLM(nn.Module):
    def __init__(self, backbone, mid_encoder, sde_dim,
                 *, reduce=True, steps=8, strategy="auto"):
        super().__init__()
        self.backbone, self.mid_encoder, self.strategy = backbone, mid_encoder, strategy
        patch_inplace_relu(backbone)
        if mid_encoder is not None: patch_inplace_relu(mid_encoder)

        C_backbone = backbone.encoder.output_channels[-1]
        self.reduce_conv = nn.Identity() if (not reduce or C_backbone == sde_dim) \
                           else nn.Conv2d(C_backbone, sde_dim, 1, bias=False)
        self.down_pool  = nn.AvgPool2d(2)
        self.sde_block  = SDEBlock(SpatialSDEFunc(sde_dim), steps=steps, tol=1e-2, adjoint=True)

        if mid_encoder is not None:
            self.mid_reduce  = nn.Identity() if C_backbone == sde_dim \
                               else nn.Conv2d(C_backbone, sde_dim, 1, bias=False)
            self.film_gamma  = nn.Conv2d(sde_dim, sde_dim, 1)
            self.film_beta   = nn.Conv2d(sde_dim, sde_dim, 1)
            self.concat_reduce = None           # lazy build

        self.up_bridge = nn.ConvTranspose2d(sde_dim, C_backbone, 2, 2, 0, bias=False)

    # --------- Compute FiLM gamma / beta -------------------------
    def _prepare_film(self, mid_pet: torch.Tensor, hw: Tuple[int, int]):
        if self.mid_encoder is None: return None, None
        if mid_pet.ndim == 5:                       # [B,M,C,H,W]
            B, M, C0, H, W = mid_pet.shape
        else:
            B, C0, H, W = mid_pet.shape; M = 1

        strat = self.strategy
        if strat == "auto":
            enc_in = next(m.in_channels for m in self.mid_encoder.modules()
                          if isinstance(m, nn.Conv2d))
            strat = "concat" if (M * C0 == enc_in) else "stack"

        # concat ---------------------------------------------------
        if strat == "concat":
            if mid_pet.ndim == 5:
                mid_pet = mid_pet.view(B, M * C0, H, W)
            enc_in = next(m.in_channels for m in self.mid_encoder.modules()
                          if isinstance(m, nn.Conv2d))
            if mid_pet.size(1) != enc_in:
                if (self.concat_reduce is None) or (self.concat_reduce.in_channels != mid_pet.size(1)):
                    self.concat_reduce = nn.Conv2d(mid_pet.size(1), enc_in, 1).to(mid_pet.device)
                mid_pet = self.concat_reduce(mid_pet)

        # stack ----------------------------------------------------
        elif strat == "stack" and mid_pet.ndim == 5:
            mid_pet = mid_pet.view(B * M, C0, H, W)

        # encode mid frames ---------------------------------------
        mid_feat = self.mid_encoder.encoder(mid_pet)[-1]
        mid_feat = F.interpolate(mid_feat, size=hw, mode="bilinear", align_corners=False)
        mid_feat = self.mid_reduce(mid_feat)       # → sde_dim

        if strat == "stack" and M > 1:
            C_lat = mid_feat.size(1)
            mid_feat = mid_feat.view(B, M, C_lat, *hw).mean(1)

        gamma = torch.tanh(self.film_gamma(mid_feat))
        beta  = 0.1 * torch.tanh(self.film_beta(mid_feat))
        return gamma, beta

    # --------- forward ------------------------------------------
    def forward(self, early_pet, t, *, mid_pet=None):
        skips = self.backbone.encoder(early_pet)
        x_bot = skips[-1]                                   # [B, C, h, w] bottleneck
        x_low = self.down_pool(self.reduce_conv(x_bot))     # [B, sde_dim, h/2, w/2]
        Hp, Wp = x_low.shape[2:]

        gamma = beta = None
        if mid_pet is not None and self.mid_encoder is not None:
            gamma, beta = self._prepare_film(mid_pet, (Hp, Wp))

        t_val  = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
        int_t  = torch.tensor([0., t_val], device=early_pet.device, dtype=early_pet.dtype)
        z_T = self.sde_block(x_low, int_t, gamma, beta)     # z(1), end of the SDE integration
        skips[-1] = self.up_bridge(z_T).contiguous()        # back to [B, C, h, w]

        pred = self.backbone.decoder(skips)
        pred = torch.sigmoid(pred)
        return pred, z_T, gamma, beta

# ---------- Builder ------------------------------------------------------
def build_lkmunet_sde_film(cfg: dict, *, strategy="auto"):
    net = LKMUNet(**cfg)
    mid = LKMUNet(**cfg)
    return LKMUNetSDE_FiLM(net, mid,
                           sde_dim=cfg['features_per_stage'][-1] // 2,
                           reduce=True, steps=8, strategy=strategy)

# ---------- quick test ---------------------------------------------------
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = dict(
        input_channels=1, n_stages=4,
        features_per_stage= [32, 64, 128, 256],
        conv_op=nn.Conv2d, kernel_sizes=[(3,3)]*4,
        strides=[(1,1),(2,2),(2,2),(2,2)],
        n_conv_per_stage=[2]*4, num_classes=1,
        n_conv_per_stage_decoder=[2,2,2],
        conv_bias=False, norm_op=nn.BatchNorm2d,
        norm_op_kwargs=dict(eps=1e-5, affine=True),
        dropout_op=None, dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU, nonlin_kwargs=dict(inplace=False),
        deep_supervision=False
    )
    model = build_lkmunet_sde_film(cfg, strategy="stack").to(device).eval()
    B, M = 2, 3
    x_early = torch.randn(B, 1, 128, 128, device=device)
    x_mid   = torch.randn(B, M, 1, 128, 128, device=device)
    with torch.no_grad():
        y, *_ = model(x_early, t=1.0, mid_pet=x_mid)
    print("Output :", y.min().item(), y.max().item(), y.shape)  # expected within [0, 1]
