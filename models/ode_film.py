# LKMUNet-ODE-FiLM
# ================================================================
#  - Sigmoid on the decoder output (output ∈ [0, 1])
#  - Drift clamped to [-3, 3] to keep the ODE stable; kinetic
#    regularization ∫‖f‖² dt is accumulated through an augmented state
#  - FiLM: gamma = tanh(·), beta = 0.1·tanh(·)
#  - Fixed-step Euler solver, 8 steps
# ================================================================
import torch, torch.nn as nn, torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from typing import Tuple
from models.lkmunet import LKMUNet

# ────────────────── Utilities ─────────────────────────────────────
def patch_inplace_relu(mod: nn.Module):
    for n, c in mod.named_children():
        if isinstance(c, (nn.ReLU, nn.LeakyReLU)) and getattr(c, "inplace", False):
            setattr(mod, n, c.__class__(inplace=False))
        else:
            patch_inplace_relu(c)

# ────────────────── ODE drift (FiLM) ────────────────────────────
class SpatialODEFunc(nn.Module):
    def __init__(self, C: int):
        super().__init__(); self.C = C
        self.conv = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, 3, 1, 1, bias=False)
        )
        self.gamma = self.beta = None

    def set_film(self, gamma, beta):
        self.gamma, self.beta = gamma, beta

    # state = [z | E], dE/dt = mean(f²) → E(T) = ∫‖f‖²/D dt (kinetic energy)
    def forward(self, t, x_aug):                      # d[z|E]/dt
        x_flat = x_aug[:, :-1]
        B, D = x_flat.shape; H = W = int((D // self.C) ** .5)
        x = x_flat.view(B, self.C, H, W)
        out = self.conv(x)
        if self.gamma is not None:
            out = self.gamma * out + self.beta
        out = torch.clamp(out, -3., 3.)                 # clamp drift to [-3, 3]
        fz = out.flatten(1)
        return torch.cat([fz, fz.pow(2).mean(1, keepdim=True)], 1)

# ────────────────── ODEBlock ────────────────────────────────────
class ODEBlock(nn.Module):
    def __init__(self, func, *, steps=8,
                 rtol=1e-3, atol=1e-3, method="euler", adjoint=True):
        super().__init__(); self.func, self.N, self.rtol, self.atol,\
            self.method, self.adjoint = func, steps, rtol, atol, method, adjoint
    def forward(self, x_map, int_t, gamma=None, beta=None):
        self.func.set_film(gamma, beta)
        B, C, H, W = x_map.shape
        x0 = torch.cat([x_map.flatten(1), x_map.new_zeros(B, 1)], 1)
        kw = dict(method=self.method, rtol=self.rtol, atol=self.atol,
                  options=dict(step_size=1. / self.N))
        if self.adjoint:   # gamma, beta are not parameters of func; pass them explicitly via adjoint_params
            film = tuple(p for p in (gamma, beta) if p is not None and p.requires_grad)
            out = odeint_adjoint(self.func, x0, int_t,
                      adjoint_params=tuple(self.func.parameters()) + film, **kw)
        else:
            out = odeint(self.func, x0, int_t, **kw)
        return out[-1, :, :-1].view(B, C, H, W), out[-1, :, -1]   # z_T, kinetic[B]

# ────────────────── Main model (ODE block) ───────────────────────
class LKMUNetODE_FiLM(nn.Module):
    def __init__(self, backbone, mid_encoder, sde_dim,
                 *, reduce=True, steps=8, strategy="auto"):
        super().__init__()
        self.backbone, self.mid_encoder, self.strategy = backbone, mid_encoder, strategy
        patch_inplace_relu(backbone)
        if mid_encoder is not None:
            patch_inplace_relu(mid_encoder)

        C_backbone = backbone.encoder.output_channels[-1]
        self.reduce_conv = nn.Identity() if (not reduce or C_backbone == sde_dim) \
                           else nn.Conv2d(C_backbone, sde_dim, 1, bias=False)
        self.down_pool = nn.AvgPool2d(2)
        self.ode_block = ODEBlock(
            SpatialODEFunc(sde_dim),
            steps=steps, rtol=1e-3, atol=1e-3,
            method="euler", adjoint=True
        )

        if mid_encoder is not None:
            self.mid_reduce  = nn.Identity() if C_backbone == sde_dim \
                               else nn.Conv2d(C_backbone, sde_dim, 1, bias=False)
            self.film_gamma  = nn.Conv2d(sde_dim, sde_dim, 1)
            self.film_beta   = nn.Conv2d(sde_dim, sde_dim, 1)
            self.concat_reduce = None           # lazy build

        self.up_bridge = nn.ConvTranspose2d(sde_dim, C_backbone, 2, 2, 0, bias=False)

    # -------- gamma / beta ----------------------------------------------
    def _prepare_film(self, mid_pet: torch.Tensor, hw: Tuple[int, int]):
        if self.mid_encoder is None:
            return None, None, None
        if mid_pet.ndim == 5:
            B, M, C0, H, W = mid_pet.shape
        else:
            B, C0, H, W = mid_pet.shape; M = 1

        strat = self.strategy
        if strat == "auto":
            enc_in = next(m.in_channels for m in self.mid_encoder.modules()
                          if isinstance(m, nn.Conv2d))
            strat = "concat" if (M * C0 == enc_in) else "stack"

        # concat -----------------------------------------------------------
        if strat == "concat":
            if mid_pet.ndim == 5:
                mid_pet = mid_pet.view(B, M * C0, H, W)
            enc_in = next(m.in_channels for m in self.mid_encoder.modules()
                          if isinstance(m, nn.Conv2d))
            if mid_pet.size(1) != enc_in:
                if (self.concat_reduce is None) or (self.concat_reduce.in_channels != mid_pet.size(1)):
                    self.concat_reduce = nn.Conv2d(mid_pet.size(1), enc_in, 1).to(mid_pet.device)
                mid_pet = self.concat_reduce(mid_pet)

        # stack ------------------------------------------------------------
        elif strat == "stack" and mid_pet.ndim == 5:
            mid_pet = mid_pet.view(B * M, C0, H, W)

        # encode mid frames -----------------------------------------------
        mid_feat = self.mid_encoder.encoder(mid_pet)[-1]
        mid_feat = F.interpolate(mid_feat, size=hw, mode="bilinear", align_corners=False)
        mid_feat = self.mid_reduce(mid_feat)

        if strat == "stack" and M > 1:
            C_lat = mid_feat.size(1)
            mid_feat = mid_feat.view(B, M, C_lat, *hw).mean(1)

        gamma = torch.tanh(self.film_gamma(mid_feat))
        beta  = 0.1 * torch.tanh(self.film_beta(mid_feat))
        return gamma, beta, mid_feat

    # -------- forward ----------------------------------------------------
    def forward(self, early_pet, t, *, mid_pet=None):
        skips = self.backbone.encoder(early_pet)
        x_bot = skips[-1]                                   # [B, C, h, w] bottleneck
        x_low = self.down_pool(self.reduce_conv(x_bot))     # [B, sde_dim, h/2, w/2]
        Hp, Wp = x_low.shape[2:]

        gamma = beta = mid_feat = None
        if (mid_pet is not None) and (self.mid_encoder is not None):
            gamma, beta, mid_feat = self._prepare_film(mid_pet, (Hp, Wp))

        t_val = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
        int_t = torch.tensor([0., t_val], device=early_pet.device, dtype=early_pet.dtype)
        x_low_out, kinetic = self.ode_block(x_low, int_t, gamma, beta)
        skips[-1] = self.up_bridge(x_low_out).contiguous()  # back to [B, C, h, w]

        pred = self.backbone.decoder(skips)
        pred = torch.sigmoid(pred)
        return pred, kinetic, x_low, mid_feat

# ────────────────── Builder (mid-encoder shares the backbone architecture) ────────────────
def build_lkmunet_ode_film(cfg: dict, *, strategy="auto"):
    net = LKMUNet(**cfg)
    mid = LKMUNet(**cfg)
    return LKMUNetODE_FiLM(
        net, mid,
        sde_dim=cfg['features_per_stage'][-1] // 2,
        reduce=True, steps=8, strategy=strategy
    )

# ────────────────── quick smoke test ───────────────────────────
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = dict(
        input_channels=1, n_stages=4, features_per_stage=[32, 64, 128, 256],
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

    model = build_lkmunet_ode_film(cfg, strategy="stack").to(device).eval()
    B, M = 2, 3
    x_early = torch.randn(B, 1, 128, 128, device=device)
    x_mid   = torch.randn(B, M, 1, 128, 128, device=device)
    with torch.no_grad():
        y, *_ = model(x_early, t=1.0, mid_pet=x_mid)
    print("Output range:", y.min().item(), y.max().item(), "| shape:", y.shape)
