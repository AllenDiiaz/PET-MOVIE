# -*- coding: utf-8 -*-
"""
train.py - training and validation of LKMUNet-SDE/ODE-FiLM
(gradient accumulation, guide-dropout, best-checkpoint selection on the validation set)
"""

from __future__ import annotations    
from pathlib import Path
from typing import Dict, List
import gc, os, json
import numpy as np              
import pandas as pd              
import matplotlib.pyplot as plt  
import seaborn as sns            

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as calc_psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from lpips import LPIPS
from colorama import Fore, Style
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import random    
import warnings
from datetime import datetime

from models.sde_film import build_lkmunet_sde_film
from models.ode_film import build_lkmunet_ode_film
from models.lkmunet import apply_gradient_checkpoint
from datasets.early2late_dataset import Early2LateWithLatentDataset

sns.set_style("whitegrid")

# ────────────────────────── Global settings ──────────────────────────
PLOT_METRIC  = True
SAVE_SYMLINK = True
ACC_STEPS    = 4   
WEIGHT_DECAY = 3e-2
CLIP_GRAD_NORM = 1.0
# ─────────────────────────────────────────────────────────────

# ═══════════════════════ Utilities ════════════════════════
def squeeze_early(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(2) if (x.ndim == 5 and x.size(2) == 1) else x

# ────────────────────────── Baseline CSV ───────────────────
def load_baseline_metrics(fold: int, csv_path: str) -> Dict[str, float]:
    """Return {'psnr':…, 'ssim':…, 'lpips':…} for given fold."""
    if not csv_path:
        return {"psnr": -np.inf, "ssim": -np.inf, "lpips": np.inf}
    if not Path(csv_path).exists():
        return {'psnr': -np.inf, 'ssim': -np.inf, 'lpips': np.inf}
    df = pd.read_csv(csv_path)
    row = df[df["Fold"] == fold]
    if row.empty:
        return {'psnr': -np.inf, 'ssim': -np.inf, 'lpips': np.inf}
    return {'psnr': row["mean_psnr"].max(),
            'ssim': row["mean_ssim"].max(),
            'lpips': row["mean_lpips"].min()}

# ────────────────────────── Helper: metric plots ────────────────
def update_and_plot_metrics(
        epoch: int,
        train_losses: List[float], valid_losses: List[float],
        psnr_scores: List[float],  ssim_scores: List[float],
        lpips_scores: List[float], mse_scores: List[float],
        save_dir: Path):

    loss_dir   = save_dir / "Loss";   loss_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = save_dir / "Metric"; metric_dir.mkdir(parents=True, exist_ok=True)

    np.save(loss_dir / "train_losses.npy",  np.array(train_losses,  np.float32))
    np.save(loss_dir / "valid_losses.npy",  np.array(valid_losses,  np.float32))
    np.save(metric_dir / "psnr_scores.npy", np.array(psnr_scores, np.float32))
    np.save(metric_dir / "ssim_scores.npy", np.array(ssim_scores, np.float32))
    np.save(metric_dir / "lpips_scores.npy",np.array(lpips_scores, np.float32))
    np.save(metric_dir / "mse_scores.npy",  np.array(mse_scores,  np.float32))

    plt.figure(figsize=(6,4))
    sns.lineplot(x=range(1,len(train_losses)+1), y=train_losses, label="Train")
    sns.lineplot(x=range(1,len(valid_losses)+1), y=valid_losses, label="Valid")
    best = int(np.argmin(valid_losses)); best_val = valid_losses[best]
    plt.scatter(best+1, best_val, color='red')
    plt.annotate(f"Best E{best+1}: {best_val:.4f}", (best+1, best_val),
                 xytext=(0,-12), textcoords="offset points", ha="center", fontsize=8)
    plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig(loss_dir/"loss_curve.png", dpi=300); plt.close()

    def _curve(arr, name, larger_better=True):
        plt.figure(figsize=(6,4))
        sns.lineplot(x=range(1,len(arr)+1), y=arr)
        idx = int(np.argmax(arr) if larger_better else np.argmin(arr))
        plt.scatter(idx+1, arr[idx], color='red')
        plt.annotate(f"Best E{idx+1}: {arr[idx]:.4f}", (idx+1, arr[idx]),
                     xytext=(0,-12), textcoords="offset points",
                     ha="center", fontsize=8)
        plt.title(f"{name} Curve"); plt.xlabel("Epoch"); plt.ylabel(name)
        plt.tight_layout()
        plt.savefig(metric_dir/f"{name.lower()}_curve.png", dpi=300)
        plt.close()
    _curve(psnr_scores, "PSNR", True)
    _curve(ssim_scores, "SSIM", True)
    _curve(lpips_scores, "LPIPS", False)
    _curve(mse_scores, "MSE", False)

def to_5d(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    if x.ndim == 5:
        if x.shape[2] in (1, 3) and x.shape[1] != 1:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x
    if x.ndim == 6:
        B, C1, M, C2, H, W = x.shape
        if C1 == 1 and C2 == 1:
            return x.view(B, M, 1, H, W)
        if C2 == 1:
            return x.permute(0, 2, 1, 4, 5).contiguous()
        raise ValueError(f"Unsupported 6-D latent shape {x.shape}")
    if x.ndim == 4:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x.unsqueeze(1).unsqueeze(2)
    raise ValueError(f"unsupported latent shape {x.shape}")


def guide_dropout(mid_5d: torch.Tensor | None,
                  p_full: float) -> torch.Tensor | None:
    if mid_5d is None:
        return None
    if torch.rand(1) < p_full:              # drop mid frames for the whole batch
        return None
    return mid_5d


# ───────────────────────────────── loss ───────────────────────
class ImageFlowNetLoss(nn.Module):
    """
    L = wm * MSE + ws * (1 - SSIM)/2 + wl * LPIPS
        + w_sm * ∫||f(z,t)||² dt + w_c * (1 - cos_sim(early, mid))
    """
    def __init__(
        self,
        mse_w: float    = 1.0,
        ssim_w: float   = 1.0,
        lpips_w: float  = 1.0,
        smooth_w: float = 5e-4,
        cont_w: float   = 1.0,
        resize_lpips: bool = True
    ):
        super().__init__()
        self.mse_w   = mse_w
        self.ssim_w  = ssim_w
        self.lpips_w = lpips_w
        self.w_sm    = smooth_w
        self.w_c     = cont_w
        self.mse     = nn.MSELoss()
        self.lpips   = LPIPS(net='vgg').eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.resize_lpips = resize_lpips

    @staticmethod
    def _to_rgb(x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 3, 1, 1)

    def forward(
        self,
        pred:  torch.Tensor,
        tgt:   torch.Tensor,
        drift: torch.Tensor | None = None,
        early: torch.Tensor | None = None,
        mid:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        mse  = self.mse(pred, tgt)
        ssim = (1.0 - ssim_fn(pred, tgt, data_range=1.0)) * 0.5
        if self.resize_lpips and pred.size(1) == 1:
            lpips_val = self.lpips(self._to_rgb(pred), self._to_rgb(tgt), normalize=True).mean()
        else:
            lpips_val = self.lpips(pred, tgt, normalize=True).mean()

        loss = (
            self.mse_w   * mse  +
            self.ssim_w  * ssim +
            self.lpips_w * lpips_val
        )

        if drift is not None:          # drift = ∫‖f‖²/D dt per sample (already squared during integration)
            loss = loss + self.w_sm * drift.mean()

        if (early is not None) and (mid is not None):
            e = F.normalize(early, p=2, dim=1)
            m = F.normalize(mid,   p=2, dim=1)
            loss = loss + self.w_c * (1.0 - (e * m).sum(1).mean())

        return loss

# ───────────────────────────────── best-checkpoint selection ─────────────────────────
class BestCheckpointSelector:
    """Decide whether the current epoch gives the best model so far on the validation set.

    The first epoch is always kept. Afterwards a model is kept only if it beats the
    optional baseline on PSNR, SSIM and LPIPS, and either PSNR + 25*SSIM increases by
    more than `delta` or LPIPS decreases by more than `delta`.
    """
    def __init__(self, delta: float, baseline: Dict[str, float]):
        self.delta = delta
        self.best_score = -np.inf
        self.best_lpips = np.inf
        self.baseline = baseline
        self.first = True
        self.criterion = (f"first epoch; afterwards must beat the baseline (if given) on PSNR, SSIM "
                          f"and LPIPS, and PSNR + 25*SSIM must increase by > {delta} "
                          f"or LPIPS must decrease by > {delta}")

    def check_improve(self, psnr: float, ssim: float, lpips: float) -> bool:
        if self.first:
            self.first = False
            return True
        better_than_base = (psnr > self.baseline['psnr'] and
                            ssim > self.baseline['ssim'] and
                            lpips < self.baseline['lpips'])
        if not better_than_base:
            return False
        score = psnr + 25*ssim
        improved = (score > self.best_score + self.delta) or (lpips < self.best_lpips - self.delta)
        if improved:
            self.best_score, self.best_lpips = score, lpips
        return improved


def write_ckpt_pointer(ckpt_dir: Path, name: str, info: Dict) -> None:
    """Write <name>.json and a <name>.pth symlink (relative to ckpt_dir) for info["checkpoint"]."""
    (ckpt_dir / f"{name}.json").write_text(json.dumps(info, indent=2))
    if SAVE_SYMLINK:
        link = ckpt_dir / f"{name}.pth"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(info["checkpoint"], link)
        except OSError as e:
            print(Fore.YELLOW + f"[symlink] {e}" + Style.RESET_ALL)


# ═══════════════════════ Train & Validate ══════════════════════
def train_and_validate(model          : nn.Module,
                       train_loader   : DataLoader,
                       valid_loader   : DataLoader,
                       optimizer,
                       warmup_sched,
                       num_epochs     : int,
                       save_path      : str | Path,
                       fold           : int,
                       acc_steps      : int = ACC_STEPS,
                       baseline_csv   : str = ""):

    device       = next(model.parameters()).device
    criterion_tr = ImageFlowNetLoss().to(device)
    lpips_val    = LPIPS(net='vgg').to(device).eval()

    save_path = Path(save_path)
    baseline  = load_baseline_metrics(fold, baseline_csv)
    selector  = BestCheckpointSelector(delta=0.05, baseline=baseline)

    ckpt_dir = save_path / f"fold_{fold}/ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    h_tr, h_val = [], []
    h_psnr, h_ssim, h_lpips, h_mse = [], [], [], []

    for epoch in range(1, num_epochs+1):

        p_full = min(1.0, 0.3 + 0.7 * (epoch - 1) / 39)   # E1 = 0.30 → E40 = 1.00
        train_mse_sum = 0.0
        nsamp_train   = 0

        # -------------------- TRAIN ----------------------------------------
        model.train(); running = 0; step = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(train_loader, desc=f'E{epoch:02d}[train]'):
            x   = batch['input'].float().to(device)
            tgt = batch['ground_truth'].float().to(device)
            mid = to_5d(batch["latent_target"].float().to(device))
            x, tgt = squeeze_early(x), squeeze_early(tgt)

            mid = guide_dropout(mid, p_full=p_full)

            pred, drift, early_lat, mid_lat = model(x, t=torch.tensor(1.0, device=device), mid_pet=mid)
            loss = criterion_tr(pred, tgt, drift, early_lat, mid_lat) / acc_steps
            loss.backward()

            mse_b = F.mse_loss(pred, tgt).item() * x.size(0)
            train_mse_sum += mse_b
            nsamp_train   += x.size(0)

            step += 1
            if step % acc_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                warmup_sched.step()

            running += loss.item() * acc_steps

        train_loss = running / len(train_loader)
        train_mse  = train_mse_sum / nsamp_train

        # ============= Validation ========================
        model.eval()
        psnr = ssim = lp = mse_sum = val_loss_sum = 0; nsamp = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'E{epoch:02d}[val]'):
                x   = batch['input'].float().to(device)
                tgt = batch['ground_truth'].float().to(device)
                x, tgt = squeeze_early(x), squeeze_early(tgt)

                pred, drift, early_lat, mid_lat = model(
                    x, t=torch.tensor(1.0, device=device), mid_pet=None)
                bs = x.size(0); nsamp += bs

                val_loss_sum += criterion_tr(pred, tgt, drift, early_lat, mid_lat).item() * bs
                psnr += calc_psnr(pred, tgt, data_range=1.).item() * bs
                ssim += ssim_fn(pred.clamp(0,1), tgt.clamp(0,1), data_range=1.).item() * bs
                lp   += lpips_val(pred.expand(-1,3,-1,-1),
                                  tgt.expand(-1,3,-1,-1), normalize=True).mean().item() * bs
                mse_sum += F.mse_loss(pred, tgt).item() * bs

        psnr /= nsamp; ssim /= nsamp; lp /= nsamp
        val_loss  = val_loss_sum / nsamp
        train_mse = train_mse_sum / nsamp_train
        val_mse   = mse_sum / nsamp

        print(f"E{epoch:02d} ▸ "
              f"train-L {train_loss:.4f} | "
              f"valid-L {val_loss:.4f} | "
              f"train-MSE {train_mse:.5f} | "
              f"val-MSE {val_mse:.5f} | "
              f"PSNR {psnr:.2f}  SSIM {ssim:.4f}  LPIPS {lp:.4f}")

        h_tr.append(train_loss)
        h_val.append(val_loss)
        h_psnr.append(psnr)
        h_ssim.append(ssim)
        h_lpips.append(lp)
        h_mse.append(val_mse)

        if PLOT_METRIC:
            update_and_plot_metrics(epoch, h_tr, h_val, h_psnr, h_ssim, h_lpips, h_mse, save_path)

        improved = selector.check_improve(psnr, ssim, lp)

        if improved:
            ckpt_name = (
                f"fold{fold}_epoch{epoch:03d}"
                f"_psnr{psnr:.2f}_ssim{ssim:.4f}_lpips{lp:.4f}.pth"
            )
            ckpt_path = ckpt_dir / ckpt_name
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict()}, ckpt_path)
            write_ckpt_pointer(ckpt_dir, "best", {
                "checkpoint": ckpt_name, "epoch": epoch,
                "psnr": psnr, "ssim": ssim, "lpips": lp,
                "criterion": selector.criterion})

            print(Fore.GREEN + "  ↑ new best – model saved" + Style.RESET_ALL)

        if epoch == num_epochs:
            last_name = (
                f"fold{fold}_epoch{epoch:03d}"
                f"_psnr{psnr:.2f}_ssim{ssim:.4f}_lpips{lp:.4f}_last.pth"
            )
            last_path = ckpt_dir / last_name
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict()}, last_path)
            write_ckpt_pointer(ckpt_dir, "last", {
                "checkpoint": last_name, "epoch": epoch,
                "psnr": psnr, "ssim": ssim, "lpips": lp,
                "criterion": "final epoch"})

            print(Fore.CYAN + f"  → final-epoch model saved to {last_path.name}" + Style.RESET_ALL)


# ═══════════════════════ Main ══════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, choices=["sde","ode"], default="sde")
    parser.add_argument("--data-root",  type=str, required=True,
                        help="Path to the preprocessed dataset")
    parser.add_argument("--save-dir",   type=str, default="./runs",
                        help="Parent directory for training outputs")
    parser.add_argument("--baseline-csv", type=str, default="",
                        help="Optional baseline-metrics CSV; checkpoints are saved only if they beat it")
    parser.add_argument("--seeds",      type=int, nargs="+", default=[42],
                        help="Random seed(s); each seed runs all requested folds. "
                             "The paper used seeds 42, 2025, 31415, 1105, and 806.")
    parser.add_argument("--folds",      type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--epochs",     type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--strategy",   type=str, default="stack",
                        choices=["auto","concat","stack"])
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("configs/stratified_5fold_all.json", "r") as f:
        split_all = json.load(f)

    cfg = dict(
        input_channels=1,
        n_stages=4,
        features_per_stage=[32,64,128,256],
        conv_op=nn.Conv2d,
        kernel_sizes=[(3,3)]*4,
        strides=[(1,1),(2,2),(2,2),(2,2)],
        n_conv_per_stage=[2]*4,
        num_classes=1,
        n_conv_per_stage_decoder=[2,2,2],
        deep_supervision=False,
        conv_bias=False,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=dict(eps=1e-5, affine=True),
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=dict(inplace=False),
    )

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        print(Fore.GREEN + f"[Seed] {seed} fixed" + Style.RESET_ALL)

    def make_loader(split: str, fold: int):
        ds = Early2LateWithLatentDataset(
            pet_dir=args.data_root,
            dataset_type=split,
            fold=fold,
            resize_image_size=(128, 128),
            split=split_all)
        return DataLoader(
            ds, batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=4, pin_memory=True
        )

    def run_one_fold(fold: int, base_path: str):
        print(Fore.CYAN + f"\n=== Train Fold {fold} ===" + Style.RESET_ALL)

        if args.model == "sde":
            model = build_lkmunet_sde_film(cfg, strategy=args.strategy).to(DEVICE)
        else:
            model = build_lkmunet_ode_film(cfg, strategy=args.strategy).to(DEVICE)
        apply_gradient_checkpoint(model)

        train_loader = make_loader("train", fold)
        valid_loader = make_loader("val",   fold)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

        warmup_fn    = lambda it: min(1., it / 2_000)
        warmup_sched = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_dir  = Path(base_path) / f"LKMU{args.model.upper()}_F{fold}_{timestamp}"   # LKMUSDE_ / LKMUODE_
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_and_validate(
            model, train_loader, valid_loader,
            optimizer, warmup_sched,
            num_epochs=args.epochs,
            save_path=fold_dir,
            fold=fold,
            acc_steps=ACC_STEPS,
            baseline_csv=args.baseline_csv,
        )

        torch.cuda.empty_cache(); gc.collect()

    # ── Main loop ──
    # The seed is set once per seed value and is not reset between folds.
    # The paper used seeds 42, 2025, 31415, 1105, and 806.
    for sd in args.seeds:
        set_seed(sd)
        base_path = str(Path(args.save_dir) / f"runs_seed_LKMUNet_{args.model.upper()}{sd:05d}")
        for f in args.folds:
            run_one_fold(f, base_path)

    print(Fore.GREEN + "All seeds & folds finished!" + Style.RESET_ALL)