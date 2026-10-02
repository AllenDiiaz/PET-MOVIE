# inference.py - evaluate LKMUNet-SDE/ODE-FiLM on the test split of all five folds
import warnings
import torch.nn as nn
from pathlib import Path
from typing import List, Dict
import torch.nn.functional as F
import torch
from colorama import Fore, Style
from lpips import LPIPS
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr_fn,
    structural_similarity_index_measure as ssim_fn,
)
from tqdm import tqdm
import numpy as np
import pandas as pd
from models.sde_film import build_lkmunet_sde_film
from models.ode_film import build_lkmunet_ode_film
from models.lkmunet import apply_gradient_checkpoint
from datasets.early2late_dataset import Early2LateWithLatentDataset
import json

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["sde", "ode"], default="ode",
                        help="Model variant: sde or ode")
    parser.add_argument("--ckpt-dirs", type=str, nargs=5, required=True,
                        help="Run directories for the five folds, in order F0-F4")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to the preprocessed dataset")
    parser.add_argument("--output-prefix", type=str, default="inference",
                        help="Filename prefix for the output CSV files")
    args = parser.parse_args()

    TARGET_DIRS = args.ckpt_dirs
    CKPT_DIRS = [
        Path(dir_str) / f"fold_{i}" / "ckpt"
        for i, dir_str in enumerate(TARGET_DIRS)
    ]

    DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = dict(
        input_channels=1,
        n_stages=4,
        features_per_stage=[32, 64, 128, 256],
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
    BATCH_SIZE = 8
    DATA_ROOT  = args.data_root
    with open("configs/stratified_5fold_all.json", "r") as f:
        split_all = json.load(f)
    warnings.filterwarnings("ignore")

    # ---------- 1. Locate the best checkpoint ------------------------------
    def find_best_ckpt(ckpt_dir: Path) -> Path:
        """Return the checkpoint recorded in best.json, else the target of best.pth."""
        meta = ckpt_dir / "best.json"
        if meta.is_file():
            ckpt = ckpt_dir / json.loads(meta.read_text())["checkpoint"]
            if not ckpt.is_file():
                raise FileNotFoundError(f"{meta} points to a missing file: {ckpt}")
            return ckpt.resolve()

        bp = ckpt_dir / "best.pth"
        if bp.is_file():
            return bp.resolve()

        raise FileNotFoundError(f"Neither best.json nor a valid best.pth found in {ckpt_dir}")

    # ---------- 2. DataLoader ------------------------------------------------
    def make_loader(fold: int) -> DataLoader:
        ds = Early2LateWithLatentDataset(
            pet_dir=DATA_ROOT,
            dataset_type="test",
            fold=fold,
            resize_image_size=(128,128),
            split=split_all
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    # ---------- 3. Load the model --------------------------------------------
    def load_model(ckpt_path: Path):
        if args.model == "sde":
            model = build_lkmunet_sde_film(cfg, strategy="stack").to(DEVICE)
        else:
            model = build_lkmunet_ode_film(cfg, strategy="stack").to(DEVICE)
        apply_gradient_checkpoint(model)
        st = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(st["model"], strict=True)
        model.eval()
        return model

    # ---------- 4. Helpers ---------------------------------------------------
    def squeeze_early(x):
        return x.squeeze(2) if (x.ndim == 5 and x.size(2) == 1) else x

    # ---------- 5. Inference and per-subject metrics -----------------------
    subject_metrics: Dict[str, Dict[str, List[float]]] = {}

    for fold, ckpt_dir in enumerate(CKPT_DIRS):
        best_ckpt = find_best_ckpt(ckpt_dir)
        print(Fore.GREEN + f"[Fold {fold}] load {best_ckpt}" + Style.RESET_ALL)
        model  = load_model(best_ckpt)
        loader = make_loader(fold)

        lpips_fn = LPIPS(net="vgg").to(DEVICE).eval()

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Fold{fold} inf"):
                subs  = batch["subject"]
                early = squeeze_early(batch["input"]).to(DEVICE)
                # inference uses the early frame only (no mid-frame guidance)
                preds = model(early, t=torch.tensor(1.0,device=DEVICE), mid_pet=None)[0].clamp(0,1)

                for i, sid in enumerate(subs):
                    p = preds[i]

                    gt  = squeeze_early(batch["ground_truth"]).to(DEVICE)[i]
                    ps  = psnr_fn(p.unsqueeze(0), gt.unsqueeze(0), data_range=1.).item()
                    ss  = ssim_fn(p.unsqueeze(0), gt.unsqueeze(0), data_range=1.).item()
                    mse = F.mse_loss(p, gt, reduction='mean').item()
                    lp  = lpips_fn(
                        p.unsqueeze(0).expand(1,3,*p.shape[-2:]),
                        gt.unsqueeze(0).expand(1,3,*gt.shape[-2:]),
                        normalize=True
                    ).item()

                    subject_metrics.setdefault(sid, {
                        "psnr": [], "ssim": [], "lpips": [], "mse": []
                    })
                    subject_metrics[sid]["psnr"].append(ps)
                    subject_metrics[sid]["ssim"].append(ss)
                    subject_metrics[sid]["lpips"].append(lp)
                    subject_metrics[sid]["mse"].append(mse)

    # ---------- 6. Mean ± std and summary ------------------------------------
    rows = []
    for sid, m in subject_metrics.items():
        rows.append({
            "subject"    : sid,
            "psnr_mean"  : np.mean(m["psnr"]),
            "psnr_std"   : np.std(m["psnr"], ddof=0),
            "ssim_mean"  : np.mean(m["ssim"]),
            "ssim_std"   : np.std(m["ssim"], ddof=0),
            "lpips_mean" : np.mean(m["lpips"]),
            "lpips_std"  : np.std(m["lpips"], ddof=0),
            "mse_mean"   : np.mean(m["mse"]),
            "mse_std"    : np.std(m["mse"], ddof=0),
        })
    df = pd.DataFrame(rows).set_index("subject")

    print("\n==== Per-subject mean ± std over slices (each subject is tested in exactly one fold) ====")
    print(df)

    overall = {
        "psnr_mean" : df["psnr_mean"].mean(),
        "psnr_std"  : df["psnr_mean"].std(),
        "ssim_mean" : df["ssim_mean"].mean(),
        "ssim_std"  : df["ssim_mean"].std(),
        "lpips_mean": df["lpips_mean"].mean(),
        "lpips_std" : df["lpips_mean"].std(),
        "mse_mean"  : df["mse_mean"].mean(),
        "mse_std"   : df["mse_mean"].std(),
    }
    print("\n==== Overall across subjects ====")
    print(pd.Series(overall))

    # ---------- 7. Save CSV files ---------------------------------------
    df.reset_index().to_csv(f"{args.output_prefix}_subject_metrics.csv", index=False)
    print(Fore.GREEN + f"Saved {args.output_prefix}_subject_metrics.csv" + Style.RESET_ALL)

    pd.DataFrame([overall]).to_csv(f"{args.output_prefix}_overall_metrics.csv", index=False)
    print(Fore.GREEN + f"Saved {args.output_prefix}_overall_metrics.csv" + Style.RESET_ALL)