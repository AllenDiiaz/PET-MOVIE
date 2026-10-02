import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import json
import random
import pandas as pd
from scipy.ndimage import rotate

class Early2LateWithLatentDataset(Dataset):
    def __init__(self, pet_dir, dataset_type="train", fold=0,
                 split=None, resize_image_size=(128, 128),
                 json_path="configs/subjects_data_with_abeta.json",
                 stats_csv="configs/subject_stats_Early2Late_withLatent_FULL_0729.csv",
                 load_latent=None):
        super().__init__()
        self.pet_dir = pet_dir
        self.dataset_type = dataset_type
        # Inference uses the early frame only; the test split skips latent_target (mid frames) by default
        self.load_latent = (dataset_type != "test") if load_latent is None else load_latent
        self.fold = fold
        self.resize_image_size = resize_image_size
        self.split = split or {}

        with open(json_path, "r") as f:
            self.subject_metadata = json.load(f)

        self.stats = self._load_stats(stats_csv)

        if str(fold) not in self.split:
            raise ValueError(f"Fold {fold} not found in split")

        val_index = (fold + len(self.split) - 3) % len(self.split)
        test_index = (fold + len(self.split) - 2) % len(self.split)
        self.val_subjects = set(self.split[str(val_index)])
        self.test_subjects = set(self.split[str(test_index)])

        self.data_files = self._prepare_files()

    def _load_stats(self, stats_csv):
        df = pd.read_csv(stats_csv)
        stats = {}
        for _, row in df.iterrows():
            stats[(row["Type"], row["Subject"])] = {
                "min": row["Min Value"],
                "max": row["Max Value"]
            }
        return stats

    def _normalize(self, data, subject, reference_key="data"):
        key = (reference_key, subject)
        if key not in self.stats:
            return data
        stat = self.stats[key]
        min_val, max_val = stat["min"], stat["max"]
        if max_val - min_val <= 1e-5:
            return np.zeros_like(data)
        return np.clip((data - min_val) / (max_val - min_val), 0, 1)

    def _prepare_files(self):
        files = []
        for group_num, subjects in self.split.items():
            data_dir = os.path.join(self.pet_dir, f"group{group_num}", "data")
            if not os.path.exists(data_dir):
                continue
            for fname in os.listdir(data_dir):
                subject = fname.split('_')[0]
                if self.dataset_type == "train" and subject not in self.val_subjects and subject not in self.test_subjects:
                    files.append((group_num, fname))
                elif self.dataset_type == "val" and subject in self.val_subjects:
                    files.append((group_num, fname))
                elif self.dataset_type == "test" and subject in self.test_subjects:
                    files.append((group_num, fname))
        return files

    def __len__(self):
        return len(self.data_files)

    def _rotate_small_angle(self, arr, angle):
        return rotate(arr, angle=angle, reshape=False, order=1, mode='reflect')

    def _augment(self, early, gt, latent):
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)          # random rotation in [-10°, 10°]
            early = self._rotate_small_angle(early, angle)
            gt = self._rotate_small_angle(gt, angle)
            if latent is not None:
                latent = np.stack([self._rotate_small_angle(latent[..., i], angle) for i in range(latent.shape[-1])], axis=-1)
        return early, gt, latent

    def __getitem__(self, idx):
        group_num, fname = self.data_files[idx]
        subject = fname.split('_')[0]
        slice_idx = int(fname.split('_')[-1].split('.')[0])

        base_path = os.path.join(self.pet_dir, f"group{group_num}")
        early_path = os.path.join(base_path, "data", fname)
        gt_path = os.path.join(base_path, "ground_truth", fname)
        latent_path = os.path.join(base_path, "latent_target", fname)

        early = np.load(early_path)
        gt = np.load(gt_path)
        latent = np.load(latent_path) if self.load_latent else None

        early = cv2.resize(early.squeeze(), self.resize_image_size)[..., np.newaxis]
        gt = cv2.resize(gt, self.resize_image_size)

        if latent is not None:
            if latent.ndim == 2:
                latent = latent[..., np.newaxis]
            latent = np.stack([cv2.resize(latent[..., i], self.resize_image_size) for i in range(latent.shape[-1])], axis=-1)

        if self.dataset_type == "train":
            early, gt, latent = self._augment(early, gt, latent)

        early = self._normalize(early, subject, "data")
        gt = self._normalize(gt, subject, "data")

        early = torch.tensor(early.transpose(2, 0, 1).copy(), dtype=torch.float32).unsqueeze(0)   # [1, 1, H, W]
        gt = torch.tensor(gt.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)               # [1, 1, H, W]

        metadata = self.subject_metadata.get(subject.replace("Subject", "patient"), {})
        abeta_value = metadata.get("Amyloid beta", -1)
        abeta_value = int(abeta_value) if isinstance(abeta_value, (int, float)) and not pd.isna(abeta_value) else -1

        sample = {
            "input": early,
            "ground_truth": gt,
            "subject": subject,
            "slice": slice_idx,
            "diagnosis": metadata.get("Diagnosis", "Unknown"),
            "instrument": metadata.get("Instrument", "Unknown"),
            "abeta": abeta_value
        }

        # default_collate cannot handle None, so omit the key when latent is not loaded
        if latent is not None:
            latent = np.stack([self._normalize(latent[..., i], subject, "data") for i in range(latent.shape[-1])], axis=-1)
            latent = torch.tensor(latent.copy(), dtype=torch.float32)                             # [H, W, M]
            sample["latent_target"] = latent.permute(2, 0, 1).unsqueeze(1).unsqueeze(0)           # [1, M, 1, H, W]

        return sample