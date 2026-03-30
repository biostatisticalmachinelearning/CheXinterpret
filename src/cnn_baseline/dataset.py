"""CheXpert Plus dataset for CNN training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CNNConfig, CNNDataConfig

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(image_size: int, augmentation_level: str, is_train: bool) -> transforms.Compose:
    """
    Returns torchvision transform pipeline.

    Train augmentation levels:
      light  — horizontal flip + mild brightness/contrast jitter
      medium — above + rotation ±10° + small translation
      heavy  — above + random erasing + elastic distortion

    Validation always: Resize → CenterCrop → ToTensor → Normalize.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not is_train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.05)),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])

    if augmentation_level == "none":
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.05)),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])

    base = [
        transforms.Resize(int(image_size * 1.10)),
        transforms.RandomCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ]

    if augmentation_level in ("medium", "heavy"):
        base += [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]

    if augmentation_level == "heavy":
        base += [
            transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        ]

    base += [transforms.ToTensor(), normalize]

    if augmentation_level == "heavy":
        base += [transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))]

    return transforms.Compose(base)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CheXpertCNNDataset(Dataset):
    """Multi-label chest X-ray dataset from a pre-built manifest CSV."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        pathology_cols: list[str],
        transform: transforms.Compose,
        sensitive_attrs: list[str],
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.pathology_cols = pathology_cols
        self.transform = transform
        self.sensitive_attrs = sensitive_attrs

        # Pre-extract label matrix as float32 array for fast indexing
        self._labels = self.manifest[pathology_cols].fillna(0.0).values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[idx]

        # Load image — CheXpert X-rays are grayscale PNGs
        img = Image.open(row["image_path"]).convert("RGB")
        img_tensor: torch.Tensor = self.transform(img)

        labels = torch.from_numpy(self._labels[idx])

        attrs = {attr: str(row.get(attr, "unknown")) for attr in self.sensitive_attrs}

        return {"image": img_tensor, "labels": labels, "attrs": attrs, "idx": idx}


# ── Positive-weight computation ───────────────────────────────────────────────

def compute_pos_weight(
    manifest: pd.DataFrame,
    pathology_cols: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Per-pathology pos_weight = n_negative / n_positive, clamped to [0.5, 50.0].
    NaN-only columns (no positives) get weight 1.0.
    """
    weights = []
    for col in pathology_cols:
        vals = manifest[col].dropna()
        n_pos = (vals == 1.0).sum()
        n_neg = (vals == 0.0).sum()
        if n_pos == 0:
            weights.append(1.0)
        else:
            w = float(n_neg) / float(n_pos)
            weights.append(float(np.clip(w, 0.5, 50.0)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    manifest: pd.DataFrame,
    cfg: CNNConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, valid_loader)."""
    dc = cfg.data
    tc = cfg.train

    train_df = manifest[manifest[dc.split_col] == "train"].copy()
    valid_df = manifest[manifest[dc.split_col] == "valid"].copy()

    if dc.max_rows_per_split is not None:
        train_df = train_df.sample(min(dc.max_rows_per_split, len(train_df)), random_state=42).reset_index(drop=True)
        valid_df = valid_df.sample(min(dc.max_rows_per_split, len(valid_df)), random_state=42).reset_index(drop=True)

    logger.info("Dataset sizes — train: %d, valid: %d", len(train_df), len(valid_df))

    train_ds = CheXpertCNNDataset(
        train_df,
        dc.pathology_cols,
        get_transforms(dc.image_size, dc.augmentation_level, is_train=True),
        dc.sensitive_attrs,
    )
    valid_ds = CheXpertCNNDataset(
        valid_df,
        dc.pathology_cols,
        get_transforms(dc.image_size, dc.augmentation_level, is_train=False),
        dc.sensitive_attrs,
    )

    pin = dc.pin_memory and device.type == "cuda"
    kwargs = dict(
        num_workers=dc.num_workers,
        pin_memory=pin,
        persistent_workers=(dc.num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=tc.batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=tc.batch_size * 2,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )

    return train_loader, valid_loader
