from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from chex_sae_fairness.config import ExperimentConfig


@dataclass(slots=True)
class ManifestBuildResult:
    manifest: pd.DataFrame
    dropped_rows: int


def build_manifest(cfg: ExperimentConfig) -> ManifestBuildResult:
    frame = pd.read_csv(cfg.paths.metadata_csv)
    _ensure_columns(frame, _required_columns(cfg))

    frame = frame.loc[frame[cfg.schema.split_col].isin(cfg.data.allowed_splits)].copy()
    frame[cfg.schema.age_col] = pd.to_numeric(frame[cfg.schema.age_col], errors="coerce")
    frame = frame.loc[
        (frame[cfg.schema.age_col] >= cfg.data.min_age) & (frame[cfg.schema.age_col] <= cfg.data.max_age)
    ].copy()

    frame = _apply_uncertain_policy(frame, cfg)

    image_root = Path(cfg.paths.image_root)
    frame["image_path"] = frame[cfg.schema.image_path_col].apply(
        lambda p: str((image_root / str(p)).resolve())
    )

    age_bin_labels = _age_bin_labels(cfg.data.age_bins)
    frame["age_group"] = pd.cut(
        frame[cfg.schema.age_col],
        bins=cfg.data.age_bins,
        right=False,
        labels=age_bin_labels,
        include_lowest=True,
    ).astype(str)

    output_cols: list[str] = [
        "image_path",
        cfg.schema.split_col,
        cfg.schema.patient_id_col,
        cfg.schema.age_col,
        "age_group",
    ]
    output_cols.extend(cfg.schema.pathology_cols)
    for col in cfg.schema.metadata_cols:
        if col not in output_cols:
            output_cols.append(col)

    before_drop = len(frame)
    frame = frame.dropna(subset=cfg.schema.pathology_cols + ["image_path"]).reset_index(drop=True)
    dropped_rows = before_drop - len(frame)
    return ManifestBuildResult(manifest=frame.loc[:, output_cols], dropped_rows=dropped_rows)


def save_manifest(manifest: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


class CheXImageDataset(Dataset[dict[str, object]]):
    def __init__(self, manifest: pd.DataFrame) -> None:
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": image,
            "index": int(index),
        }


def split_manifest(manifest: pd.DataFrame, split_col: str, split_name: str) -> pd.DataFrame:
    return manifest.loc[manifest[split_col] == split_name].reset_index(drop=True)


def materialize_targets(
    manifest: pd.DataFrame,
    pathology_cols: Iterable[str],
    metadata_cols: Iterable[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    pathologies = manifest.loc[:, list(pathology_cols)].astype(np.float32).to_numpy()
    metadata = manifest.loc[:, list(metadata_cols)].copy()
    return pathologies, metadata


def _required_columns(cfg: ExperimentConfig) -> list[str]:
    cols = {
        cfg.schema.image_path_col,
        cfg.schema.split_col,
        cfg.schema.patient_id_col,
        cfg.schema.age_col,
        *cfg.schema.pathology_cols,
        *cfg.schema.metadata_cols,
    }
    return sorted(cols)


def _ensure_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Metadata CSV missing required columns: {missing_str}")


def _apply_uncertain_policy(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    uncertain_value = -1
    cols = cfg.schema.pathology_cols
    policy = cfg.data.uncertain_label_policy.lower()

    if policy == "zero":
        frame.loc[:, cols] = frame.loc[:, cols].replace(uncertain_value, 0)
    elif policy == "one":
        frame.loc[:, cols] = frame.loc[:, cols].replace(uncertain_value, 1)
    elif policy == "ignore":
        frame.loc[:, cols] = frame.loc[:, cols].replace(uncertain_value, np.nan)
    else:
        raise ValueError(f"Unknown uncertain label policy: {cfg.data.uncertain_label_policy}")

    frame.loc[:, cols] = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return frame


def _age_bin_labels(age_bins: list[int]) -> list[str]:
    return [f"{age_bins[i]}-{age_bins[i + 1] - 1}" for i in range(len(age_bins) - 1)]
