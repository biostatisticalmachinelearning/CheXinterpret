from __future__ import annotations

import json
import re
import zipfile
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from chex_sae_fairness.config import ExperimentConfig


@dataclass(slots=True)
class ManifestBuildResult:
    manifest: pd.DataFrame
    dropped_rows: int


def build_manifest(cfg: ExperimentConfig) -> ManifestBuildResult:
    frame = pd.read_csv(cfg.paths.metadata_csv, low_memory=False)
    frame = _apply_common_column_aliases(frame, cfg)

    frame = _maybe_merge_pathology_labels(frame, cfg)
    frame = _derive_split_column_if_missing(frame, cfg)
    frame = _derive_patient_id_if_missing(frame, cfg)

    _ensure_columns(frame, _required_columns(cfg))

    frame = frame.loc[frame[cfg.schema.split_col].isin(cfg.data.allowed_splits)].copy()
    frame[cfg.schema.age_col] = pd.to_numeric(frame[cfg.schema.age_col], errors="coerce")
    frame = frame.loc[
        (frame[cfg.schema.age_col] >= cfg.data.min_age) & (frame[cfg.schema.age_col] <= cfg.data.max_age)
    ].copy()

    frame = _apply_uncertain_policy(frame, cfg)

    image_root = Path(cfg.paths.image_root).resolve()
    frame["image_path"] = frame[cfg.schema.image_path_col].apply(
        lambda p: _resolve_png_image_path(str(p), image_root)
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

    if len(frame) == 0:
        _raise_zero_rows_error(image_root)

    return ManifestBuildResult(manifest=frame.loc[:, output_cols], dropped_rows=dropped_rows)


def save_manifest(manifest: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def audit_png_layout(cfg: ExperimentConfig, sample_size: int = 2000) -> dict[str, object]:
    frame = pd.read_csv(cfg.paths.metadata_csv, low_memory=False)
    frame = _apply_common_column_aliases(frame, cfg)

    if cfg.schema.image_path_col not in frame.columns:
        raise ValueError(
            f"Could not find image path column `{cfg.schema.image_path_col}` in metadata CSV."
        )

    path_series = frame[cfg.schema.image_path_col].dropna().astype(str)
    sampled = path_series.sample(min(sample_size, len(path_series)), random_state=13)

    image_root = Path(cfg.paths.image_root).resolve()
    resolved = sampled.apply(lambda p: _resolve_png_image_path(p, image_root))
    success_mask = resolved.notna()

    unresolved_examples = sampled.loc[~success_mask].head(10).tolist()
    return {
        "n_total_rows": int(len(frame)),
        "n_sampled": int(len(sampled)),
        "n_resolved": int(success_mask.sum()),
        "resolve_rate": float(success_mask.mean()) if len(sampled) > 0 else float("nan"),
        "unresolved_examples": unresolved_examples,
        "image_root": str(image_root),
        "has_train_dir": bool((image_root / "train").exists()),
        "has_png_train_dir": bool((image_root / "PNG" / "train").exists()),
        "png_chunk_zips_in_root": sorted([p.name for p in image_root.glob("png_chexpert_plus_chunk_*.zip")]),
        "png_chunk_zips_in_png_dir": sorted(
            [p.name for p in (image_root / "PNG").glob("png_chexpert_plus_chunk_*.zip")]
            if (image_root / "PNG").exists()
            else []
        ),
    }


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


def _apply_common_column_aliases(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    alias_candidates: dict[str, list[str]] = {
        cfg.schema.image_path_col: ["path_to_image", "Path", "path"],
        cfg.schema.split_col: ["split", "Split"],
        cfg.schema.patient_id_col: ["patient_id", "subject_id", "patient"],
        cfg.schema.age_col: ["patient_age", "age"],
        cfg.schema.sex_col: ["patient_sex", "sex", "gender"],
        cfg.schema.race_col: ["patient_race", "race"],
    }

    rename_map: dict[str, str] = {}
    for target, options in alias_candidates.items():
        if target in frame.columns:
            continue
        source = next((candidate for candidate in options if candidate in frame.columns), None)
        if source is not None and source != target:
            rename_map[source] = target

    if not rename_map:
        return frame
    return frame.rename(columns=rename_map)


def _ensure_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Metadata table missing required columns: {missing_str}")


def _maybe_merge_pathology_labels(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    missing_pathology = [col for col in cfg.schema.pathology_cols if col not in frame.columns]
    if not missing_pathology:
        return frame

    labels_path = cfg.paths.chexbert_labels_json
    if not labels_path:
        missing_str = ", ".join(missing_pathology)
        raise ValueError(
            "Pathology columns are missing from metadata CSV. "
            f"Missing: {missing_str}. Provide `paths.chexbert_labels_json` to merge labels."
        )

    labels = _read_label_table(Path(labels_path))

    label_path_col_candidates = [cfg.schema.image_path_col, "path_to_image", "Path"]
    label_path_col = next((c for c in label_path_col_candidates if c in labels.columns), None)
    if label_path_col is None:
        raise ValueError(
            "Could not find image path column in label file. "
            "Expected one of: " + ", ".join(label_path_col_candidates)
        )

    if label_path_col != cfg.schema.image_path_col:
        labels = labels.rename(columns={label_path_col: cfg.schema.image_path_col})

    overlap = [col for col in cfg.schema.pathology_cols if col in labels.columns]
    if len(overlap) != len(cfg.schema.pathology_cols):
        missing = [col for col in cfg.schema.pathology_cols if col not in labels.columns]
        raise ValueError(
            "Label file does not contain all requested pathology columns. "
            "Missing: " + ", ".join(missing)
        )

    labels = labels[[cfg.schema.image_path_col] + overlap].drop_duplicates(
        subset=[cfg.schema.image_path_col], keep="last"
    )

    merged = frame.merge(labels, on=cfg.schema.image_path_col, how="left", suffixes=("", "_label"))
    return merged


def _read_label_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                name for name in archive.namelist() if name.lower().endswith(".json")
            ]
            if not members:
                raise ValueError(f"No JSON label file found inside archive: {path}")
            target = next(
                (name for name in members if "findings_fixed" in name.lower()),
                members[0],
            )
            with archive.open(target) as handle:
                text = handle.read().decode("utf-8")
            return _read_label_table_from_text(text, f"{path}:{target}")

    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read()
        return _read_label_table_from_text(text, str(path))


def _read_label_table_from_text(text: str, source_name: str) -> pd.DataFrame:
    try:
        return pd.read_json(StringIO(text), lines=True)
    except ValueError:
        payload = json.loads(text)

    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        return pd.DataFrame(payload["data"])

    raise ValueError(
        f"Unsupported label file structure at {source_name}. "
        "Expected JSONL or a JSON list of records."
    )


def _derive_split_column_if_missing(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if cfg.schema.split_col in frame.columns:
        return frame

    derived = frame[cfg.schema.image_path_col].astype(str).apply(_derive_split_from_path)
    frame = frame.copy()
    frame[cfg.schema.split_col] = derived
    return frame


def _derive_patient_id_if_missing(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if cfg.schema.patient_id_col in frame.columns:
        return frame

    derived = frame[cfg.schema.image_path_col].astype(str).apply(_derive_patient_id_from_path)
    frame = frame.copy()
    frame[cfg.schema.patient_id_col] = derived
    return frame


def _derive_split_from_path(raw: str) -> str:
    parts = [part.lower() for part in Path(raw).parts]
    for part in parts:
        if part in {"train", "valid", "test"}:
            return part
    return "unknown"


def _derive_patient_id_from_path(raw: str) -> str:
    match = re.search(r"(patient\d+)", raw, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unknown_patient"


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


def _resolve_png_image_path(raw_path: str, image_root: Path) -> str | None:
    raw = raw_path.strip()
    if not raw or raw.lower() == "nan":
        return None

    input_path = Path(raw)
    parts = [part for part in input_path.parts if part not in {"", "."}]

    # If the metadata path is absolute, prefer it directly.
    if input_path.is_absolute() and input_path.exists():
        return str(input_path.resolve())

    variants: list[Path] = []
    base_rel = Path(*parts)
    variants.append(base_rel)

    # If paths include dataset prefixes (e.g., PNG/train/... or CheXpert-v1.0-small/train/...)
    # also try from the first split segment.
    split_index = next(
        (idx for idx, part in enumerate(parts) if part.lower() in {"train", "valid", "test"}),
        None,
    )
    if split_index is not None and split_index > 0:
        variants.append(Path(*parts[split_index:]))

    candidate_bases = [image_root, image_root / "PNG"]
    suffixes_to_try = []

    current_suffix = base_rel.suffix.lower()
    if current_suffix:
        suffixes_to_try.append(current_suffix)
    for suffix in [".jpg", ".png", ".jpeg"]:
        if suffix not in suffixes_to_try:
            suffixes_to_try.append(suffix)

    candidates: list[Path] = []
    for rel in variants:
        for base in candidate_bases:
            if rel.suffix:
                candidates.append(base / rel)
                for suffix in suffixes_to_try:
                    candidates.append((base / rel).with_suffix(suffix))
            else:
                for suffix in suffixes_to_try:
                    candidates.append((base / rel).with_suffix(suffix))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate.resolve())

    return None


def _raise_zero_rows_error(image_root: Path) -> None:
    png_dir = image_root / "PNG"
    root_zips = list(image_root.glob("png_chexpert_plus_chunk_*.zip"))
    png_zips = list(png_dir.glob("png_chexpert_plus_chunk_*.zip")) if png_dir.exists() else []

    hints: list[str] = []
    if root_zips or png_zips:
        hints.append(
            "Detected PNG chunk zip files. Extract `png_chexpert_plus_chunk_*.zip` before building manifest."
        )

    hints.append(
        "Expected relative image paths like `train/patient.../study.../view1_frontal.jpg` from `path_to_image`."
    )
    hints.append(
        "Set `paths.image_root` to a directory where either `train/` exists directly or `PNG/train/` exists."
    )

    raise ValueError("No rows remained after resolving PNG image paths. " + " ".join(hints))
