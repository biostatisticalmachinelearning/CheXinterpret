from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from chex_sae_fairness.config import ExperimentConfig


def create_timestamped_pipeline_dir(
    base_output_root: str | Path,
    pipeline_name: str,
    run_name: str | None = None,
) -> Path:
    base = Path(base_output_root).expanduser() / "publication" / pipeline_name
    base.mkdir(parents=True, exist_ok=True)

    stem = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / stem
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    suffix = 1
    while True:
        candidate = base / f"{stem}_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        suffix += 1


def write_experiment_config(cfg: ExperimentConfig, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(cfg), handle, sort_keys=False)
    return out


def with_output_root(cfg: ExperimentConfig, output_root: str | Path) -> ExperimentConfig:
    return replace(cfg, paths=replace(cfg.paths, output_root=str(Path(output_root).resolve())))


def load_prediction_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML at {path} must contain an object.")
    return payload
