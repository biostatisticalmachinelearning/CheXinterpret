from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import yaml


@dataclass(slots=True)
class PathsConfig:
    image_root: str
    metadata_csv: str
    output_root: str
    chexbert_labels_json: str | None = None


@dataclass(slots=True)
class SchemaConfig:
    image_path_col: str
    split_col: str
    patient_id_col: str
    age_col: str
    sex_col: str
    race_col: str
    pathology_cols: list[str] = field(default_factory=list)
    metadata_cols: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DataConfig:
    allowed_splits: list[str] = field(default_factory=lambda: ["train", "valid", "test"])
    validation_split_name: str = "valid"
    test_split_name: str = "test"
    uncertain_label_policy: str = "zero"
    min_age: int = 0
    max_age: int = 120
    age_bins: list[int] = field(default_factory=lambda: [0, 40, 60, 80, 120])
    allowed_views: list[str] = field(default_factory=list)
    max_rows_per_split: int | None = None


@dataclass(slots=True)
class FeatureConfig:
    model_name: str
    cache_dir: str | None = ".cache/huggingface"
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 4
    precision: str = "fp16"
    pooling: str = "mean"
    force_recompute: bool = False


@dataclass(slots=True)
class SAEConfig:
    latent_dim: int
    variant: str = "l1"
    topk_k: int = 64
    l1_lambda: float = 1e-4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 512
    epochs: int = 40
    eval_every_epochs: int = 5


@dataclass(slots=True)
class ProbeConfig:
    c_value: float = 1.0
    max_iter: int = 2000


@dataclass(slots=True)
class FairnessConfig:
    target_metric: str = "macro_auroc"
    bootstrap_samples: int = 200
    debias_strength: float = 1.0
    debias_mode: str = "train_and_test"
    threshold: float = 0.5


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    paths: PathsConfig
    schema: SchemaConfig
    data: DataConfig
    features: FeatureConfig
    sae: SAEConfig
    probes: ProbeConfig
    fairness: FairnessConfig

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ExperimentConfig":
        payload = _read_yaml(config_path)
        return cls(
            seed=payload["seed"],
            paths=PathsConfig(**payload["paths"]),
            schema=SchemaConfig(**payload["schema"]),
            data=DataConfig(**payload.get("data", {})),
            features=FeatureConfig(**payload["features"]),
            sae=SAEConfig(**payload["sae"]),
            probes=_parse_probe_config(payload.get("probes", {})),
            fairness=FairnessConfig(**payload.get("fairness", {})),
        )

    @property
    def output_root(self) -> Path:
        return Path(self.paths.output_root)

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "manifest.csv"

    @property
    def feature_path(self) -> Path:
        return self.output_root / "features.npz"

    @property
    def sae_checkpoint_path(self) -> Path:
        return self.output_root / "sae.pt"

    @property
    def study_metrics_path(self) -> Path:
        return self.output_root / "study_metrics.json"

    @property
    def study_predictions_path(self) -> Path:
        return self.output_root / "study_predictions.npz"

    def ensure_output_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)


def _read_yaml(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a YAML object.")
    return payload


def _parse_probe_config(raw: dict[str, Any] | None) -> ProbeConfig:
    payload = dict(raw or {})
    valid_keys = set(ProbeConfig.__dataclass_fields__.keys())
    unknown = sorted(set(payload.keys()) - valid_keys)
    if unknown:
        warnings.warn(
            "Ignoring unknown probe config keys: " + ", ".join(unknown),
            RuntimeWarning,
            stacklevel=2,
        )
        for key in unknown:
            payload.pop(key, None)
    return ProbeConfig(**payload)
