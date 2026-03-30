"""Configuration dataclasses for the CNN baseline module."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chex_sae_fairness.constants import PATHOLOGY_14

SENSITIVE_ATTRS = ["sex", "age_group", "race", "insurance_type"]


@dataclass(slots=True)
class CNNPathsConfig:
    manifest_csv: str
    image_root: str
    output_root: str = "cnn_baseline/outputs/default"


@dataclass(slots=True)
class CNNDataConfig:
    split_col: str = "split"
    pathology_cols: list[str] = field(default_factory=lambda: list(PATHOLOGY_14))
    sensitive_attrs: list[str] = field(default_factory=lambda: list(SENSITIVE_ATTRS))
    image_size: int = 320
    num_workers: int = 8
    pin_memory: bool = True
    augmentation_level: str = "medium"  # light | medium | heavy
    max_rows_per_split: int | None = None  # cap per split for fast sweeps; null = use all


@dataclass(slots=True)
class CNNTrainConfig:
    architecture: str = "densenet121"  # densenet121|resnet50|efficientnet_b4|convnext_small
    pretrained: bool = True
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 7
    warmup_epochs: int = 3
    amp: bool = True
    grad_clip: float = 1.0
    pos_weight_mode: str = "auto"  # auto | none


@dataclass(slots=True)
class CNNSweepConfig:
    n_trials: int = 50
    timeout_seconds: int | None = None
    sampler: str = "tpe"  # tpe | random
    architectures: list[str] = field(
        default_factory=lambda: ["densenet121", "resnet50", "efficientnet_b4", "convnext_small"]
    )
    lr_low: float = 1e-5
    lr_high: float = 1e-3
    batch_sizes: list[int] = field(default_factory=lambda: [16, 32, 64])
    weight_decays: list[float] = field(default_factory=lambda: [1e-6, 1e-5, 1e-4, 1e-3])
    augmentation_levels: list[str] = field(default_factory=lambda: ["light", "medium", "heavy"])
    image_sizes: list[int] = field(default_factory=lambda: [224, 320, 448])


@dataclass(slots=True)
class CNNConfig:
    seed: int
    paths: CNNPathsConfig
    data: CNNDataConfig
    train: CNNTrainConfig
    sweep: CNNSweepConfig

    # ── factories ────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CNNConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())

        def _build(dc, d):
            """Recursively build a dataclass from a dict."""
            if d is None:
                d = {}
            fields = {f.name: f for f in dc.__dataclass_fields__.values()}
            kwargs = {}
            for k, v in d.items():
                if k not in fields:
                    continue
                ft = fields[k].type
                # Nested dataclasses
                if hasattr(ft, "__dataclass_fields__"):
                    kwargs[k] = _build(ft, v)
                else:
                    kwargs[k] = v
            return dc(**{**{f.name: f.default if f.default is not dataclass_field_missing
                            else (f.default_factory() if f.default_factory is not dataclass_field_missing
                                  else None)
                            for f in fields.values()}, **kwargs})

        # Use simpler direct construction
        paths = CNNPathsConfig(**raw.get("paths", {}))
        data = CNNDataConfig(**{k: v for k, v in raw.get("data", {}).items()
                                if k in CNNDataConfig.__dataclass_fields__})
        train = CNNTrainConfig(**{k: v for k, v in raw.get("train", {}).items()
                                  if k in CNNTrainConfig.__dataclass_fields__})
        sweep = CNNSweepConfig(**{k: v for k, v in raw.get("sweep", {}).items()
                                  if k in CNNSweepConfig.__dataclass_fields__})
        return cls(seed=raw.get("seed", 42), paths=paths, data=data, train=train, sweep=sweep)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def output_root(self) -> Path:
        return Path(self.paths.output_root)

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "best_model.pt"

    @property
    def train_log_path(self) -> Path:
        return self.output_root / "train_log.csv"


# sentinel to handle dataclass field defaults in from_yaml
import dataclasses as _dc
dataclass_field_missing = _dc.MISSING
