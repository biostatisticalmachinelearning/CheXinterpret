from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chex_sae_fairness.publication.common import read_yaml


@dataclass(slots=True)
class CoreSpec:
    sweep_config_path: str = "configs/sae_sweep.yaml"
    run_name: str | None = None
    force_recompute_features: bool = False
    debias_ablation_modes: list[str] = field(default_factory=lambda: ["train_and_test", "test_only", "train_only"])
    debias_ablation_strengths: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5])


@dataclass(slots=True)
class SupplementSpec:
    run_name: str | None = None
    seeds: list[int] = field(default_factory=lambda: [13, 17, 23])
    uncertain_policies: list[str] = field(default_factory=lambda: ["zero", "one", "ignore"])
    debias_modes: list[str] = field(default_factory=lambda: ["train_and_test", "test_only", "train_only"])
    debias_strengths: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5])
    age_bin_sets: list[list[int]] = field(
        default_factory=lambda: [[18, 35, 50, 65, 80, 120], [18, 40, 60, 80, 120], [18, 45, 60, 75, 120]]
    )
    permutation_repeats: int = 5
    missing_metadata_fractions: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    baseline_methods: list[str] = field(
        default_factory=lambda: [
            "raw",
            "pca",
            "nmf",
            "supervised_bottleneck",
            "group_reweighted",
            "group_threshold",
            "equalized_odds",
            "adversarial_debiasing",
        ]
    )
    fairness_thresholds: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    external_config_paths: list[str] = field(default_factory=list)
    human_eval_csv: str | None = None
    force_recompute_features: bool = False


@dataclass(slots=True)
class PublicationSpec:
    core: CoreSpec = field(default_factory=CoreSpec)
    supplement: SupplementSpec = field(default_factory=SupplementSpec)

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "PublicationSpec":
        if path is None:
            return cls()
        payload = read_yaml(path)
        core_payload = payload.get("core", {})
        supplement_payload = payload.get("supplement", {})
        if not isinstance(core_payload, dict):
            raise ValueError("`core` section in publication config must be an object.")
        if not isinstance(supplement_payload, dict):
            raise ValueError("`supplement` section in publication config must be an object.")
        return cls(
            core=CoreSpec(**core_payload),
            supplement=SupplementSpec(**supplement_payload),
        )


def write_publication_spec_template(path: str | Path) -> Path:
    from dataclasses import asdict
    import yaml

    spec = PublicationSpec()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(spec), handle, sort_keys=False)
    return out
