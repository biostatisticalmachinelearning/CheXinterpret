from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.pipeline import run_full_study
from chex_sae_fairness.publication.common import (
    create_timestamped_pipeline_dir,
    load_prediction_bundle,
    write_experiment_config,
)
from chex_sae_fairness.publication.figures import generate_core_publication_figures
from chex_sae_fairness.publication.repro import build_reproducibility_appendix
from chex_sae_fairness.publication.spec import CoreSpec
from chex_sae_fairness.publication.statistics import (
    attach_multiple_testing_corrections,
    paired_bootstrap_method_tests,
)
from chex_sae_fairness.publication.tables import (
    build_core_table_cohort,
    build_core_table_group_fairness,
    build_core_table_paired_tests,
    build_core_table_intervention_ablation,
    build_core_table_main_results,
    write_table,
)
from chex_sae_fairness.study_runner import run_comprehensive_study
from chex_sae_fairness.utils.io import read_json, write_json

logger = logging.getLogger(__name__)


def run_core_publication_pipeline(
    config_path: str,
    core_spec: CoreSpec,
) -> dict[str, Any]:
    base_cfg = ExperimentConfig.from_yaml(config_path)
    run_dir = create_timestamped_pipeline_dir(
        base_output_root=base_cfg.output_root,
        pipeline_name="core",
        run_name=core_spec.run_name,
    )
    logger.info("Core publication run dir: %s", run_dir)

    comprehensive_result = run_comprehensive_study(
        config_path=config_path,
        sweep_config_path=core_spec.sweep_config_path,
        run_name=run_dir.name,
        run_root=run_dir / "comprehensive",
        force_recompute_features=core_spec.force_recompute_features,
    )

    best_report_path = Path(str(comprehensive_result["best_study_report"]))
    best_report = read_json(best_report_path)
    best_cfg = ExperimentConfig.from_yaml(Path(str(comprehensive_result["run_root"])) / "configs" / "best_sae_config.yaml")
    best_prediction_path = best_report_path.parent / "study_predictions.npz"
    prediction_bundle = (
        load_prediction_bundle(best_prediction_path)
        if best_prediction_path.exists()
        else None
    )
    sweep_summary = pd.read_csv(comprehensive_result["sweep_summary_csv"])

    figures = generate_core_publication_figures(
        sweep_summary=sweep_summary,
        best_report=best_report,
        output_dir=run_dir / "figures",
    )

    table_paths: dict[str, dict[str, str]] = {}
    cohort_table = build_core_table_cohort(best_report, prediction_bundle)
    table_paths["table1_cohort"] = write_table(cohort_table, run_dir / "tables" / "table1_cohort")

    if prediction_bundle is not None:
        threshold = float(best_report.get("debiasing", {}).get("fairness_threshold", 0.5))
        main_results = build_core_table_main_results(
            report=best_report,
            prediction_bundle=prediction_bundle,
            threshold=threshold,
            bootstrap_samples=300,
        )
        table_paths["table2_main_results"] = write_table(
            main_results,
            run_dir / "tables" / "table2_main_results",
        )
        group_table = build_core_table_group_fairness(
            report=best_report,
            prediction_bundle=prediction_bundle,
            threshold=threshold,
            bootstrap_samples=300,
        )
        table_paths["table2c_group_fairness"] = write_table(
            group_table,
            run_dir / "tables" / "table2c_group_fairness",
        )
        paired = _build_paired_test_rows(
            prediction_bundle=prediction_bundle,
            threshold=threshold,
            bootstrap_samples=300,
        )
        paired = attach_multiple_testing_corrections(paired, p_key="p_value")
        paired_table = build_core_table_paired_tests(paired)
        table_paths["table2b_paired_tests"] = write_table(
            paired_table,
            run_dir / "tables" / "table2b_paired_tests",
        )

    ablations = _run_debias_ablations(
        best_config_path=Path(str(comprehensive_result["run_root"])) / "configs" / "best_sae_config.yaml",
        output_dir=run_dir / "debias_ablation",
        modes=core_spec.debias_ablation_modes,
        strengths=core_spec.debias_ablation_strengths,
    )
    ablation_table = build_core_table_intervention_ablation(ablations)
    table_paths["table3_intervention_ablation"] = write_table(
        ablation_table,
        run_dir / "tables" / "table3_intervention_ablation",
    )

    artifact = {
        "run_dir": str(run_dir.resolve()),
        "comprehensive_result": comprehensive_result,
        "figures": [str(path.resolve()) for path in figures],
        "tables": table_paths,
        "debias_ablation_results": ablations,
    }
    repro = build_reproducibility_appendix(
        cfg=best_cfg,
        report=best_report,
        extra={"pipeline": "core", "run_name": run_dir.name},
    )
    write_json(repro, run_dir / "reproducibility_appendix.json")
    artifact["reproducibility_appendix"] = str((run_dir / "reproducibility_appendix.json").resolve())
    write_json(artifact, run_dir / "core_pipeline_summary.json")
    return artifact


def _run_debias_ablations(
    best_config_path: Path,
    output_dir: Path,
    modes: list[str],
    strengths: list[float],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = ExperimentConfig.from_yaml(best_config_path)
    rows: list[dict[str, Any]] = []

    for mode in modes:
        for strength in strengths:
            run_name = f"mode_{mode}_strength_{strength:.2f}".replace(".", "p")
            run_out = output_dir / run_name
            cfg = replace(
                base_cfg,
                paths=replace(base_cfg.paths, output_root=str(run_out)),
                fairness=replace(base_cfg.fairness, debias_mode=mode, debias_strength=float(strength)),
            )
            _prime_cached_features(base_cfg, cfg)
            cfg_path = write_experiment_config(cfg, run_out / "config.yaml")
            logger.info("Running debias ablation: mode=%s strength=%.2f", mode, strength)
            report = run_full_study(str(cfg_path))
            perf = report.get("sae_concept_probe_debiased", {}).get("performance", {})
            fair = report.get("sae_concept_probe_debiased", {}).get("fairness", {})
            rows.append(
                {
                    "debias_mode": mode,
                    "debias_strength": float(strength),
                    "macro_auroc": perf.get("macro_auroc"),
                    "macro_accuracy": perf.get("macro_accuracy"),
                    "worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
                    "worst_group_macro_accuracy": _worst_group_value(fair.get("worst_group_macro_accuracy")),
                    "macro_auroc_gap": fair.get("macro_auroc_gap"),
                    "macro_accuracy_gap": fair.get("macro_accuracy_gap"),
                    "report_path": str((run_out / "study_metrics.json").resolve()),
                }
            )
    return rows


def _worst_group_value(entry: Any) -> float:
    if isinstance(entry, dict):
        value = entry.get("value")
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def _prime_cached_features(source_cfg: ExperimentConfig, target_cfg: ExperimentConfig) -> None:
    source_feature = source_cfg.feature_path
    source_manifest = source_cfg.manifest_path
    target_feature = target_cfg.feature_path
    target_manifest = target_cfg.manifest_path
    target_feature.parent.mkdir(parents=True, exist_ok=True)
    if source_feature.exists() and not target_feature.exists():
        shutil.copy2(source_feature, target_feature)
    if source_manifest.exists() and not target_manifest.exists():
        shutil.copy2(source_manifest, target_manifest)


def _build_paired_test_rows(
    prediction_bundle: dict[str, Any],
    threshold: float,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    labels = [str(v) for v in prediction_bundle["pathology_cols"].tolist()]
    paired = paired_bootstrap_method_tests(
        y_true=prediction_bundle["y_true"].astype(np.float32),
        age_groups=prediction_bundle["age_groups"].astype(str),
        label_names=labels,
        threshold=threshold,
        method_scores={
            "Baseline": prediction_bundle["baseline_scores"].astype(np.float32),
            "SAE Concepts": prediction_bundle["concept_scores"].astype(np.float32),
            "SAE Concepts (Debiased)": prediction_bundle["debiased_scores"].astype(np.float32),
        },
        method_pairs=[
            ("Baseline", "SAE Concepts"),
            ("Baseline", "SAE Concepts (Debiased)"),
            ("SAE Concepts", "SAE Concepts (Debiased)"),
        ],
        metrics=[
            "worst_group_macro_auroc",
            "macro_auroc_gap",
            "macro_auroc",
            "macro_accuracy",
            "macro_brier",
            "macro_ece",
        ],
        n_bootstrap=max(200, int(bootstrap_samples)),
        seed=13,
    )
    rows: list[dict[str, Any]] = []
    for row in paired:
        rows.append(
            {
                "metric": row.metric,
                "method_a": row.method_a,
                "method_b": row.method_b,
                "observed_delta": row.observed_delta,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "p_value": row.p_value,
            }
        )
    return rows
