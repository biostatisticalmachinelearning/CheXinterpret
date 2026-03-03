from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.data.splits import build_split_masks
from chex_sae_fairness.evaluation.disentanglement import summarize_latent_correlations
from chex_sae_fairness.models.chexagent_features import load_feature_bundle
from chex_sae_fairness.models.sae import SparseAutoencoder
from chex_sae_fairness.pipeline import run_full_study
from chex_sae_fairness.publication.baselines import BaselineSuiteInputs, run_baseline_suite
from chex_sae_fairness.publication.common import (
    create_timestamped_pipeline_dir,
    load_prediction_bundle,
    write_experiment_config,
)
from chex_sae_fairness.publication.figures import generate_supplement_figures
from chex_sae_fairness.publication.spec import SupplementSpec
from chex_sae_fairness.publication.statistics import evaluate_prediction_bundle
from chex_sae_fairness.publication.tables import (
    build_supplement_table_ablations,
    build_supplement_table_seed_stability,
    write_table,
)
from chex_sae_fairness.study_runner import run_comprehensive_study
from chex_sae_fairness.training.train_sae import encode_features
from chex_sae_fairness.utils.io import read_json, write_json

logger = logging.getLogger(__name__)


def run_supplement_publication_pipeline(
    config_path: str,
    spec: SupplementSpec,
) -> dict[str, Any]:
    base_cfg = ExperimentConfig.from_yaml(config_path)
    run_dir = create_timestamped_pipeline_dir(
        base_output_root=base_cfg.output_root,
        pipeline_name="supplement",
        run_name=spec.run_name,
    )
    logger.info("Supplement publication run dir: %s", run_dir)

    anchor = run_comprehensive_study(
        config_path=config_path,
        sweep_config_path=None,
        run_name=run_dir.name,
        run_root=run_dir / "anchor",
        force_recompute_features=spec.force_recompute_features,
    )

    best_cfg_path = Path(str(anchor["run_root"])) / "configs" / "best_sae_config.yaml"
    best_cfg = ExperimentConfig.from_yaml(best_cfg_path)
    best_report_path = Path(str(anchor["best_study_report"]))
    best_report = read_json(best_report_path)
    prediction_bundle = load_prediction_bundle(best_report_path.parent / "study_predictions.npz")

    seed_stability = _run_seed_stability(best_cfg, spec, run_dir / "seed_stability")
    uncertain_policy = _run_uncertain_policy_ablation(best_cfg, spec, run_dir / "uncertain_policy")
    debias_ablation = _run_debias_ablation(best_cfg, spec, run_dir / "debias_ablation")
    age_bin_sensitivity = _run_age_bin_sensitivity(best_cfg, spec, run_dir / "age_bins")
    threshold_sensitivity = _run_threshold_sensitivity(best_cfg, spec, run_dir / "threshold_sensitivity")

    baseline_comparison = _run_baseline_comparison(best_cfg, spec, run_dir / "baselines")
    missingness_sensitivity = _run_missingness_sensitivity(
        best_report=best_report,
        prediction_bundle=prediction_bundle,
        missing_fractions=spec.missing_metadata_fractions,
        output_dir=run_dir / "missingness_sensitivity",
    )
    permutation_control = _run_permutation_controls(best_cfg, spec, run_dir / "permutation_control")

    external_validation = _run_external_validation(spec.external_config_paths, run_dir / "external_validation")
    human_eval = _run_human_eval_summary(spec.human_eval_csv)

    figures = generate_supplement_figures(
        output_dir=run_dir / "figures",
        seed_stability=seed_stability,
        uncertain_policy=uncertain_policy,
        debias_ablation=debias_ablation,
        age_bin_sensitivity=age_bin_sensitivity,
        baseline_comparison=baseline_comparison,
        threshold_sensitivity=threshold_sensitivity,
        missingness_sensitivity=missingness_sensitivity,
        permutation_control=permutation_control,
    )

    table_paths: dict[str, dict[str, str]] = {}
    table_paths["stable_seeds"] = write_table(
        build_supplement_table_seed_stability(seed_stability.to_dict(orient="records")),
        run_dir / "tables" / "stable_seeds",
    )
    table_paths["uncertain_policy"] = write_table(
        build_supplement_table_ablations(uncertain_policy.to_dict(orient="records"), ["uncertain_policy", "method"]),
        run_dir / "tables" / "uncertain_policy",
    )
    table_paths["debias_ablation"] = write_table(
        build_supplement_table_ablations(debias_ablation.to_dict(orient="records"), ["debias_mode", "debias_strength"]),
        run_dir / "tables" / "debias_ablation",
    )
    table_paths["age_bin_sensitivity"] = write_table(
        build_supplement_table_ablations(age_bin_sensitivity.to_dict(orient="records"), ["age_bins", "method"]),
        run_dir / "tables" / "age_bin_sensitivity",
    )
    table_paths["baseline_comparison"] = write_table(
        build_supplement_table_ablations(baseline_comparison.to_dict(orient="records"), ["method"]),
        run_dir / "tables" / "baseline_comparison",
    )
    table_paths["threshold_sensitivity"] = write_table(
        build_supplement_table_ablations(threshold_sensitivity.to_dict(orient="records"), ["threshold", "method"]),
        run_dir / "tables" / "threshold_sensitivity",
    )
    table_paths["missingness_sensitivity"] = write_table(
        build_supplement_table_ablations(missingness_sensitivity.to_dict(orient="records"), ["missing_fraction", "method"]),
        run_dir / "tables" / "missingness_sensitivity",
    )
    if not permutation_control.empty:
        table_paths["permutation_control"] = write_table(
            permutation_control,
            run_dir / "tables" / "permutation_control",
        )
    if not external_validation.empty:
        table_paths["external_validation"] = write_table(
            external_validation,
            run_dir / "tables" / "external_validation",
        )
    if not human_eval.empty:
        table_paths["human_eval"] = write_table(
            human_eval,
            run_dir / "tables" / "human_eval",
        )

    artifact = {
        "run_dir": str(run_dir.resolve()),
        "anchor": anchor,
        "tables": table_paths,
        "figures": [str(path.resolve()) for path in figures],
        "external_validation_enabled": bool(spec.external_config_paths),
        "human_eval_enabled": spec.human_eval_csv is not None,
    }
    write_json(artifact, run_dir / "supplement_pipeline_summary.json")
    return artifact


def _run_seed_stability(cfg: ExperimentConfig, spec: SupplementSpec, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in spec.seeds:
        run_out = output_dir / f"seed_{seed}"
        run_cfg = replace(cfg, seed=int(seed), paths=replace(cfg.paths, output_root=str(run_out)))
        _prime_cached_features(cfg, run_cfg)
        run_cfg_path = write_experiment_config(run_cfg, run_out / "config.yaml")
        report = run_full_study(str(run_cfg_path))
        for method_key, method_label in [
            ("baseline_feature_probe", "Baseline"),
            ("sae_concept_probe", "SAE"),
            ("sae_concept_probe_debiased", "SAE Debiased"),
        ]:
            perf = report.get(method_key, {}).get("performance", {})
            fair = report.get(method_key, {}).get("fairness", {})
            rows.append(
                {
                    "seed": int(seed),
                    "method": method_label,
                    "macro_auroc": perf.get("macro_auroc"),
                    "macro_accuracy": perf.get("macro_accuracy"),
                    "macro_auroc_gap": fair.get("macro_auroc_gap"),
                    "worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
                }
            )
    return pd.DataFrame(rows)


def _run_uncertain_policy_ablation(cfg: ExperimentConfig, spec: SupplementSpec, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in spec.uncertain_policies:
        run_out = output_dir / f"policy_{policy}"
        run_cfg = replace(
            cfg,
            data=replace(cfg.data, uncertain_label_policy=str(policy)),
            paths=replace(cfg.paths, output_root=str(run_out)),
        )
        run_cfg_path = write_experiment_config(run_cfg, run_out / "config.yaml")
        report = run_full_study(str(run_cfg_path))
        for method_key, method_label in [
            ("baseline_feature_probe", "Baseline"),
            ("sae_concept_probe", "SAE"),
            ("sae_concept_probe_debiased", "SAE Debiased"),
        ]:
            perf = report.get(method_key, {}).get("performance", {})
            fair = report.get(method_key, {}).get("fairness", {})
            rows.append(
                {
                    "uncertain_policy": str(policy),
                    "method": method_label,
                    "macro_auroc": perf.get("macro_auroc"),
                    "macro_accuracy": perf.get("macro_accuracy"),
                    "macro_auroc_gap": fair.get("macro_auroc_gap"),
                }
            )
    return pd.DataFrame(rows)


def _run_debias_ablation(cfg: ExperimentConfig, spec: SupplementSpec, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in spec.debias_modes:
        for strength in spec.debias_strengths:
            run_out = output_dir / f"mode_{mode}_strength_{strength:.2f}".replace(".", "p")
            run_cfg = replace(
                cfg,
                fairness=replace(cfg.fairness, debias_mode=str(mode), debias_strength=float(strength)),
                paths=replace(cfg.paths, output_root=str(run_out)),
            )
            _prime_cached_features(cfg, run_cfg)
            run_cfg_path = write_experiment_config(run_cfg, run_out / "config.yaml")
            report = run_full_study(str(run_cfg_path))
            perf = report.get("sae_concept_probe_debiased", {}).get("performance", {})
            fair = report.get("sae_concept_probe_debiased", {}).get("fairness", {})
            rows.append(
                {
                    "debias_mode": str(mode),
                    "debias_strength": float(strength),
                    "macro_auroc": perf.get("macro_auroc"),
                    "macro_accuracy": perf.get("macro_accuracy"),
                    "macro_auroc_gap": fair.get("macro_auroc_gap"),
                    "macro_accuracy_gap": fair.get("macro_accuracy_gap"),
                    "worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
                }
            )
    return pd.DataFrame(rows)


def _run_age_bin_sensitivity(cfg: ExperimentConfig, spec: SupplementSpec, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for age_bins in spec.age_bin_sets:
        if len(age_bins) < 2:
            continue
        token = "-".join(str(v) for v in age_bins)
        run_out = output_dir / f"bins_{token}"
        run_cfg = replace(
            cfg,
            data=replace(cfg.data, age_bins=list(age_bins)),
            paths=replace(cfg.paths, output_root=str(run_out)),
        )
        run_cfg_path = write_experiment_config(run_cfg, run_out / "config.yaml")
        report = run_full_study(str(run_cfg_path))
        for method_key, method_label in [
            ("baseline_feature_probe", "Baseline"),
            ("sae_concept_probe_debiased", "SAE Debiased"),
        ]:
            perf = report.get(method_key, {}).get("performance", {})
            fair = report.get(method_key, {}).get("fairness", {})
            rows.append(
                {
                    "age_bins": token,
                    "method": method_label,
                    "macro_auroc": perf.get("macro_auroc"),
                    "macro_auroc_gap": fair.get("macro_auroc_gap"),
                }
            )
    return pd.DataFrame(rows)


def _run_threshold_sensitivity(cfg: ExperimentConfig, spec: SupplementSpec, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in spec.fairness_thresholds:
        run_out = output_dir / f"threshold_{threshold:.2f}".replace(".", "p")
        run_cfg = replace(
            cfg,
            fairness=replace(cfg.fairness, threshold=float(threshold)),
            paths=replace(cfg.paths, output_root=str(run_out)),
        )
        _prime_cached_features(cfg, run_cfg)
        run_cfg_path = write_experiment_config(run_cfg, run_out / "config.yaml")
        report = run_full_study(str(run_cfg_path))
        for method_key, method_label in [
            ("baseline_feature_probe", "Baseline"),
            ("sae_concept_probe_debiased", "SAE Debiased"),
        ]:
            fair = report.get(method_key, {}).get("fairness", {})
            rows.append(
                {
                    "threshold": float(threshold),
                    "method": method_label,
                    "macro_auroc_gap": fair.get("macro_auroc_gap"),
                    "macro_accuracy_gap": fair.get("macro_accuracy_gap"),
                }
            )
    return pd.DataFrame(rows)


def _run_baseline_comparison(cfg: ExperimentConfig, spec: SupplementSpec, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_feature_bundle(str(cfg.feature_path))
    x = bundle["features"].astype(np.float32)
    y = bundle["y_pathology"].astype(np.float32)
    splits = bundle["split"].astype(str)
    split_masks = build_split_masks(
        splits=splits,
        valid_name=cfg.data.validation_split_name,
        test_name=cfg.data.test_split_name,
        context="feature bundle",
    )

    path_cols = [str(c) for c in bundle["pathology_cols"].tolist()]
    inputs = BaselineSuiteInputs(
        x_train=x[split_masks.train],
        x_test=x[split_masks.test],
        y_train=y[split_masks.train],
        y_test=y[split_masks.test],
        age_groups_train=bundle["age_group"][split_masks.train].astype(str),
        age_groups_test=bundle["age_group"][split_masks.test].astype(str),
        pathology_cols=path_cols,
        threshold=cfg.fairness.threshold,
        bootstrap_samples=cfg.fairness.bootstrap_samples,
        probe_c_value=cfg.probes.c_value,
        probe_max_iter=cfg.probes.max_iter,
        latent_dim=cfg.sae.latent_dim,
    )
    suite = run_baseline_suite(inputs, methods=spec.baseline_methods)

    rows: list[dict[str, Any]] = []
    for method, result in suite.items():
        perf = result.get("performance", {})
        fair = result.get("fairness", {})
        rows.append(
            {
                "method": method,
                "macro_auroc": perf.get("macro_auroc"),
                "macro_accuracy": perf.get("macro_accuracy"),
                "macro_auroc_gap": fair.get("macro_auroc_gap"),
                "worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
                "macro_accuracy_gap": fair.get("macro_accuracy_gap"),
            }
        )

    write_json(suite, output_dir / "baseline_suite.json")
    return pd.DataFrame(rows)


def _run_missingness_sensitivity(
    best_report: dict[str, Any],
    prediction_bundle: dict[str, np.ndarray],
    missing_fractions: list[float],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = prediction_bundle["y_true"].astype(np.float32)
    age_groups = prediction_bundle["age_groups"].astype(str)
    labels = [str(v) for v in prediction_bundle["pathology_cols"].tolist()]
    threshold = float(best_report.get("debiasing", {}).get("fairness_threshold", 0.5))
    rng = np.random.default_rng(13)

    method_scores = {
        "Baseline": prediction_bundle["baseline_scores"].astype(np.float32),
        "SAE": prediction_bundle["concept_scores"].astype(np.float32),
        "SAE Debiased": prediction_bundle["debiased_scores"].astype(np.float32),
    }

    rows: list[dict[str, Any]] = []
    n = len(y_true)
    all_indices = np.arange(n)
    for frac in missing_fractions:
        frac = float(frac)
        keep = max(1, int(round((1.0 - frac) * n)))
        sampled = np.sort(rng.choice(all_indices, size=keep, replace=False))
        for method, scores in method_scores.items():
            out = evaluate_prediction_bundle(
                y_true=y_true[sampled],
                y_score=scores[sampled],
                age_groups=age_groups[sampled],
                label_names=labels,
                threshold=threshold,
                bootstrap_samples=0,
            )
            rows.append(
                {
                    "missing_fraction": frac,
                    "method": method,
                    "macro_auroc": out["performance"]["macro_auroc"],
                    "macro_accuracy": out["performance"]["macro_accuracy"],
                    "macro_auroc_gap": out["fairness"]["macro_auroc_gap"],
                }
            )
    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "missingness_sensitivity")
    return frame


def _run_permutation_controls(cfg: ExperimentConfig, spec: SupplementSpec, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    z_test, y_test, metadata_test, path_cols, meta_cols = _load_test_latents_and_targets(cfg)
    observed = summarize_latent_correlations(
        z=z_test,
        y_pathology=y_test,
        pathology_cols=path_cols,
        metadata=metadata_test,
        metadata_cols=meta_cols,
    )
    observed_corr = float(observed.get("mean_pathology_max_abs_corr", float("nan")))
    rng = np.random.default_rng(13)
    null_scores: list[float] = []
    for _ in range(max(1, int(spec.permutation_repeats))):
        perm = rng.permutation(len(y_test))
        perm_corr = summarize_latent_correlations(
            z=z_test,
            y_pathology=y_test[perm],
            pathology_cols=path_cols,
            metadata=metadata_test.iloc[perm].reset_index(drop=True),
            metadata_cols=meta_cols,
        )
        null_scores.append(float(perm_corr.get("mean_pathology_max_abs_corr", float("nan"))))

    null_arr = np.array([v for v in null_scores if np.isfinite(v)], dtype=float)
    if len(null_arr) > 0 and np.isfinite(observed_corr):
        p_value = float((1.0 + np.sum(null_arr >= observed_corr)) / (len(null_arr) + 1.0))
    else:
        p_value = float("nan")
    frame = pd.DataFrame(
        {
            "observed_mean_pathology_corr": [observed_corr] * len(null_scores),
            "null_mean_pathology_corr": null_scores,
            "permutation_p_value": [p_value] * len(null_scores),
        }
    )
    write_table(frame, output_dir / "permutation_control")
    return frame


def _run_external_validation(config_paths: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for config_path in config_paths:
        path = Path(config_path).expanduser()
        if not path.exists():
            logger.warning("Skipping missing external config: %s", path)
            continue
        cfg = ExperimentConfig.from_yaml(path)
        run_cfg = replace(cfg, paths=replace(cfg.paths, output_root=str(output_dir / path.stem)))
        cfg_path = write_experiment_config(run_cfg, output_dir / path.stem / "config.yaml")
        report = run_full_study(str(cfg_path))
        perf = report.get("sae_concept_probe_debiased", {}).get("performance", {})
        fair = report.get("sae_concept_probe_debiased", {}).get("fairness", {})
        rows.append(
            {
                "dataset": path.stem,
                "macro_auroc": perf.get("macro_auroc"),
                "macro_accuracy": perf.get("macro_accuracy"),
                "macro_auroc_gap": fair.get("macro_auroc_gap"),
                "worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
            }
        )
    return pd.DataFrame(rows)


def _run_human_eval_summary(human_eval_csv: str | None) -> pd.DataFrame:
    if not human_eval_csv:
        return pd.DataFrame()
    path = Path(human_eval_csv).expanduser()
    if not path.exists():
        logger.warning("Human evaluation CSV not found: %s", path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"latent_index", "human_label", "auto_label"}
    if not required.issubset(frame.columns):
        logger.warning("Human evaluation CSV missing required columns: %s", required - set(frame.columns))
        return pd.DataFrame()
    agreement = float(np.mean(frame["human_label"].astype(str) == frame["auto_label"].astype(str)))
    return pd.DataFrame(
        [
            {
                "n_features_reviewed": int(len(frame)),
                "human_auto_agreement": agreement,
            }
        ]
    )


def _load_test_latents_and_targets(
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str]]:
    bundle = load_feature_bundle(str(cfg.feature_path))
    x = bundle["features"].astype(np.float32)
    y = bundle["y_pathology"].astype(np.float32)
    splits = bundle["split"].astype(str)
    path_cols = [str(v) for v in bundle["pathology_cols"].tolist()]
    meta_cols = [str(v) for v in bundle["metadata_cols"].tolist()]
    metadata = pd.DataFrame(bundle["metadata"], columns=meta_cols)
    masks = build_split_masks(
        splits=splits,
        valid_name=cfg.data.validation_split_name,
        test_name=cfg.data.test_split_name,
        context="feature bundle",
    )

    ckpt = torch.load(cfg.sae_checkpoint_path, map_location="cpu")
    model = SparseAutoencoder(
        input_dim=int(ckpt["input_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        variant=str(ckpt["variant"]),
        topk_k=int(ckpt.get("topk_k", cfg.sae.topk_k)),
    )
    model.load_state_dict(ckpt["state_dict"])
    z_test = encode_features(model=model, features=x[masks.test], batch_size=512, device="cpu")
    return (
        z_test.astype(np.float32),
        y[masks.test].astype(np.float32),
        metadata.loc[masks.test].reset_index(drop=True),
        path_cols,
        meta_cols,
    )


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
