from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import NMF, PCA

from chex_sae_fairness.evaluation.fairness import evaluate_group_fairness, evaluate_multilabel_predictions
from chex_sae_fairness.training.train_probe import fit_multilabel_probe


@dataclass(slots=True)
class BaselineSuiteInputs:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    age_groups_train: np.ndarray
    age_groups_test: np.ndarray
    pathology_cols: list[str]
    threshold: float
    bootstrap_samples: int
    probe_c_value: float
    probe_max_iter: int
    latent_dim: int


def run_baseline_suite(inputs: BaselineSuiteInputs, methods: list[str]) -> dict[str, dict[str, object]]:
    method_set = {method.strip().lower() for method in methods}
    results: dict[str, dict[str, object]] = {}

    if "raw" in method_set:
        raw_scores = _fit_and_predict(
            x_train=inputs.x_train,
            x_test=inputs.x_test,
            y_train=inputs.y_train,
            c_value=inputs.probe_c_value,
            max_iter=inputs.probe_max_iter,
        )
        results["raw"] = _evaluate_scores(inputs, raw_scores)

    if "pca" in method_set:
        pca = PCA(n_components=min(inputs.latent_dim, inputs.x_train.shape[1]), random_state=13)
        x_train_pca = pca.fit_transform(inputs.x_train)
        x_test_pca = pca.transform(inputs.x_test)
        scores = _fit_and_predict(
            x_train=x_train_pca,
            x_test=x_test_pca,
            y_train=inputs.y_train,
            c_value=inputs.probe_c_value,
            max_iter=inputs.probe_max_iter,
        )
        results["pca"] = _evaluate_scores(inputs, scores)
        results["pca"]["explained_variance_ratio"] = float(np.sum(pca.explained_variance_ratio_))

    if "nmf" in method_set:
        shift = float(min(inputs.x_train.min(), inputs.x_test.min()))
        x_train_pos = inputs.x_train - shift + 1e-6
        x_test_pos = inputs.x_test - shift + 1e-6
        nmf = NMF(
            n_components=min(inputs.latent_dim, inputs.x_train.shape[1]),
            init="nndsvda",
            max_iter=400,
            random_state=13,
        )
        x_train_nmf = nmf.fit_transform(x_train_pos)
        x_test_nmf = nmf.transform(x_test_pos)
        scores = _fit_and_predict(
            x_train=x_train_nmf,
            x_test=x_test_nmf,
            y_train=inputs.y_train,
            c_value=inputs.probe_c_value,
            max_iter=inputs.probe_max_iter,
        )
        results["nmf"] = _evaluate_scores(inputs, scores)
        results["nmf"]["nmf_reconstruction_err"] = float(nmf.reconstruction_err_)

    if "group_reweighted" in method_set:
        sample_weight = _inverse_group_frequency_weights(inputs.age_groups_train)
        scores = _fit_and_predict(
            x_train=inputs.x_train,
            x_test=inputs.x_test,
            y_train=inputs.y_train,
            c_value=inputs.probe_c_value,
            max_iter=inputs.probe_max_iter,
            sample_weight=sample_weight,
        )
        results["group_reweighted"] = _evaluate_scores(inputs, scores)

    if "group_threshold" in method_set:
        # Train base scorer on raw features, then calibrate per-group thresholds.
        train_scores, test_scores = _fit_and_predict_with_train_scores(
            x_train=inputs.x_train,
            x_test=inputs.x_test,
            y_train=inputs.y_train,
            c_value=inputs.probe_c_value,
            max_iter=inputs.probe_max_iter,
        )
        threshold_map = _fit_group_thresholds(
            y_true=inputs.y_train,
            y_score=train_scores,
            groups=inputs.age_groups_train.astype(str),
            global_threshold=inputs.threshold,
        )
        results["group_threshold"] = _evaluate_group_threshold_method(
            inputs=inputs,
            y_score=test_scores,
            threshold_map=threshold_map,
        )

    return results


def _fit_and_predict(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    probe = fit_multilabel_probe(
        x_train=x_train,
        y_train=y_train,
        c_value=c_value,
        max_iter=max_iter,
        sample_weight=sample_weight,
    )
    return probe.predict_proba(x_test).astype(np.float32)


def _fit_and_predict_with_train_scores(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    probe = fit_multilabel_probe(
        x_train=x_train,
        y_train=y_train,
        c_value=c_value,
        max_iter=max_iter,
    )
    return (
        probe.predict_proba(x_train).astype(np.float32),
        probe.predict_proba(x_test).astype(np.float32),
    )


def _evaluate_scores(inputs: BaselineSuiteInputs, scores: np.ndarray) -> dict[str, object]:
    perf = evaluate_multilabel_predictions(
        y_true=inputs.y_test,
        y_score=scores,
        label_names=inputs.pathology_cols,
        threshold=inputs.threshold,
    )
    fairness = evaluate_group_fairness(
        y_true=inputs.y_test,
        y_score=scores,
        groups=inputs.age_groups_test,
        label_names=inputs.pathology_cols,
        threshold=inputs.threshold,
        bootstrap_samples=inputs.bootstrap_samples,
    )
    return {"performance": perf, "fairness": fairness}


def _inverse_group_frequency_weights(groups: np.ndarray) -> np.ndarray:
    values = groups.astype(str)
    unique, counts = np.unique(values, return_counts=True)
    freq = {group: count for group, count in zip(unique.tolist(), counts.tolist())}
    n_total = float(len(values))
    n_groups = float(len(unique))
    weights = np.array([n_total / (n_groups * float(freq[g])) for g in values], dtype=np.float32)
    return weights


def _fit_group_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    global_threshold: float,
) -> dict[str, float]:
    target_positive_rate = float(np.mean(y_true))
    grid = np.linspace(0.1, 0.9, 17)
    threshold_map: dict[str, float] = {}
    for group in sorted(np.unique(groups).tolist()):
        mask = groups == group
        if int(mask.sum()) == 0:
            continue
        group_scores = y_score[mask]
        best_threshold = float(global_threshold)
        best_gap = float("inf")
        for t in grid:
            pred = (group_scores >= t).astype(int)
            rate = float(np.mean(pred))
            gap = abs(rate - target_positive_rate)
            if gap < best_gap:
                best_gap = gap
                best_threshold = float(t)
        threshold_map[str(group)] = best_threshold
    return threshold_map


def _evaluate_group_threshold_method(
    inputs: BaselineSuiteInputs,
    y_score: np.ndarray,
    threshold_map: dict[str, float],
) -> dict[str, object]:
    y_pred = np.zeros_like(y_score, dtype=np.int32)
    for idx, group in enumerate(inputs.age_groups_test.astype(str)):
        threshold = threshold_map.get(group, inputs.threshold)
        y_pred[idx] = (y_score[idx] >= threshold).astype(np.int32)

    perf = evaluate_multilabel_predictions(
        y_true=inputs.y_test,
        y_score=y_score,
        label_names=inputs.pathology_cols,
        threshold=inputs.threshold,
    )
    group_metrics: dict[str, dict[str, object]] = {}
    for group in sorted(np.unique(inputs.age_groups_test.astype(str)).tolist()):
        mask = inputs.age_groups_test.astype(str) == group
        group_acc = float(np.mean(y_pred[mask] == inputs.y_test[mask])) if int(mask.sum()) else float("nan")
        group_metrics[group] = {
            "n": int(mask.sum()),
            "macro_accuracy": group_acc,
            "threshold": float(threshold_map.get(group, inputs.threshold)),
        }

    group_acc_values = [m["macro_accuracy"] for m in group_metrics.values() if np.isfinite(m["macro_accuracy"])]
    if group_acc_values:
        gap = float(max(group_acc_values) - min(group_acc_values))
        worst_group = min(group_metrics.items(), key=lambda kv: kv[1]["macro_accuracy"])[0]
    else:
        gap = float("nan")
        worst_group = None

    fairness = {
        "method": "group_threshold_postprocessing",
        "groups": group_metrics,
        "macro_accuracy_gap": gap,
        "worst_group_macro_accuracy": (
            {"group": str(worst_group), "value": float(group_metrics[str(worst_group)]["macro_accuracy"])}
            if worst_group is not None
            else None
        ),
        "note": "AUROC metrics are inherited from the underlying score model.",
    }
    return {"performance": perf, "fairness": fairness}
