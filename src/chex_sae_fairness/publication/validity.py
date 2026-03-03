from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from chex_sae_fairness.evaluation.disentanglement import summarize_latent_correlations
from chex_sae_fairness.publication.statistics import (
    attach_multiple_testing_corrections,
    evaluate_prediction_bundle,
)


def build_concept_precision_recall_table(
    z_train: np.ndarray,
    z_test: np.ndarray,
    y_path_train: np.ndarray,
    y_path_test: np.ndarray,
    pathology_cols: list[str],
    metadata_train: pd.DataFrame,
    metadata_test: pd.DataFrame,
    metadata_cols: list[str],
    seed: int = 13,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scaler = StandardScaler()
    x_train = scaler.fit_transform(z_train)
    x_test = scaler.transform(z_test)

    for idx, concept in enumerate(pathology_cols):
        row = _binary_concept_metrics(
            x_train=x_train,
            x_test=x_test,
            y_train=y_path_train[:, idx].astype(int),
            y_test=y_path_test[:, idx].astype(int),
            concept=concept,
            concept_type="pathology",
            seed=seed,
        )
        rows.append(row)

    for concept in metadata_cols:
        if concept not in metadata_train.columns or concept not in metadata_test.columns:
            continue
        train_series = metadata_train[concept]
        test_series = metadata_test[concept]
        if _is_numeric(train_series):
            y_train, y_test, note = _binarize_numeric_target(train_series, test_series)
            row = _binary_concept_metrics(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                concept=concept,
                concept_type="metadata_numeric",
                seed=seed,
            )
            row["target_note"] = note
            rows.append(row)
            continue

        y_train = train_series.astype(str).to_numpy()
        y_test = test_series.astype(str).to_numpy()
        row = _categorical_concept_metrics(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            concept=concept,
            seed=seed,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_concept_permutation_table(
    z: np.ndarray,
    y_pathology: np.ndarray,
    pathology_cols: list[str],
    metadata: pd.DataFrame,
    metadata_cols: list[str],
    repeats: int = 20,
    seed: int = 13,
) -> pd.DataFrame:
    observed = summarize_latent_correlations(
        z=z,
        y_pathology=y_pathology,
        pathology_cols=pathology_cols,
        metadata=metadata,
        metadata_cols=metadata_cols,
    )
    observed_rows = _rows_by_concept(observed)
    if not observed_rows:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    null_by_concept: dict[str, list[float]] = {concept: [] for concept in observed_rows}
    n = len(y_pathology)
    for _ in range(max(1, int(repeats))):
        perm = rng.permutation(n)
        perm_summary = summarize_latent_correlations(
            z=z,
            y_pathology=y_pathology[perm],
            pathology_cols=pathology_cols,
            metadata=metadata.iloc[perm].reset_index(drop=True),
            metadata_cols=metadata_cols,
        )
        perm_rows = _rows_by_concept(perm_summary)
        for concept, payload in perm_rows.items():
            value = payload.get("max_abs_corr", float("nan"))
            if np.isfinite(value):
                null_by_concept.setdefault(concept, []).append(float(value))

    rows: list[dict[str, Any]] = []
    for concept, payload in observed_rows.items():
        observed_corr = float(payload.get("max_abs_corr", float("nan")))
        null_values = np.asarray(
            [v for v in null_by_concept.get(concept, []) if np.isfinite(v)],
            dtype=float,
        )
        if len(null_values) and np.isfinite(observed_corr):
            p_value = float((1.0 + np.sum(null_values >= observed_corr)) / (len(null_values) + 1.0))
            null_mean = float(np.mean(null_values))
            null_std = float(np.std(null_values))
        else:
            p_value = float("nan")
            null_mean = float("nan")
            null_std = float("nan")
        rows.append(
            {
                "concept": concept,
                "concept_type": payload.get("concept_type"),
                "observed_max_abs_corr": observed_corr,
                "observed_latent_index": payload.get("latent_index"),
                "null_mean_max_abs_corr": null_mean,
                "null_std_max_abs_corr": null_std,
                "permutation_repeats": int(repeats),
                "p_value": p_value,
            }
        )

    rows = attach_multiple_testing_corrections(rows, p_key="p_value")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(by=["concept_type", "concept"])


def build_patient_split_leakage_table(
    split: np.ndarray,
    patient_id: np.ndarray | None,
) -> pd.DataFrame:
    split_values = split.astype(str)
    rows: list[dict[str, Any]] = []
    for split_name in sorted(np.unique(split_values).tolist()):
        rows.append(
            {
                "check": "split_count",
                "split_a": split_name,
                "split_b": "",
                "n_overlap": int(np.sum(split_values == split_name)),
            }
        )

    if patient_id is None or len(patient_id) == 0:
        rows.append(
            {
                "check": "patient_id_presence",
                "split_a": "",
                "split_b": "",
                "n_overlap": -1,
                "note": "patient_id unavailable in feature bundle",
            }
        )
        return pd.DataFrame(rows)

    patient_values = patient_id.astype(str)
    unique_splits = sorted(np.unique(split_values).tolist())
    for idx, split_a in enumerate(unique_splits):
        patients_a = set(patient_values[split_values == split_a].tolist())
        for split_b in unique_splits[idx + 1:]:
            patients_b = set(patient_values[split_values == split_b].tolist())
            overlap = patients_a.intersection(patients_b)
            rows.append(
                {
                    "check": "patient_overlap",
                    "split_a": split_a,
                    "split_b": split_b,
                    "n_overlap": int(len(overlap)),
                    "has_overlap": bool(len(overlap) > 0),
                }
            )
    return pd.DataFrame(rows)


def build_view_sensitivity_table(
    prediction_bundle: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    if "test_view_type" not in prediction_bundle:
        return pd.DataFrame()
    views = prediction_bundle["test_view_type"].astype(str)
    if len(views) == 0:
        return pd.DataFrame()

    y_true = prediction_bundle["y_true"].astype(np.float32)
    age_groups = prediction_bundle["age_groups"].astype(str)
    labels = [str(v) for v in prediction_bundle["pathology_cols"].tolist()]
    method_scores = {
        "Baseline": prediction_bundle["baseline_scores"].astype(np.float32),
        "SAE": prediction_bundle["concept_scores"].astype(np.float32),
        "SAE Debiased": prediction_bundle["debiased_scores"].astype(np.float32),
    }

    rows: list[dict[str, Any]] = []
    for view in sorted(np.unique(views).tolist()):
        mask = views == view
        if int(mask.sum()) == 0:
            continue
        for method, score in method_scores.items():
            out = evaluate_prediction_bundle(
                y_true=y_true[mask],
                y_score=score[mask],
                age_groups=age_groups[mask],
                label_names=labels,
                threshold=threshold,
                bootstrap_samples=0,
            )
            perf = out["performance"]
            fair = out["fairness"]
            rows.append(
                {
                    "view_type": view,
                    "method": method,
                    "n_samples": int(mask.sum()),
                    "macro_auroc": perf.get("macro_auroc"),
                    "macro_accuracy": perf.get("macro_accuracy"),
                    "macro_auroc_gap": fair.get("macro_auroc_gap"),
                    "worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
                }
            )
    return pd.DataFrame(rows)


def _binary_concept_metrics(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    concept: str,
    concept_type: str,
    seed: int,
) -> dict[str, Any]:
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "concept": concept,
            "concept_type": concept_type,
            "task_type": "binary",
            "n_test": int(len(y_test)),
            "positive_rate": float(np.mean(y_test)) if len(y_test) else float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
        }

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.2,
        max_iter=2500,
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "concept": concept,
        "concept_type": concept_type,
        "task_type": "binary",
        "n_test": int(len(y_test)),
        "positive_rate": float(np.mean(y_test)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_test, prob)),
    }


def _categorical_concept_metrics(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    concept: str,
    seed: int,
) -> dict[str, Any]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "concept": concept,
            "concept_type": "metadata_categorical",
            "task_type": "categorical",
            "n_test": int(len(y_test)),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
        }

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.2,
        max_iter=3000,
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "concept": concept,
        "concept_type": "metadata_categorical",
        "task_type": "categorical",
        "n_test": int(len(y_test)),
        "precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "auroc": float("nan"),
    }


def _binarize_numeric_target(train: pd.Series, test: pd.Series) -> tuple[np.ndarray, np.ndarray, str]:
    tr = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float)
    te = pd.to_numeric(test, errors="coerce").to_numpy(dtype=float)
    threshold = float(np.nanmedian(tr)) if np.isfinite(np.nanmedian(tr)) else 0.0
    y_train = np.where(np.isfinite(tr), (tr >= threshold).astype(int), 0)
    y_test = np.where(np.isfinite(te), (te >= threshold).astype(int), 0)
    return y_train.astype(int), y_test.astype(int), f"binarized_at_train_median={threshold:.4f}"


def _rows_by_concept(summary: dict[str, object]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = summary.get("concepts", []) if isinstance(summary, dict) else []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        concept = row.get("concept")
        if concept is None:
            continue
        out[str(concept)] = row
    return out


def _is_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().mean() >= 0.95


def _worst_group_value(entry: Any) -> float:
    if isinstance(entry, dict):
        value = entry.get("value")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    return float("nan")
