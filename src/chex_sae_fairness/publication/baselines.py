from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import NMF, PCA
import torch
from torch import nn

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

    if "equalized_odds" in method_set:
        train_scores, test_scores = _fit_and_predict_with_train_scores(
            x_train=inputs.x_train,
            x_test=inputs.x_test,
            y_train=inputs.y_train,
            c_value=inputs.probe_c_value,
            max_iter=inputs.probe_max_iter,
        )
        threshold_map = _fit_equalized_odds_thresholds(
            y_true=inputs.y_train,
            y_score=train_scores,
            groups=inputs.age_groups_train.astype(str),
            global_threshold=inputs.threshold,
        )
        results["equalized_odds"] = _evaluate_group_threshold_method(
            inputs=inputs,
            y_score=test_scores,
            threshold_map=threshold_map,
            method_name="equalized_odds_postprocessing",
        )

    if "supervised_bottleneck" in method_set:
        scores = _fit_supervised_bottleneck_predict(
            x_train=inputs.x_train,
            x_test=inputs.x_test,
            y_train=inputs.y_train,
            latent_dim=inputs.latent_dim,
            seed=13,
        )
        results["supervised_bottleneck"] = _evaluate_scores(inputs, scores)

    if "adversarial_debiasing" in method_set:
        scores = _fit_adversarial_debias_predict(
            x_train=inputs.x_train,
            x_test=inputs.x_test,
            y_train=inputs.y_train,
            age_groups_train=inputs.age_groups_train.astype(str),
            latent_dim=inputs.latent_dim,
            seed=13,
        )
        results["adversarial_debiasing"] = _evaluate_scores(inputs, scores)

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
    method_name: str = "group_threshold_postprocessing",
) -> dict[str, object]:
    y_pred = _apply_group_thresholds(
        y_score=y_score,
        groups=inputs.age_groups_test.astype(str),
        threshold_map=threshold_map,
        global_threshold=inputs.threshold,
    )
    perf, fair = _evaluate_with_custom_predictions(
        y_true=inputs.y_test,
        y_score=y_score,
        y_pred=y_pred,
        groups=inputs.age_groups_test.astype(str),
        label_names=inputs.pathology_cols,
        global_threshold=inputs.threshold,
    )
    fair["method"] = method_name
    fair["threshold_map"] = {k: float(v) for k, v in threshold_map.items()}
    return {"performance": perf, "fairness": fair}


def _fit_supervised_bottleneck_predict(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    latent_dim: int,
    seed: int,
    epochs: int = 25,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = ((x_train - x_mean) / x_std).astype(np.float32)
    x_test_n = ((x_test - x_mean) / x_std).astype(np.float32)

    model = _BottleneckProbe(
        input_dim=x_train.shape[1],
        latent_dim=min(latent_dim, x_train.shape[1]),
        n_labels=y_train.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    x_tensor = torch.tensor(x_train_n, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(x_test_n, dtype=torch.float32, device=device))
        score = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return score


def _fit_adversarial_debias_predict(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    age_groups_train: np.ndarray,
    latent_dim: int,
    seed: int,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    adv_weight: float = 0.3,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = ((x_train - x_mean) / x_std).astype(np.float32)
    x_test_n = ((x_test - x_mean) / x_std).astype(np.float32)

    unique_groups = sorted(np.unique(age_groups_train).tolist())
    group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
    g_train = np.array([group_to_idx[g] for g in age_groups_train.tolist()], dtype=np.int64)

    model = _AdversarialBottleneck(
        input_dim=x_train.shape[1],
        latent_dim=min(latent_dim, x_train.shape[1]),
        n_labels=y_train.shape[1],
        n_groups=len(unique_groups),
    ).to(device)
    optimizer_enc = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.label_head.parameters()),
        lr=learning_rate,
        weight_decay=1e-6,
    )
    optimizer_adv = torch.optim.AdamW(model.group_head.parameters(), lr=learning_rate, weight_decay=1e-6)
    label_criterion = nn.BCEWithLogitsLoss()
    group_criterion = nn.CrossEntropyLoss()

    x_tensor = torch.tensor(x_train_n, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
    g_tensor = torch.tensor(g_train, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, g_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb, gb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            gb = gb.to(device)

            # Step 1: update adversary to predict age groups from frozen encoder features.
            with torch.no_grad():
                latent_detached = model.encode(xb)
            group_logits = model.group_head(latent_detached)
            loss_group = group_criterion(group_logits, gb)
            optimizer_adv.zero_grad(set_to_none=True)
            loss_group.backward()
            optimizer_adv.step()

            # Step 2: update encoder+label head to predict labels while confusing adversary.
            latent = model.encode(xb)
            label_logits = model.label_head(latent)
            group_logits_adv = model.group_head(latent)
            loss_label = label_criterion(label_logits, yb)
            loss_confuse = -adv_weight * group_criterion(group_logits_adv, gb)
            loss_total = loss_label + loss_confuse
            optimizer_enc.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer_enc.step()

    model.eval()
    with torch.no_grad():
        latent = model.encode(torch.tensor(x_test_n, dtype=torch.float32, device=device))
        logits = model.label_head(latent)
        score = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return score


def _fit_equalized_odds_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    global_threshold: float,
) -> dict[str, float]:
    # Approximate EO post-processing with per-group scalar threshold selected to
    # match global macro TPR/FPR on train data.
    y_pred_global = (y_score >= global_threshold).astype(int)
    target_tpr, target_fpr = _macro_tpr_fpr(y_true, y_pred_global)

    grid = np.linspace(0.1, 0.9, 17)
    threshold_map: dict[str, float] = {}
    for group in sorted(np.unique(groups).tolist()):
        mask = groups == group
        if int(mask.sum()) == 0:
            continue
        best_threshold = float(global_threshold)
        best_score = float("inf")
        for threshold in grid:
            y_pred = (y_score[mask] >= threshold).astype(int)
            tpr, fpr = _macro_tpr_fpr(y_true[mask], y_pred)
            objective = abs(tpr - target_tpr) + abs(fpr - target_fpr)
            if objective < best_score:
                best_score = objective
                best_threshold = float(threshold)
        threshold_map[str(group)] = best_threshold
    return threshold_map


def _apply_group_thresholds(
    y_score: np.ndarray,
    groups: np.ndarray,
    threshold_map: dict[str, float],
    global_threshold: float,
) -> np.ndarray:
    y_pred = np.zeros_like(y_score, dtype=np.int32)
    for idx, group in enumerate(groups):
        threshold = float(threshold_map.get(str(group), global_threshold))
        y_pred[idx] = (y_score[idx] >= threshold).astype(np.int32)
    return y_pred


def _evaluate_with_custom_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    label_names: list[str],
    global_threshold: float,
) -> tuple[dict[str, object], dict[str, object]]:
    perf = evaluate_multilabel_predictions(
        y_true=y_true,
        y_score=y_score,
        label_names=label_names,
        threshold=global_threshold,
    )
    perf["macro_accuracy"] = float(np.mean(np.mean(y_pred == y_true.astype(int), axis=0)))
    perf["micro_accuracy"] = float(np.mean(y_pred == y_true.astype(int)))
    perf["label_accuracy"] = {
        label: float(np.mean(y_pred[:, idx] == y_true[:, idx].astype(int)))
        for idx, label in enumerate(label_names)
    }

    groups_unique = sorted(np.unique(groups).tolist())
    group_metrics: dict[str, dict[str, object]] = {}
    group_auroc_values: list[float] = []
    group_acc_values: list[float] = []
    group_tpr_values: list[float] = []
    group_fpr_values: list[float] = []

    for group in groups_unique:
        mask = groups == group
        if int(mask.sum()) == 0:
            continue
        group_perf = evaluate_multilabel_predictions(
            y_true=y_true[mask],
            y_score=y_score[mask],
            label_names=label_names,
            threshold=global_threshold,
        )
        macro_acc = float(np.mean(np.mean(y_pred[mask] == y_true[mask].astype(int), axis=0)))
        group_perf["macro_accuracy"] = macro_acc
        group_perf["micro_accuracy"] = float(np.mean(y_pred[mask] == y_true[mask].astype(int)))
        group_perf["label_accuracy"] = {
            label: float(np.mean(y_pred[mask][:, idx] == y_true[mask][:, idx].astype(int)))
            for idx, label in enumerate(label_names)
        }
        tpr, fpr = _macro_tpr_fpr(y_true[mask], y_pred[mask])
        group_metrics[str(group)] = {
            **group_perf,
            "macro_tpr": tpr,
            "macro_fpr": fpr,
            "n": int(mask.sum()),
        }
        if np.isfinite(group_perf["macro_auroc"]):
            group_auroc_values.append(float(group_perf["macro_auroc"]))
        if np.isfinite(macro_acc):
            group_acc_values.append(macro_acc)
        if np.isfinite(tpr):
            group_tpr_values.append(tpr)
        if np.isfinite(fpr):
            group_fpr_values.append(fpr)

    worst_group_auroc = _min_group_metric(group_metrics, "macro_auroc")
    worst_group_acc = _min_group_metric(group_metrics, "macro_accuracy")
    fairness = {
        "groups": group_metrics,
        "overall": perf,
        "macro_auroc_gap": (
            float(max(group_auroc_values) - min(group_auroc_values))
            if group_auroc_values
            else float("nan")
        ),
        "macro_accuracy_gap": (
            float(max(group_acc_values) - min(group_acc_values))
            if group_acc_values
            else float("nan")
        ),
        "worst_group_macro_auroc": worst_group_auroc,
        "worst_group_macro_accuracy": worst_group_acc,
        "equalized_odds_tpr_gap": (
            float(max(group_tpr_values) - min(group_tpr_values))
            if group_tpr_values
            else float("nan")
        ),
        "equalized_odds_fpr_gap": (
            float(max(group_fpr_values) - min(group_fpr_values))
            if group_fpr_values
            else float("nan")
        ),
        "bootstrap_macro_auroc_gap": {},
    }
    return perf, fairness


def _macro_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tpr_values: list[float] = []
    fpr_values: list[float] = []
    for idx in range(y_true.shape[1]):
        yt = y_true[:, idx].astype(int)
        yp = y_pred[:, idx].astype(int)
        tp = float(np.sum((yt == 1) & (yp == 1)))
        fn = float(np.sum((yt == 1) & (yp == 0)))
        fp = float(np.sum((yt == 0) & (yp == 1)))
        tn = float(np.sum((yt == 0) & (yp == 0)))
        if tp + fn > 0:
            tpr_values.append(tp / (tp + fn))
        if fp + tn > 0:
            fpr_values.append(fp / (fp + tn))
    macro_tpr = float(np.mean(tpr_values)) if tpr_values else float("nan")
    macro_fpr = float(np.mean(fpr_values)) if fpr_values else float("nan")
    return macro_tpr, macro_fpr


def _min_group_metric(group_metrics: dict[str, dict[str, object]], metric: str) -> dict[str, object] | None:
    candidates: list[tuple[str, float]] = []
    for group, values in group_metrics.items():
        candidate = values.get(metric)
        if isinstance(candidate, (int, float, np.integer, np.floating)) and np.isfinite(float(candidate)):
            candidates.append((group, float(candidate)))
    if not candidates:
        return None
    group, value = min(candidates, key=lambda item: item[1])
    return {"group": str(group), "value": float(value)}


class _BottleneckProbe(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_labels: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, n_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.activation(self.encoder(x))
        logits = self.classifier(latent)
        return logits, latent


class _AdversarialBottleneck(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_labels: int, n_groups: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.activation = nn.ReLU()
        self.label_head = nn.Linear(latent_dim, n_labels)
        self.group_head = nn.Linear(latent_dim, n_groups)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.encoder(x))
