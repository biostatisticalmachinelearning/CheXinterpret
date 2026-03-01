from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ConceptResidualizer:
    global_mean: np.ndarray
    group_means: dict[str, np.ndarray]
    strength: float

    def transform(self, z: np.ndarray, groups: np.ndarray) -> np.ndarray:
        centered = np.empty_like(z)
        for idx, g in enumerate(groups.astype(str)):
            group_mean = self.group_means.get(g, self.global_mean)
            shift = self.strength * (group_mean - self.global_mean)
            centered[idx] = z[idx] - shift
        return centered


def fit_concept_residualizer(
    z_train: np.ndarray,
    age_groups: np.ndarray,
    strength: float,
) -> ConceptResidualizer:
    global_mean = z_train.mean(axis=0)
    group_means: dict[str, np.ndarray] = {}

    for group in sorted(np.unique(age_groups).tolist()):
        mask = age_groups == group
        group_means[str(group)] = z_train[mask].mean(axis=0)

    return ConceptResidualizer(global_mean=global_mean, group_means=group_means, strength=strength)


def rank_age_associated_concepts(
    z: np.ndarray,
    groups: np.ndarray,
    top_k: int = 25,
) -> list[dict[str, float]]:
    group_ids = sorted(np.unique(groups).tolist())
    if len(group_ids) < 2:
        return []

    scores: list[tuple[int, float]] = []
    global_mean = z.mean(axis=0)

    for dim in range(z.shape[1]):
        # Between-group to total variance ratio; higher values indicate stronger group association.
        between = 0.0
        total = float(np.mean((z[:, dim] - global_mean[dim]) ** 2))

        for g in group_ids:
            mask = groups == g
            group_mean = float(np.mean(z[mask, dim]))
            between += float(mask.mean()) * (group_mean - global_mean[dim]) ** 2

        score = between / (total + 1e-8)
        scores.append((dim, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [{"latent_index": int(dim), "age_assoc_score": float(score)} for dim, score in scores[:top_k]]
