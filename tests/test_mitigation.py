import numpy as np

from chex_sae_fairness.mitigation.concept_debias import (
    fit_concept_residualizer,
    rank_age_associated_concepts,
)


def test_concept_residualizer_reduces_group_shift() -> None:
    rng = np.random.default_rng(13)
    g0 = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.2, size=(100, 3))
    g1 = rng.normal(loc=[2.0, 0.0, 0.0], scale=0.2, size=(100, 3))
    z = np.vstack([g0, g1]).astype(np.float32)
    groups = np.array(["18-39"] * len(g0) + ["60-79"] * len(g1))

    residualizer = fit_concept_residualizer(z, groups, strength=1.0)
    z_debiased = residualizer.transform(z, groups)

    m0 = z_debiased[groups == "18-39"].mean(axis=0)
    m1 = z_debiased[groups == "60-79"].mean(axis=0)

    assert abs(float(m0[0] - m1[0])) < 1e-4


def test_rank_age_associated_concepts() -> None:
    rng = np.random.default_rng(7)
    z0 = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(50, 2))
    z1 = rng.normal(loc=[1.5, 0.0], scale=0.2, size=(50, 2))
    z = np.vstack([z0, z1]).astype(np.float32)
    groups = np.array(["young"] * 50 + ["old"] * 50)

    ranking = rank_age_associated_concepts(z, groups, top_k=2)

    assert len(ranking) == 2
    assert ranking[0]["latent_index"] == 0
