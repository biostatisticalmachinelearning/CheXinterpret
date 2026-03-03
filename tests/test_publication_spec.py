from pathlib import Path

from chex_sae_fairness.publication.spec import PublicationSpec, write_publication_spec_template


def test_publication_spec_defaults_load_without_file() -> None:
    spec = PublicationSpec.from_yaml(None)
    assert len(spec.supplement.seeds) >= 1
    assert "train_and_test" in spec.core.debias_ablation_modes


def test_write_publication_spec_template(tmp_path: Path) -> None:
    out = write_publication_spec_template(tmp_path / "paper.yaml")
    assert out.exists()
    parsed = PublicationSpec.from_yaml(out)
    assert parsed.core.sweep_config_path.endswith("sae_sweep.yaml")
