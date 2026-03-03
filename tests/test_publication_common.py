from pathlib import Path

from chex_sae_fairness.publication.common import create_timestamped_pipeline_dir


def test_create_timestamped_pipeline_dir_deduplicates_names(tmp_path: Path) -> None:
    first = create_timestamped_pipeline_dir(tmp_path, pipeline_name="core", run_name="fixed")
    second = create_timestamped_pipeline_dir(tmp_path, pipeline_name="core", run_name="fixed")
    assert first != second
    assert first.exists()
    assert second.exists()
