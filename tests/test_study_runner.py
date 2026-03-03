from pathlib import Path

from chex_sae_fairness.study_runner import create_timestamped_run_dir


def test_create_timestamped_run_dir_avoids_overwrite(tmp_path: Path) -> None:
    first = create_timestamped_run_dir(tmp_path, run_name="fixed")
    second = create_timestamped_run_dir(tmp_path, run_name="fixed")

    assert first != second
    assert first.name == "fixed"
    assert second.name.startswith("fixed_")
    assert first.exists()
    assert second.exists()
