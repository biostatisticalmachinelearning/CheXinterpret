from pathlib import Path

from chex_sae_fairness.data.chexpert_plus import _derive_split_from_path, _resolve_png_image_path


def test_resolve_png_path_when_train_under_root(tmp_path: Path) -> None:
    image_path = tmp_path / "train" / "patient00001" / "study1" / "view1_frontal.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"x")

    resolved = _resolve_png_image_path("train/patient00001/study1/view1_frontal.jpg", tmp_path)
    assert resolved == str(image_path.resolve())


def test_resolve_png_path_when_train_under_png_dir(tmp_path: Path) -> None:
    image_path = tmp_path / "PNG" / "train" / "patient00001" / "study1" / "view1_frontal.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"x")

    resolved = _resolve_png_image_path("train/patient00001/study1/view1_frontal.jpg", tmp_path)
    assert resolved == str(image_path.resolve())


def test_derive_split_from_prefixed_path() -> None:
    split = _derive_split_from_path("CheXpert-v1.0-small/train/patient00003/study1/view1_frontal.jpg")
    assert split == "train"
