import json
import zipfile
from pathlib import Path

from chex_sae_fairness.data.chexpert_plus import _read_label_table


def test_read_label_table_from_zip_archive(tmp_path: Path) -> None:
    rows = [
        {
            "path_to_image": "train/patient00001/study1/view1_frontal.jpg",
            "No Finding": 1,
            "Cardiomegaly": 0,
        }
    ]

    zip_path = tmp_path / "chexbert_labels.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("chexbert_labels/findings_fixed.json", "\n".join(json.dumps(r) for r in rows))

    labels = _read_label_table(zip_path)

    assert len(labels) == 1
    assert labels.iloc[0]["path_to_image"] == rows[0]["path_to_image"]
    assert int(labels.iloc[0]["No Finding"]) == 1
