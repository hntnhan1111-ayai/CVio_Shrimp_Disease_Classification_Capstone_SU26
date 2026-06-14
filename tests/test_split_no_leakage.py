import csv
from pathlib import Path

import pytest


def test_seed42_manifest_has_no_filename_leakage():
    candidates = [
        Path("artifacts/manifests/split_manifest_seed42.csv"),
        Path("artifacts/manifests/split_manifest_seed42_generated.csv"),
    ]
    manifest = next((path for path in candidates if path.is_file()), None)
    if manifest is None:
        pytest.skip("No seed-42 split manifest is present")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_split = {
        split: {row["rel_path"] for row in rows if row["split"] == split}
        for split in ("train", "val", "test")
    }
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
