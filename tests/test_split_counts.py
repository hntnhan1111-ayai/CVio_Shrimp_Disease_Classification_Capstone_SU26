"""Validate the seed-42 split counts against the expected paper distribution."""

import csv
from pathlib import Path

import pytest

MANIFEST_CANDIDATES = [
    Path("artifacts/manifests/split_manifest_seed42.csv"),
    Path("artifacts/manifests/split_manifest_seed42_generated.csv"),
]

EXPECTED = {
    "train": {"Healthy": 282, "BG": 139, "WSSV": 229, "WSSV_BG": 154},
    "val": {"Healthy": 60, "BG": 30, "WSSV": 49, "WSSV_BG": 33},
    "test": {"Healthy": 61, "BG": 29, "WSSV": 50, "WSSV_BG": 33},
}
EXPECTED_TOTALS = {"train": 804, "val": 172, "test": 173}


def _load_manifest():
    for candidate in MANIFEST_CANDIDATES:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
    return None


def test_split_counts_match_expected():
    rows = _load_manifest()
    if rows is None:
        pytest.skip("No seed-42 split manifest present (run 01_prepare_dataset_and_split.py)")
    for split, per_class in EXPECTED.items():
        for class_name, count in per_class.items():
            actual = sum(
                1 for r in rows if r["split"] == split and r["class_name"] == class_name
            )
            assert actual == count, (
                f"{split}/{class_name}: expected {count}, got {actual}"
            )
        total = sum(1 for r in rows if r["split"] == split)
        assert total == EXPECTED_TOTALS[split], f"{split} total: expected {EXPECTED_TOTALS[split]}, got {total}"


def test_split_has_no_path_leakage():
    rows = _load_manifest()
    if rows is None:
        pytest.skip("No seed-42 split manifest present")
    by_split = {
        split: {r["rel_path"] for r in rows if r["split"] == split}
        for split in ("train", "val", "test")
    }
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
