#!/usr/bin/env python3
"""Assert split protocol integrity.

Validates partition-wise combination:
- Combined train == ShrimpDB train union ShrimpDiseaseDB train
- Combined val   == ShrimpDB val   union ShrimpDiseaseDB val
- Combined test  == ShrimpDB test  union ShrimpDiseaseDB test
- no SHA-256 crosses partitions
- counts match the audited dataset_summary.json
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> int:
    expected = {
        "shrimpdb3": {
            "train_class_counts": [49, 78, 94],
            "distribution": {
                "train": {"Healthy": 49, "BG": 78, "WSSV": 94},
                "val":   {"Healthy": 10, "BG": 17, "WSSV": 20},
                "test":  {"Healthy": 11, "BG": 16, "WSSV": 20},
            },
            "total": 315,
        },
        "combined4": {
            "train_class_counts": [331, 217, 323, 154],
            "distribution": {
                "train": {"Healthy": 331, "BG": 217, "WSSV": 323, "WSSV_BG": 154},
                "val":   {"Healthy": 70,  "BG": 47,  "WSSV": 69,  "WSSV_BG": 33},
                "test":  {"Healthy": 72,  "BG": 45,  "WSSV": 70,  "WSSV_BG": 33},
            },
            "total": 1464,
        },
    }

    for exp_name, exp in expected.items():
        summary_path = Path(f"artifacts/experiments/{exp_name}/dataset_summary.json")
        manifest_path = Path(f"artifacts/experiments/{exp_name}/manifest.csv")
        assert summary_path.is_file(), f"Missing summary: {summary_path}"
        assert manifest_path.is_file(), f"Missing manifest: {manifest_path}"

        summary = _read_json(summary_path)
        assert summary["class_names"] == ["Healthy", "BG", "WSSV"] if exp_name == "shrimpdb3" else ["Healthy", "BG", "WSSV", "WSSV_BG"]
        assert summary["train_class_counts"] == exp["train_class_counts"]
        assert summary["distribution"] == exp["distribution"]
        assert summary["total"] == exp["total"]

        rows = _read_csv(manifest_path)
        # partition-wise combination rule for combined4
        if exp_name == "combined4":
            sd_rows = [r for r in rows if r.get("source_dataset") == "ShrimpDB"]
            sdd_rows = [r for r in rows if r.get("source_dataset") == "ShrimpDiseaseDB"]
            assert sd_rows and sdd_rows, "Combined manifest must contain both sources"
            sd_split = defaultdict(set)
            sdd_split = defaultdict(set)
            for r in sd_rows:
                sd_split[r["split"]].add(r["sha256"])
            for r in sdd_rows:
                sdd_split[r["split"]].add(r["sha256"])
            combined_split = defaultdict(set)
            for r in rows:
                combined_split[r["split"]].add(r["sha256"])
            for split in ("train", "val", "test"):
                expected_union = sd_split[split] | sdd_split[split]
                assert combined_split[split] == expected_union, f"Combined split {split} != ShrimpDB | ShrimpDiseaseDB"
                assert sd_split[split] & sdd_split[split] == set(), f"Cross-source hash overlap in split {split}"

        # no SHA-256 crosses splits
        by_hash: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            by_hash[r["split"]].add(r["sha256"])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            assert by_hash[left].isdisjoint(by_hash[right]), f"Hash leakage {left}/{right}"

        # counts match summary per split/class
        counts = defaultdict(lambda: defaultdict(int))
        for r in rows:
            counts[r["split"]][r["class_name"]] += 1
        for split, per_cls in exp["distribution"].items():
            for cls, n in per_cls.items():
                assert counts[split].get(cls, 0) == n, f"{exp_name}: {split}/{cls} count mismatch"

    LOGGER.info("All split protocol assertions passed")
    return 0


import json

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate split protocol")
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Split protocol failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Split protocol error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def test_main() -> None:
    assert run() == 0
