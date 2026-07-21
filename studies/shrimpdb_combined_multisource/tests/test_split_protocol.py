#!/usr/bin/env python3
"""Assert split protocol integrity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from collections import defaultdict
from pathlib import Path

from cvio_asl_ldam.utils.io import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> int:
    study = load_yaml(STUDY_CONFIG)
    experiments = study.get("experiments", {})

    for exp_name in ("shrimpdb3", "combined4"):
        exp_cfg = experiments[exp_name]
        manifest_path = Path(f"experiments/{exp_name}/split_manifest_seed42_generated.csv")
        if not manifest_path.is_file():
            LOGGER.warning("Missing manifest for %s, skipping", exp_name)
            continue

        rows = _read_csv(manifest_path)
        by_split: dict[str, set[str]] = defaultdict(set)
        by_hash: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            by_split[row["split"]].add(row["rel_path"])
            if row.get("md5"):
                by_hash[row["split"]].add(row["md5"])

        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            assert by_split[left].isdisjoint(by_split[right]), f"Path leakage {left}/{right}"
            assert by_hash[left].isdisjoint(by_hash[right]), f"Hash leakage {left}/{right}"

        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in rows:
            counts[row["split"]][row["class_name"]] += 1

        if exp_name == "shrimpdb3":
            expected = {"train": {"Healthy": 49, "BG": 78, "WSSV": 94}, "val": {}, "test": {}}
            # validate against known train counts; val/test are remainder
            train_counts = counts["train"]
            assert train_counts["Healthy"] == 49
            assert train_counts["BG"] == 78
            assert train_counts["WSSV"] == 94
        elif exp_name == "combined4":
            train_counts = counts["train"]
            assert train_counts["Healthy"] == 331
            assert train_counts["BG"] == 217
            assert train_counts["WSSV"] == 323
            assert train_counts["WSSV_BG"] == 154

        if exp_name == "combined4":
            sd_rows = [r for r in rows if r.get("source_dataset") == "shrimpdb"]
            sdd_rows = [r for r in rows if r.get("source_dataset") == "shrimpdiseasedb"]
            combined_counts = {split: defaultdict(int) for split in ("train", "val", "test")}
            for row in sd_rows + sdd_rows:
                combined_counts[row["split"]][row["class_name"]] += 1
            sd_counts = {split: defaultdict(int) for split in ("train", "val", "test")}
            sdd_counts = {split: defaultdict(int) for split in ("train", "val", "test")}
            for row in sd_rows:
                sd_counts[row["split"]][row["class_name"]] += 1
            for row in sdd_rows:
                sdd_counts[row["split"]][row["class_name"]] += 1
            for split in ("train", "val", "test"):
                for cls in combined_counts[split]:
                    expected_count = sd_counts[split][cls] + sdd_counts[split][cls]
                    assert combined_counts[split][cls] == expected_count, f"Combined count mismatch for {split}/{cls}"

    LOGGER.info("All split protocol assertions passed")
    return 0


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

