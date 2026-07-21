#!/usr/bin/env python3
"""Create and verify seed-42 split manifests for all experiments."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from cvio_asl_ldam.data.make_split import create_split, materialize_split_tree
from cvio_asl_ldam.utils.io import load_yaml, write_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_combined_split(
    shrimpdb_rows: list[dict[str, str]],
    shrimpdisease_rows: list[dict[str, str]],
    expected: dict[str, dict[str, int]],
) -> dict[str, Any]:
    combined: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in shrimpdb_rows + shrimpdisease_rows:
        combined[row["split"]][row["class_name"]] += 1

    if combined != expected:
        raise AssertionError(f"Combined split mismatch: expected={expected} actual={dict(combined)}")
    return {"split_distribution": {s: dict(v) for s, v in combined.items()}, "total": sum(len(shrimpdb_rows), len(shrimpdisease_rows))}


def run(config_path: str | Path = STUDY_CONFIG) -> int:
    study = load_yaml(config_path)
    seed = int(study.get("study", {}).get("seed", 42))
    experiments = study.get("experiments", {})

    for exp_name, exp_cfg in experiments.items():
        dataset_key = exp_cfg.get("dataset", exp_name)
        if dataset_key == "shrimpdb":
            ds_yaml = Path("configs/datasets/shrimpdb.yaml")
        elif dataset_key == "combined4":
            ds_yaml = Path("configs/datasets/combined4.yaml")
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

        LOGGER.info("Preparing split for experiment=%s dataset=%s", exp_name, dataset_key)
        output_dir = Path(f"experiments/{exp_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "split_manifest_seed42_generated.csv"

        if dataset_key == "shrimpdb":
            manifest = create_split(ds_yaml, study["datasets"]["shrimpdb"]["root"], output_dir, seed)
            rows = _read_csv(manifest)
            counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for row in rows:
                counts[row["split"]][row["class_name"]] += 1
            audit = {
                "experiment": exp_name,
                "dataset": dataset_key,
                "seed": seed,
                "manifest": str(manifest),
                "distribution": {s: dict(v) for s, v in counts.items()},
                "total": len(rows),
            }
        elif dataset_key == "combined4":
            shrimpdb_root = Path(study["datasets"]["shrimpdb"]["root"])
            shrimpdisease_root = Path(study["datasets"]["shrimpdiseasedb"]["root"])

            shrimpdb_manifest = create_split(
                Path("configs/datasets/shrimpdb.yaml"),
                shrimpdb_root,
                output_dir,
                seed,
            )
            shrimpdisease_manifest = create_split(
                Path("configs/datasets/shrimpdiseasedb.yaml"),
                shrimpdisease_root,
                output_dir,
                seed,
            )

            sd_rows = _read_csv(shrimpdb_manifest)
            sdd_rows = _read_csv(shrimpdisease_manifest)

            combined_rows: list[dict[str, str]] = []
            for row in sd_rows:
                new_row = dict(row)
                new_row["source_dataset"] = "shrimpdb"
                combined_rows.append(new_row)
            for row in sdd_rows:
                new_row = dict(row)
                new_row["source_dataset"] = "shrimpdiseasedb"
                combined_rows.append(new_row)

            combined_rows.sort(key=lambda r: (r["split"], r["label"], r["rel_path"]))

            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
                writer.writeheader()
                writer.writerows(combined_rows)

            expected_dist = {split: {cls: 0 for cls in exp_cfg.get("class_names", [])} for split in ("train", "val", "test")}
            for row in sd_rows:
                expected_dist[row["split"]][row["class_name"]] = expected_dist[row["split"]].get(row["class_name"], 0) + 1
            for row in sdd_rows:
                expected_dist[row["split"]][row["class_name"]] = expected_dist[row["split"]].get(row["class_name"], 0) + 1

            audit = _validate_combined_split(sd_rows, sdd_rows, expected_dist)
            audit.update({
                "experiment": exp_name,
                "dataset": dataset_key,
                "seed": seed,
                "manifest": str(manifest_path),
                "shrimpdb_manifest": str(shrimpdb_manifest),
                "shrimpdiseasedb_manifest": str(shrimpdisease_manifest),
            })

        audit_path = output_dir / "split_audit_seed42.json"
        write_json(audit_path, audit)
        LOGGER.info("Wrote split audit to %s", audit_path)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare split manifests")
    parser.add_argument("--config", default=str(STUDY_CONFIG))
    args = parser.parse_args()
    try:
        return run(args.config)
    except Exception as exc:
        LOGGER.error("Split preparation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
