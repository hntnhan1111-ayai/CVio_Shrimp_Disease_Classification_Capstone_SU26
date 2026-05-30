"""Prepare dataset manifests, fixed split, and YOLO classification folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shrimp_scripts import config
from shrimp_scripts.dataset import prepare_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, audit, split, and prepare the processed shrimp dataset.")
    parser.add_argument("--output_dir", default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_rebuild_yolo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        payload = {
            "dry_run": True,
            "dataset_id": config.DATASET_ID,
            "download_call": 'kagglehub.dataset_download("uynnhy/processed-images")',
            "expected_counts": config.EXPECTED_NAME_COUNTS,
            "expected_total_images": config.EXPECTED_TOTAL_IMAGES,
            "output_dir": args.output_dir,
            "planned_outputs": [
                "source_processed_manifest_with_md5.csv",
                "fixed_split_manifest_seed42_with_md5.csv",
                "yolo_split_manifest_seed42_with_md5.csv",
                "yolo_fixed_dataset/",
            ],
        }
    else:
        payload = prepare_all(args.output_dir, dry_run=False, force_rebuild_yolo=args.force_rebuild_yolo)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
