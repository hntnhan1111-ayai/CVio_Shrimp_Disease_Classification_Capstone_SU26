#!/usr/bin/env python3
"""Resolve the dataset, audit class counts, and create the fixed seed-42 split.

Workflow:
  1. Resolve `--data-root` or download via KaggleHub (`uynnhy/processed-images`).
  2. Robustly detect the class root (recursively searches for the 4 class dirs).
  3. Validate class counts (403/198/328/220 = 1149).
  4. Create the fixed seed-42 image-level split (804/172/173).
  5. Materialize `runs/prepared_seed42/{train,val,test}` in Ultralytics layout.
  6. Write `artifacts/manifests/split_manifest_seed42.csv` and
     `artifacts/metadata/dataset_audit.json`.

Does not train. Skips if already complete unless `--force`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from cvio_asl_ldam.data.audit_dataset import audit_dataset, resolve_dataset_root
from cvio_asl_ldam.data.make_split import create_split, materialize_split_tree
from cvio_asl_ldam.utils.io import load_yaml, write_json

PROJECT_ROOT = Path.cwd()
DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset" / "shrimpdiseasebd_seed42.yaml"
MANIFEST = PROJECT_ROOT / "artifacts" / "manifests" / "split_manifest_seed42.csv"
DATASET_AUDIT = PROJECT_ROOT / "artifacts" / "metadata" / "dataset_audit.json"
PREPARED = PROJECT_ROOT / "runs" / "prepared_seed42"
KAGGLE_DATASET = "uynnhy/processed-images"


def _download_via_kagglehub() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Install it or pass --data-root pointing "
            "to a folder containing the four class directories."
        ) from exc
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"KaggleHub downloaded to: {path}")
    return Path(path)


def _is_complete() -> bool:
    return (
        MANIFEST.is_file()
        and DATASET_AUDIT.is_file()
        and (PREPARED / "train").is_dir()
        and (PREPARED / "val").is_dir()
        and (PREPARED / "test").is_dir()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Dataset root containing the four class folders.")
    parser.add_argument("--config", default=str(DATASET_CONFIG))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--skip-if-complete", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    if args.skip_if_complete and not args.force and _is_complete():
        print("Split already complete. Skipping (use --force to re-run).")
        print(f"  manifest: {MANIFEST}")
        print(f"  prepared: {PREPARED}")
        return

    if args.force:
        if PREPARED.exists():
            shutil.rmtree(PREPARED)

    config = load_yaml(args.config)
    data_root = Path(args.data_root) if args.data_root else _download_via_kagglehub()

    # Robust recursive detection of the class root + count validation.
    root, rows, audit = audit_dataset(data_root, args.config, verify_images=True)
    DATASET_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    write_json(DATASET_AUDIT, audit)
    print(f"Dataset root: {root}")
    print(f"Class counts: {audit['class_counts']} (total {audit['total_images']})")

    manifest_dir = MANIFEST.parent
    manifest = create_split(args.config, root, manifest_dir, args.seed)
    # create_split writes split_manifest_seed42_generated.csv; copy to canonical name.
    canonical = manifest_dir / "split_manifest_seed42.csv"
    if manifest != canonical:
        shutil.copy2(manifest, canonical)

    output_root = Path(args.output_dir)
    prepared = materialize_split_tree(root, canonical, output_root / "prepared_seed42")
    print(f"Materialized split tree: {prepared}")

    # Final completion record.
    write_json(
        Path("artifacts/metadata/split_completion.json"),
        {
            "status": "complete",
            "seed": args.seed,
            "manifest": str(canonical),
            "dataset_audit": str(DATASET_AUDIT),
            "prepared_root": str(prepared),
        },
    )
    print(f"Done. Manifest: {canonical}")


if __name__ == "__main__":
    main()
