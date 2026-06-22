#!/usr/bin/env python3
"""Package a small reviewer-friendly artifact bundle.

Includes ONLY small allowed artifacts: README snapshot, configs, split manifest,
result tables, figures, export README, TFLite files (if present/allowed), and
sanity checks. Excludes raw datasets, training runs, .pt weights, ONNX,
SavedModel folders, corrupted image caches, and notebook checkpoints.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path.cwd()

INCLUDE_DIRS = [
    "configs",
    "artifacts/manifests",
    "artifacts/tables",
    "artifacts/figures",
    "docs",
]
INCLUDE_FILES = [
    "README.md",
    "CITATION.cff",
    "requirements.txt",
    "requirements-export-litert.txt",
    "artifacts/manifests/split_manifest_seed42.csv",
    "export/README.md",
    "export/tflite_sanity_check.json",
    "export/export_environment.json",
]
INCLUDE_GLOBS = ["export/*.tflite"]

EXCLUDE_SUFFIXES = {".pt", ".pth", ".onnx", ".pb", ".npy", ".npz", ".engine", ".safetensors"}
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", "datasets", "runs", "wandb"}


def _should_exclude(path: Path) -> bool:
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    for part in path.parts:
        if part in EXCLUDE_PARTS:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/review_package.zip")
    parser.add_argument("--max-tflite-mib", type=float, default=100.0,
                        help="Skip TFLite files larger than this (GitHub blocks >100 MiB).")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    added = 0
    skipped: list[str] = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in INCLUDE_DIRS:
            base = PROJECT_ROOT / entry
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and not _should_exclude(path):
                    archive.write(path, path.relative_to(PROJECT_ROOT))
                    added += 1
        for entry in INCLUDE_FILES:
            path = PROJECT_ROOT / entry
            if path.is_file() and not _should_exclude(path):
                archive.write(path, path.relative_to(PROJECT_ROOT))
                added += 1
        for pattern in INCLUDE_GLOBS:
            for path in (PROJECT_ROOT).glob(pattern):
                size_mib = path.stat().st_size / (1024 * 1024)
                if size_mib > args.max_tflite_mib:
                    skipped.append(f"{path} ({size_mib:.1f} MiB > {args.max_tflite_mib} MiB)")
                    continue
                archive.write(path, path.relative_to(PROJECT_ROOT))
                added += 1

    print(f"Packaged {added} files into {out} ({out.stat().st_size / 1024:.0f} KiB)")
    if skipped:
        print("Skipped (size limit):")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
