#!/usr/bin/env python3
"""Validate required artifact directories and exclusion rules."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

REQUIRED_DIRS = [
    "artifacts/figures",
    "artifacts/manifests",
    "artifacts/metadata",
    "artifacts/reports",
    "artifacts/tables",
    "experiments/shrimpdb3",
    "experiments/combined4",
    "runs",
    "evaluation/shrimpdb3",
    "evaluation/combined4",
    "model_registry",
    "logs",
]
EXCLUDE_DIRS = {"raw_datasets", ".venv", "__pycache__", ".git", "node_modules"}
MAX_FILE_SIZE_MB = 100


def run(root: Path = Path(".")) -> int:
    failures = []
    for d in REQUIRED_DIRS:
        if not (root / d).is_dir():
            failures.append(f"Missing required directory: {d}")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(ex in path.parts for ex in EXCLUDE_DIRS):
            failures.append(f"Excluded path should not be tracked: {path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            failures.append(f"File exceeds {MAX_FILE_SIZE_MB} MB: {path} ({size_mb:.1f} MB)")

    if failures:
        for f in failures:
            LOGGER.error(f)
        return 1
    LOGGER.info("Artifact tree validation passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate artifact tree")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    try:
        return run(Path(args.root))
    except Exception as exc:
        LOGGER.error("Validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
