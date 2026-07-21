#!/usr/bin/env python3
"""Validate required artifact directories and exclusion rules.

Only flags content that is tracked in git. On-disk caches, virtual
environments, and build artifacts are not the validator's concern.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
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
    "artifacts/experiments/shrimpdb3",
    "artifacts/experiments/combined4",
    "artifacts/evaluation/shrimpdb3",
    "artifacts/evaluation/combined4",
    "artifacts/logs",
    "artifacts/audit",
    "artifacts/final_application_model",
    "model_registry",
]
EXCLUDE_DIRS = {
    "raw_datasets",
    ".venv",
    "venv",
    "__pycache__",
    ".ipynb_checkpoints",
    ".git",
    "node_modules",
    "downloaded_datasets",
    "kaggle_cache",
    "prepared",
}
MAX_FILE_SIZE_MB = 50.0
ALLOWED_PT_FILES = {
    "artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt",
}


def _tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(root),
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8", errors="replace")
    return [p for p in out.split("\x00") if p]


def run(root: Path = Path(".")) -> int:
    failures: list[str] = []

    for d in REQUIRED_DIRS:
        if not (root / d).is_dir():
            failures.append(f"Missing required directory: {d}")

    try:
        tracked = _tracked_files(root)
    except Exception as exc:  # pragma: no cover - defensive
        failures.append(f"git ls-files failed: {exc}")
        tracked = []

    for rel in tracked:
        parts = rel.split("/")
        if any(ex in parts for ex in EXCLUDE_DIRS):
            failures.append(f"Tracked forbidden path: {rel}")
        if rel.lower().endswith(".pt") and rel not in ALLOWED_PT_FILES:
            failures.append(f"Unexpected tracked .pt file: {rel}")
        full = root / rel
        if full.is_file():
            size_mb = full.stat().st_size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                failures.append(
                    f"Tracked file exceeds {MAX_FILE_SIZE_MB:.0f} MB: {rel} ({size_mb:.1f} MB)"
                )

    if failures:
        for f in failures:
            LOGGER.error(f)
        return 1
    LOGGER.info("Artifact tree validation passed (tracked files: %d)", len(tracked))
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
