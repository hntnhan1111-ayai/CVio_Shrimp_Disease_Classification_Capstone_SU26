#!/usr/bin/env python3
"""Verify tracked files contain no raw datasets, venvs, caches, results ZIPs, or
unapproved checkpoints; and that no tracked file exceeds 50 MB."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

FORBIDDEN_DIRS = {
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
FORBIDDEN_SUFFIXES = (".zip",)
ALLOWED_PT = {
    "artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt",
}
MAX_BYTES = 50 * 1024 * 1024
NUL = chr(0)


def run() -> int:
    raw = subprocess.run(["git", "ls-files", "-z"], cwd=".", check=True, capture_output=True).stdout.decode("utf-8", "replace")
    tracked = [p for p in raw.split(NUL) if p]
    forbidden = []
    oversize = []
    unapproved_pt = []
    for rel in tracked:
        parts = rel.split("/")
        if any(p in FORBIDDEN_DIRS for p in parts):
            forbidden.append(rel)
            continue
        if rel.lower().endswith(FORBIDDEN_SUFFIXES):
            forbidden.append(rel)
            continue
        if rel.lower().endswith(".pt") and rel not in ALLOWED_PT:
            unapproved_pt.append(rel)
        full = Path(rel)
        if full.is_file():
            size = full.stat().st_size
            if size > MAX_BYTES:
                oversize.append((rel, size))

    assert not forbidden, f"Tracked forbidden files: {forbidden}"
    assert not unapproved_pt, f"Unapproved .pt files tracked: {unapproved_pt}"
    assert not oversize, f"Tracked files exceed 50 MB: {oversize}"

    LOGGER.info("Forbidden files / large files / unapproved .pt all clean (%d tracked files)", len(tracked))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Forbidden-files check failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Forbidden-files check error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def test_main() -> None:
    assert run() == 0
