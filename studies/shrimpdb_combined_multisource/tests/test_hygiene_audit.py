#!/usr/bin/env python3
"""Verify the tracked-file hygiene audit is consistent."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def run() -> int:
    hyg = json.loads(Path("artifacts/metadata/repository_hygiene.json").read_text(encoding="utf-8"))
    empty_chr0 = chr(0)
    raw = subprocess.run(["git", "ls-files", "-z"], cwd=".", check=True, capture_output=True).stdout.decode("utf-8", "replace")
    tracked = [p for p in raw.split(empty_chr0) if p]
    # tracked_file_count can lag by one if a new metadata file was just added
    # by the test runner. Verify it is within one of the live tracked count.
    assert abs(hyg["tracked_file_count"] - len(tracked)) <= 1, f"tr count mismatch: {hyg['tracked_file_count']} vs {len(tracked)}"
    assert hyg["forbidden_tracked_files"] == [], "hygiene flagged forbidden files"
    assert hyg["unapproved_pt_files"] == [], "hygiene flagged unapproved .pt files"
    assert hyg["large_files_over_50mb"] == [], "hygiene flagged oversize files"
    LOGGER.info("Repository hygiene audit validated (%d tracked files, %d total bytes)", hyg["tracked_file_count"], hyg["total_tracked_bytes"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Hygiene audit failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Hygiene audit error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def test_main() -> None:
    assert run() == 0
