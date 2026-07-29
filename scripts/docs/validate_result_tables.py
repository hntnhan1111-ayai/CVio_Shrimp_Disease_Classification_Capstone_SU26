#!/usr/bin/env python3
"""Validate that result documentation exposes the required provenance fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_RESULT = [
    "Experiment ID",
    "Track",
    "Dataset / split",
    "Model",
    "Seed",
    "Primary metric",
    "Checkpoint",
    "Config",
    "Status",
]
REQUIRED_PROVENANCE = [
    "Experiment ID",
    "Metric",
    "Value",
    "Dataset",
    "Split",
    "Seed",
    "Checkpoint",
    "Config",
    "Evaluation command",
    "Evidence file",
    "Status",
]


def require_header(path: Path, columns: list[str]) -> list[str]:
    content = path.read_text(encoding="utf-8")
    missing = [column for column in columns if column not in content]
    return [
        f"{path.relative_to(ROOT)}: missing column {column!r}" for column in missing
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    failures = require_header(ROOT / "docs" / "RESULTS.md", REQUIRED_RESULT)
    failures += require_header(
        ROOT / "docs" / "audits" / "metric_provenance.md", REQUIRED_PROVENANCE
    )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Result and metric-provenance tables expose all required fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
