"""Record a small, portable environment report without importing a dataset."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

PACKAGES = ("torch", "ultralytics", "numpy", "PyYAML")
ROOT = Path(__file__).resolve().parents[1]


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/generated/environment.json",
        help="Output path for the environment report.",
    )
    args = parser.parse_args()
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in PACKAGES},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
