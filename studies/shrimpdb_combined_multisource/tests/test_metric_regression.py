#!/usr/bin/env python3
"""Assert metric regression matches verified values."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> int:
    tolerance = 1e-6

    shrimpdb_best = _load_metrics(Path("artifacts/evaluation/shrimpdb3/best_pt/metrics_raw.json"))
    assert abs(shrimpdb_best["accuracy"] - 0.9148936170212766) < tolerance, "ShrimpDB-3 accuracy mismatch"
    assert abs(shrimpdb_best["macro_f1"] - 0.9129374237733371) < tolerance, "ShrimpDB-3 macro_f1 mismatch"

    combined_best = _load_metrics(Path("artifacts/evaluation/combined4/best_pt/metrics_raw.json"))
    assert abs(combined_best["accuracy"] - 0.8318181818181818) < tolerance, "Combined-4 accuracy mismatch"
    assert abs(combined_best["macro_f1"] - 0.8218010087330127) < tolerance, "Combined-4 macro_f1 mismatch"

    LOGGER.info("All metric regression assertions passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate metric regression")
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Metric regression failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Metric regression error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

def test_main() -> None:
    assert run() == 0

