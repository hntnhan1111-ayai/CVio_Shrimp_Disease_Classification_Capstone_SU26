#!/usr/bin/env python3
"""Assert metric regression matches verified artifact values."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def _check(name: str, raw: dict, expected: dict) -> None:
    for key, value in expected.items():
        actual = raw.get(key)
        assert _approx(actual, value), f"{name}: {key} expected={value} actual={actual}"


def run() -> int:
    root = Path("artifacts/evaluation")

    s3_best = _load(root / "shrimpdb3/best_pt/metrics_raw.json")
    _check("ShrimpDB-3 best.pt", s3_best, {
        "n": 47,
        "accuracy": 0.9148936170212766,
        "balanced_accuracy": 0.915530303030303,
        "macro_f1": 0.9129374237733371,
        "weighted_f1": 0.916276925103748,
        "cohen_kappa": 0.8694444444444445,
        "top1_accuracy": 0.9148936170212766,
        "ece_15_bins": 0.19298748513485522,
    })
    for cls in ("Healthy", "BG", "WSSV"):
        cp = s3_best["classification_report"][cls]
        for k in ("precision", "recall", "f1-score", "support"):
            assert isinstance(cp[k], (int, float)), f"ShrimpDB-3/{cls}/{k} missing"

    c4_best = _load(root / "combined4/best_pt/metrics_raw.json")
    _check("Combined-4 best.pt", c4_best, {
        "n": 220,
        "accuracy": 0.8318181818181818,
        "balanced_accuracy": 0.8385461760461761,
        "macro_f1": 0.8218010087330127,
        "weighted_f1": 0.8318690217921992,
        "cohen_kappa": 0.7714510332434861,
        "top1_accuracy": 0.8318181818181818,
        "ece_15_bins": 0.33457528420469973,
        "top2_accuracy": 0.9545454545454546,
    })

    c4_last = _load(root / "combined4/last_pt/metrics_raw.json")
    _check("Combined-4 last.pt", c4_last, {
        "n": 220,
        "accuracy": 0.8818181818181818,
        "balanced_accuracy": 0.8817730880230881,
        "macro_f1": 0.8760915764930275,
        "weighted_f1": 0.8813121770758151,
    })

    app = _load(Path("artifacts/final_application_model/test_metrics.json"))
    assert app["application_checkpoint_sha256"] == "9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606"
    assert _approx(app["accuracy"], c4_best["accuracy"])
    assert _approx(app["macro_f1"], c4_best["macro_f1"])

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
