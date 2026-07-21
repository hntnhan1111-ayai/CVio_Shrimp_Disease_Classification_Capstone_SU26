#!/usr/bin/env python3
"""Verify runtime environment matches study requirements."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from cvio_asl_ldam.utils.io import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

REQUIRED_PACKAGES: dict[str, str] = {
    "torch": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "ultralytics": "8.4.75",
    "pandas": "2.3.3",
    "scikit_learn": "1.6.1",
    "matplotlib": "3.10.0",
    "PIL": "11.3.0",
    "yaml": "6.0.3",
}


def _check_package(name: str, expected: str) -> tuple[bool, str]:
    try:
        if name == "PIL":
            import PIL

            actual = PIL.__version__
        elif name == "yaml":
            import yaml

            actual = yaml.__version__
        elif name == "sklearn":
            import sklearn

            actual = sklearn.__version__
        else:
            module = __import__(name)
            actual = getattr(module, "__version__", "unknown")
    except ImportError as exc:
        return False, f"missing ({exc})"
    if actual == expected:
        return True, actual
    LOGGER.warning("Version mismatch for %s: expected=%s actual=%s", name, expected, actual)
    return True, f"{actual} (mismatch)"


def _check_cuda(expected: bool) -> bool:
    try:
        import torch

        available = torch.cuda.is_available()
        if expected and not available:
            LOGGER.error("CUDA expected but not available")
            return False
        if available:
            LOGGER.info("CUDA available: devices=%d", torch.cuda.device_count())
        else:
            LOGGER.info("CUDA not available (cpu fallback)")
        return True
    except ImportError as exc:
        LOGGER.error("torch import failed: %s", exc)
        return False


def run(config_path: str | Path) -> int:
    config = load_yaml(config_path)
    study = config.get("study", {})
    runtime = config.get("runtime", {})
    LOGGER.info("Study: %s (seed=%s)", study.get("name"), study.get("seed"))
    LOGGER.info("Split: train=%s val=%s test=%s", study.get("split", {}).get("train"), study.get("split", {}).get("val"), study.get("split", {}).get("test"))

    package_ok = True
    package_summary: dict[str, str] = {}
    for name, expected in REQUIRED_PACKAGES.items():
        ok, actual = _check_package(name, expected)
        package_summary[name] = actual
        if not ok:
            package_ok = False

    cuda_ok = _check_cuda(bool(runtime.get("cuda_available", False)))

    env_summary = {
        "study": study.get("name"),
        "seed": study.get("seed"),
        "split": study.get("split"),
        "model": config.get("training", {}).get("model"),
        "method": config.get("training", {}).get("method"),
        "packages": package_summary,
        "cuda_available": runtime.get("cuda_available", False),
        "gpu_count": runtime.get("gpu_count", 0),
        "passed": package_ok and cuda_ok,
    }
    out_path = Path("artifacts/metadata/environment.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"{__import__('json').dumps(env_summary, indent=2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote environment summary to %s", out_path)

    if env_summary["passed"]:
        LOGGER.info("Environment check PASSED")
        return 0
    LOGGER.error("Environment check FAILED")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Check study environment")
    parser.add_argument("--config", default="configs/study.yaml", help="Path to study.yaml")
    args = parser.parse_args()
    sys.exit(run(args.config))


if __name__ == "__main__":
    main()
