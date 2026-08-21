"""Run one config-locked Ultralytics instance-segmentation benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from cvio_shrimp_seg.benchmark import (  # noqa: E402
    ConfigurationError,
    build_run_plan,
    load_config,
    verify_parameter_count,
    verify_reference_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ultralytics model id from config.yaml")
    parser.add_argument("--data-yaml", help="Override CVIO_MRTU_DATA_YAML")
    parser.add_argument("--smoke", action="store_true", help="Run one 64px epoch with batch size 1")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without importing torch")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        reference = verify_reference_source(config)
        plan = build_run_plan(config, args.model, data_yaml=args.data_yaml, smoke=args.smoke)
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    summary = {
        "model": plan.model_id,
        "checkpoint": plan.checkpoint,
        "reference_source": str(reference),
        "expected_params_millions": plan.expected_params_millions,
        "parameter_limit_millions": plan.parameter_limit_millions,
        "train_args": plan.train_args,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    vendored = REPO_ROOT / "yolov11n_attention" / "ultralytics"
    if vendored.is_dir():
        sys.path.insert(0, str(vendored))
    from ultralytics import YOLO

    model = YOLO(plan.checkpoint)
    actual = sum(parameter.numel() for parameter in model.model.parameters())
    actual_millions = verify_parameter_count(actual, plan)
    print(f"Verified parameter count: {actual_millions:.3f}M")
    environment = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": plan.model_id,
        "checkpoint": plan.checkpoint,
        "parameter_count": actual,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "pillow", "PyYAML")
        },
        "train_args": plan.train_args,
    }
    environment_path = ROOT / "artifacts" / "generated" / "environments" / f"{plan.run_name}.json"
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(f"Environment record: {environment_path}")
    model.train(**plan.train_args)


if __name__ == "__main__":
    main()
