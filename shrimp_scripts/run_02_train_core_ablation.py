"""Run the main 12-run ConvNeXt/ShrimpXNet and YOLOv26m-cls core ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shrimp_scripts import config
from shrimp_scripts.dataset import load_split_manifest, load_yolo_manifest
from shrimp_scripts.models_torch import list_core_torch_runs, train_torch_with_fallback
from shrimp_scripts.models_yolo import list_core_yolo_runs, train_yolo_with_fallback
from shrimp_scripts.utils import ensure_dir, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 12-run main paper core ablation.")
    parser.add_argument("--output_dir", default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--list_runs", action="store_true")
    return parser.parse_args()


def planned_runs(smoke_test: bool = False) -> list[dict]:
    return list_core_torch_runs(smoke_test=smoke_test) + list_core_yolo_runs(smoke_test=smoke_test)


def main() -> None:
    args = parse_args()
    rows = planned_runs(smoke_test=args.smoke_test)
    if args.list_runs:
        print(json.dumps({"run_count": len(rows), "runs": rows}, indent=2))
        return
    output_dir = ensure_dir(args.output_dir)
    split_manifest = load_split_manifest(output_dir)
    yolo_manifest = load_yolo_manifest(output_dir)
    save_csv(__import__("pandas").DataFrame(rows), output_dir / "experiment_plan_core_ablation.csv")
    results = []
    for row in rows:
        condition = {"condition_key": row["condition_key"], "loss_key": row["loss_key"], "randaugment": row["randaugment"]}
        if row["backend"] == "ultralytics":
            results.append(train_yolo_with_fallback(row["model"], condition, yolo_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test))
        else:
            results.append(train_torch_with_fallback(row["model_key"], row["model"], condition, split_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test))
    print(json.dumps({"completed_or_attempted": len(results), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
