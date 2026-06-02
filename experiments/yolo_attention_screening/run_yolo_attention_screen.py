"""Run YOLOv26m-cls attention module screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shrimp_scripts.dataset import load_yolo_manifest
from shrimp_scripts.progress import log_event
from shrimp_scripts.utils import ensure_dir, save_csv, write_json

from experiments.yolo_attention_screening.screening_lib import (
    DEFAULT_OUTPUT_DIR,
    YOLO_ATTENTION_SCREEN_EPOCHS,
    attention_supported_before_training,
    list_attention_runs,
    mark_attention_skipped,
    run_attention_module_unit_test,
    run_model_injection_test,
    train_attention_row,
    validate_attention_resume_rows,
    write_attention_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen attention modules on YOLOv26m-cls with CE and RandAugment disabled.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=YOLO_ATTENTION_SCREEN_EPOCHS)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--list_runs", action="store_true")
    parser.add_argument("--validate_resume", action="store_true")
    parser.add_argument("--self_test_attention_modules", action="store_true")
    parser.add_argument("--self_test_model_injection", action="store_true")
    parser.add_argument("--only_attention", default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    return parser.parse_args()


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = rows
    if args.only_attention:
        selected = [row for row in selected if row["attention_key"] == args.only_attention]
    if args.start > 1:
        selected = selected[args.start - 1 :]
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]
    return selected


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    all_rows = list_attention_runs(epochs=args.epochs)
    rows = select_rows(all_rows, args)
    write_attention_plan(output_dir, all_rows)
    if not rows:
        raise SystemExit(f"No attention screening runs selected. only_attention={args.only_attention!r}, start={args.start}, limit={args.limit!r}")
    if args.list_runs:
        print(json.dumps({"run_count": len(rows), "available_before_filter": len(all_rows), "runs": rows}, indent=2))
        return
    if args.validate_resume:
        summary = validate_attention_resume_rows(rows, output_dir)
        print(json.dumps({"resume_validation": summary}, indent=2, default=str))
        return
    if args.self_test_attention_modules:
        result = run_attention_module_unit_test(output_dir)
        print(json.dumps(result, indent=2, default=str))
        if result.get("status") != "passed":
            raise SystemExit(1)
        return
    if args.self_test_model_injection:
        result = run_model_injection_test(output_dir, rows)
        print(json.dumps(result, indent=2, default=str))
        if result.get("status") != "passed":
            raise SystemExit(1)
        return
    if args.progress:
        log_event("Starting YOLOv26m attention screening.", output_dir=output_dir, extra={"planned_runs": len(rows), "epochs": args.epochs})
    yolo_manifest = load_yolo_manifest(output_dir)
    results = []
    for index, row in enumerate(rows, start=1):
        if args.progress:
            log_event("Starting attention screening run.", run_id=row["run_id"], output_dir=output_dir, extra={"index": index, "total": len(rows), "attention_key": row["attention_key"]})
        supported, support_reason = attention_supported_before_training(row["attention_key"])
        if not supported:
            result = mark_attention_skipped(output_dir, row, support_reason)
            results.append(result)
            if args.progress:
                log_event("Skipping unsupported attention module before training.", level="WARNING", run_id=row["run_id"], output_dir=output_dir, extra={"reason": support_reason})
            continue
        result = train_attention_row(row, yolo_manifest, output_dir, resume=args.resume, progress_enabled=args.progress)
        results.append(result)
        if args.progress:
            log_event("Finished attention screening run.", run_id=row["run_id"], output_dir=output_dir, extra={"status": result.get("status", "completed")})
    write_json(output_dir / "attention_screening_run_results.json", {"planned_count": len(rows), "results": results})
    save_csv(pd.DataFrame(results), output_dir / "attention_screening_run_results.csv")


if __name__ == "__main__":
    main()
