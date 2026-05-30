"""Run YOLO classification family comparison for m-size variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shrimp_scripts import config
from shrimp_scripts.dataset import load_yolo_manifest
from shrimp_scripts.models_yolo import list_yolo_family_runs, probe_yolo_availability, train_yolo_with_fallback
from shrimp_scripts.progress import log_event
from shrimp_scripts.utils import ensure_dir, save_csv, write_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO classification family comparison runs.")
    parser.add_argument("--output_dir", default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--list_runs", action="store_true")
    parser.add_argument("--skip_probe", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list_yolo_family_runs(smoke_test=args.smoke_test)
    if args.list_runs:
        print(json.dumps({"run_count": len(rows), "runs": rows}, indent=2))
        return
    output_dir = ensure_dir(args.output_dir)
    if args.progress:
        log_event("Starting YOLO family training.", output_dir=output_dir, extra={"planned_runs": len(rows), "skip_probe": args.skip_probe})
    yolo_manifest = load_yolo_manifest(output_dir)
    save_csv(pd.DataFrame(rows), output_dir / "experiment_plan_yolo_family.csv")
    availability = None if args.skip_probe else probe_yolo_availability(output_dir, progress_enabled=args.progress)
    results = []
    for index, row in enumerate(rows, start=1):
        if args.progress:
            log_event("Starting planned YOLO family run.", run_id=row["run_id"], output_dir=output_dir, extra={
                "index": index,
                "total": len(rows),
                "model": row["model"],
                "condition": row["condition_key"],
                "loss": row["loss_key"],
                "randaugment": row["randaugment"],
            })
        if availability is not None:
            match = availability[availability["model"] == row["model"]]
            if not match.empty and match.iloc[0]["availability_status"] != "available_pretrained":
                reason = str(match.iloc[0].get("error", "unavailable"))
                run_dir = ensure_dir(output_dir / "runs" / row["run_id"])
                write_status(run_dir, "skipped", run_id=row["run_id"], error=reason)
                results.append({"run_id": row["run_id"], "status": "skipped", "error": reason})
                if args.progress:
                    log_event("Skipping unavailable YOLO model.", level="WARNING", run_id=row["run_id"], output_dir=output_dir, extra={"reason": reason})
                continue
        condition = {"condition_key": row["condition_key"], "loss_key": row["loss_key"], "randaugment": row["randaugment"]}
        result = train_yolo_with_fallback(row["model"], condition, yolo_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test, progress_enabled=args.progress)
        results.append(result)
        if args.progress:
            log_event("Finished planned YOLO family run.", run_id=row["run_id"], output_dir=output_dir, extra={"status": result.get("status", "completed")})
    if args.progress:
        log_event("YOLO family script completed.", output_dir=output_dir, extra={"attempted": len(results)})
    print(json.dumps({"completed_or_attempted": len(results), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
