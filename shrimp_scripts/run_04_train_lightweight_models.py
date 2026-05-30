"""Run lightweight/mobile-friendly model comparison with resume and chunking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shrimp_scripts import config
from shrimp_scripts.dataset import load_split_manifest
from shrimp_scripts.models_torch import list_lightweight_runs, train_torch_with_fallback
from shrimp_scripts.progress import log_event
from shrimp_scripts.utils import ensure_dir, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight model comparison runs.")
    parser.add_argument("--output_dir", default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--list_runs", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start index for chunked execution.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of runs for chunked execution.")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = list_lightweight_runs(smoke_test=args.smoke_test)
    rows = list_lightweight_runs(smoke_test=args.smoke_test, start=args.start, limit=args.limit)
    if args.list_runs:
        print(json.dumps({"run_count": len(all_rows), "selected_count": len(rows), "start": args.start, "limit": args.limit, "runs": rows}, indent=2))
        return
    output_dir = ensure_dir(args.output_dir)
    if args.progress:
        log_event("Starting lightweight model training.", output_dir=output_dir, extra={"planned_runs": len(all_rows), "selected_runs": len(rows), "start": args.start, "limit": args.limit})
    split_manifest = load_split_manifest(output_dir)
    save_csv(pd.DataFrame(all_rows), output_dir / "experiment_plan_lightweight_models.csv")
    results = []
    for index, row in enumerate(rows, start=1):
        if args.progress:
            log_event("Starting planned lightweight run.", run_id=row["run_id"], output_dir=output_dir, extra={
                "index": index,
                "selected_total": len(rows),
                "model": row["model"],
                "condition": row["condition_key"],
                "loss": row["loss_key"],
                "randaugment": row["randaugment"],
            })
        condition = {"condition_key": row["condition_key"], "loss_key": row["loss_key"], "randaugment": row["randaugment"]}
        result = train_torch_with_fallback(row["model_key"], row["model"], condition, split_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test, progress_enabled=args.progress)
        results.append(result)
        if args.progress:
            log_event("Finished planned lightweight run.", run_id=row["run_id"], output_dir=output_dir, extra={"status": result.get("status", "completed")})
    if args.progress:
        log_event("Lightweight model script completed.", output_dir=output_dir, extra={"attempted": len(results)})
    print(json.dumps({"completed_or_attempted": len(results), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
