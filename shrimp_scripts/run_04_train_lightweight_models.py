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
from shrimp_scripts.models_torch import list_lightweight_diagnostic_runs, list_lightweight_runs, make_run_config as make_torch_run_config, train_torch_with_fallback
from shrimp_scripts.progress import log_event
from shrimp_scripts.utils import ensure_dir, save_csv, stable_hash, validate_run_completion, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight model comparison runs.")
    parser.add_argument("--output_dir", default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--list_runs", action="store_true")
    parser.add_argument("--validate_resume", action="store_true", help="Validate skip/fresh-rerun decisions without training.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start index for chunked execution.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of runs for chunked execution.")
    parser.add_argument("--run_id", default=None, help="Run exactly one default or diagnostic lightweight run by ID.")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    return parser.parse_args()


def select_runs(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], list[dict], str]:
    default_rows = list_lightweight_runs(smoke_test=args.smoke_test)
    diagnostic_rows = list_lightweight_diagnostic_runs(smoke_test=args.smoke_test)
    if args.run_id:
        candidates = default_rows + diagnostic_rows
        rows = [row for row in candidates if row["run_id"] == args.run_id]
        if not rows:
            known = [row["run_id"] for row in candidates]
            raise SystemExit(f"Unknown --run_id {args.run_id!r}. Known run IDs: {known}")
        return default_rows, diagnostic_rows, rows, rows, "single_run_id"
    selected_rows = list_lightweight_runs(smoke_test=args.smoke_test, start=args.start, limit=args.limit)
    return default_rows, diagnostic_rows, selected_rows, selected_rows, "default_plan"


def condition_from_row(row: dict) -> dict:
    condition = {
        "condition_key": row["condition_key"],
        "loss_key": row["loss_key"],
        "randaugment": row["randaugment"],
    }
    if row.get("diagnostic_extra"):
        condition["diagnostic_extra"] = True
        condition["experiment_key"] = row.get("experiment_key", "lightweight_diagnostic")
        condition["experiment_group"] = row.get("experiment_group", "lightweight_diagnostic")
    return condition


def validate_resume_rows(rows: list[dict], output_dir: Path, smoke_test: bool) -> dict:
    flat_rows = []
    detailed = []
    for row in rows:
        condition = condition_from_row(row)
        validations = []
        for micro_batch in config.TORCH_MICRO_BATCH_FALLBACKS:
            run_config = make_torch_run_config(row["model_key"], row["model"], condition, output_dir, smoke_test=smoke_test, micro_batch=micro_batch)
            validations.append({"micro_batch": micro_batch, **validate_run_completion(row["run_id"], output_dir, stable_hash(run_config), row["backend"])})
        valid = next((item for item in validations if item["validation_status"] == "completed_with_valid_final_metrics"), None)
        if valid:
            validation_status = "completed_with_valid_final_metrics"
            resume_decision = "skipped_completed_with_final_metrics"
            selected_micro_batch = valid["micro_batch"]
            errors = []
        else:
            validation_status = "failed" if any(item["validation_status"] == "failed" for item in validations) else "skipped" if any(item["validation_status"] == "skipped" for item in validations) else "incomplete_missing_final_metrics"
            resume_decision = "will_fresh_rerun"
            selected_micro_batch = ""
            errors = sorted({str(error) for item in validations for error in item.get("errors", [])})
        flat_rows.append({
            "run_id": row["run_id"],
            "backend": row["backend"],
            "model": row["model"],
            "condition": row["condition_key"],
            "loss_key": row["loss_key"],
            "validation_status": validation_status,
            "resume_decision": resume_decision,
            "valid_micro_batch": selected_micro_batch,
            "errors": " | ".join(errors),
        })
        detailed.append({"run_id": row["run_id"], "validations": validations})
    summary = {
        "planned_count": len(rows),
        "completed_with_valid_final_metrics": sum(1 for row in flat_rows if row["validation_status"] == "completed_with_valid_final_metrics"),
        "incomplete_missing_final_metrics": sum(1 for row in flat_rows if row["validation_status"] == "incomplete_missing_final_metrics"),
        "failed": sum(1 for row in flat_rows if row["validation_status"] == "failed"),
        "skipped": sum(1 for row in flat_rows if row["validation_status"] == "skipped"),
        "will_fresh_rerun": sum(1 for row in flat_rows if row["resume_decision"] == "will_fresh_rerun"),
        "skipped_completed_with_final_metrics": sum(1 for row in flat_rows if row["resume_decision"] == "skipped_completed_with_final_metrics"),
    }
    write_json(output_dir / "stage04_resume_validation.json", {"summary": summary, "runs": detailed})
    save_csv(pd.DataFrame(flat_rows), output_dir / "stage04_resume_validation.csv")
    return {"summary": summary, "runs": flat_rows}


def main() -> None:
    args = parse_args()
    default_rows, diagnostic_rows, rows, selected_plan_rows, selection_mode = select_runs(args)
    if args.list_runs:
        print(json.dumps({
            "default_run_count": len(default_rows),
            "diagnostic_run_count": len(diagnostic_rows),
            "selected_count": len(rows),
            "selection_mode": selection_mode,
            "start": args.start,
            "limit": args.limit,
            "run_id": args.run_id,
            "runs": rows,
            "diagnostic_runs_available": diagnostic_rows,
        }, indent=2))
        return
    output_dir = ensure_dir(args.output_dir)
    if args.validate_resume:
        summary = validate_resume_rows(rows, output_dir, smoke_test=args.smoke_test)
        print(json.dumps({"resume_validation": summary}, indent=2, default=str))
        return
    if args.progress:
        log_event("Starting lightweight model training.", output_dir=output_dir, extra={
            "default_planned_runs": len(default_rows),
            "diagnostic_runs_available": len(diagnostic_rows),
            "selected_runs": len(rows),
            "selection_mode": selection_mode,
            "start": args.start,
            "limit": args.limit,
            "run_id": args.run_id,
        })
    split_manifest = load_split_manifest(output_dir)
    save_csv(pd.DataFrame(default_rows), output_dir / "experiment_plan_lightweight_models.csv")
    save_csv(pd.DataFrame(selected_plan_rows), output_dir / "experiment_plan_lightweight_selected_runs.csv")
    if diagnostic_rows:
        save_csv(pd.DataFrame(diagnostic_rows), output_dir / "experiment_plan_lightweight_diagnostic_runs.csv")
    results = []
    for index, row in enumerate(rows, start=1):
        if args.progress:
            log_event("Starting planned lightweight run.", run_id=row["run_id"], output_dir=output_dir, extra={
                "index": index,
                "selected_total": len(rows),
                "selection_mode": selection_mode,
                "model": row["model"],
                "condition": row["condition_key"],
                "loss": row["loss_key"],
                "randaugment": row["randaugment"],
                "diagnostic_extra": bool(row.get("diagnostic_extra", False)),
                "experiment_group": row.get("experiment_group", "lightweight_default"),
            })
        condition = condition_from_row(row)
        result = train_torch_with_fallback(row["model_key"], row["model"], condition, split_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test, progress_enabled=args.progress)
        results.append(result)
        if args.progress:
            log_event("Finished planned lightweight run.", run_id=row["run_id"], output_dir=output_dir, extra={"status": result.get("status", "completed")})
    if args.progress:
        log_event("Lightweight model script completed.", output_dir=output_dir, extra={"attempted": len(results)})
    print(json.dumps({"completed_or_attempted": len(results), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
