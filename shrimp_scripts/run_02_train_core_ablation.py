"""Run the main ConvNeXt/ShrimpXNet and YOLOv26m-cls core comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shrimp_scripts import config
from shrimp_scripts.dataset import load_split_manifest, load_yolo_manifest
from shrimp_scripts.models_torch import list_core_torch_runs, make_run_config as make_torch_run_config, train_torch_with_fallback
from shrimp_scripts.models_yolo import list_core_yolo_runs, make_run_config as make_yolo_run_config, train_yolo_with_fallback
from shrimp_scripts.progress import log_event
from shrimp_scripts.utils import ensure_dir, save_csv, stable_hash, validate_run_completion, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the main paper core comparison.")
    parser.add_argument("--output_dir", default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--list_runs", action="store_true")
    parser.add_argument("--validate_resume", action="store_true", help="Validate skip/fresh-rerun decisions without training.")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    return parser.parse_args()


def planned_runs(smoke_test: bool = False) -> list[dict]:
    return list_core_torch_runs(smoke_test=smoke_test) + list_core_yolo_runs(smoke_test=smoke_test)


def condition_from_row(row: dict) -> dict:
    return {"condition_key": row["condition_key"], "loss_key": row["loss_key"], "randaugment": row["randaugment"]}


def validate_resume_rows(rows: list[dict], output_dir: Path, smoke_test: bool) -> dict:
    flat_rows = []
    detailed = []
    for row in rows:
        condition = condition_from_row(row)
        validations = []
        if row["backend"] == "ultralytics":
            for batch in config.YOLO_BATCH_FALLBACKS:
                run_config = make_yolo_run_config(row["model"], condition, output_dir, smoke_test=smoke_test, batch=batch)
                validations.append({"batch": batch, **validate_run_completion(row["run_id"], output_dir, stable_hash(run_config), "ultralytics")})
        else:
            backend = row["backend"]
            for micro_batch in config.TORCH_MICRO_BATCH_FALLBACKS:
                run_config = make_torch_run_config(row["model_key"], row["model"], condition, output_dir, smoke_test=smoke_test, micro_batch=micro_batch)
                validations.append({"micro_batch": micro_batch, **validate_run_completion(row["run_id"], output_dir, stable_hash(run_config), backend)})
        valid = next((item for item in validations if item["validation_status"] == "completed_with_valid_final_metrics"), None)
        if valid:
            validation_status = "completed_with_valid_final_metrics"
            resume_decision = "skipped_completed_with_final_metrics"
            errors = []
        else:
            validation_status = "failed" if any(item["validation_status"] == "failed" for item in validations) else "skipped" if any(item["validation_status"] == "skipped" for item in validations) else "incomplete_missing_final_metrics"
            resume_decision = "will_fresh_rerun"
            errors = sorted({str(error) for item in validations for error in item.get("errors", [])})
        flat_rows.append({
            "run_id": row["run_id"],
            "backend": row["backend"],
            "model": row["model"],
            "condition": row["condition_key"],
            "loss_key": row["loss_key"],
            "validation_status": validation_status,
            "resume_decision": resume_decision,
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
    write_json(output_dir / "stage02_resume_validation.json", {"summary": summary, "runs": detailed})
    save_csv(__import__("pandas").DataFrame(flat_rows), output_dir / "stage02_resume_validation.csv")
    return {"summary": summary, "runs": flat_rows}


def main() -> None:
    args = parse_args()
    rows = planned_runs(smoke_test=args.smoke_test)
    if args.list_runs:
        print(json.dumps({"run_count": len(rows), "runs": rows}, indent=2))
        return
    output_dir = ensure_dir(args.output_dir)
    if args.validate_resume:
        summary = validate_resume_rows(rows, output_dir, smoke_test=args.smoke_test)
        print(json.dumps({"resume_validation": summary}, indent=2, default=str))
        return
    if args.progress:
        log_event("Starting core ablation training.", output_dir=output_dir, extra={"planned_runs": len(rows), "smoke_test": args.smoke_test, "resume": args.resume})
    split_manifest = load_split_manifest(output_dir)
    yolo_manifest = load_yolo_manifest(output_dir)
    save_csv(__import__("pandas").DataFrame(rows), output_dir / "experiment_plan_core_ablation.csv")
    results = []
    for index, row in enumerate(rows, start=1):
        condition = {"condition_key": row["condition_key"], "loss_key": row["loss_key"], "randaugment": row["randaugment"]}
        if args.progress:
            log_event("Starting planned core run.", run_id=row["run_id"], output_dir=output_dir, extra={
                "index": index,
                "total": len(rows),
                "backend": row["backend"],
                "model": row["model"],
                "condition": row["condition_key"],
                "loss": row["loss_key"],
                "randaugment": row["randaugment"],
            })
        if row["backend"] == "ultralytics":
            result = train_yolo_with_fallback(row["model"], condition, yolo_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test, progress_enabled=args.progress)
        else:
            result = train_torch_with_fallback(row["model_key"], row["model"], condition, split_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test, progress_enabled=args.progress)
        results.append(result)
        if args.progress:
            log_event("Finished planned core run.", run_id=row["run_id"], output_dir=output_dir, extra={"status": result.get("status", "completed")})
    if args.progress:
        log_event("Core ablation script completed.", output_dir=output_dir, extra={"attempted": len(results)})
    print(json.dumps({"completed_or_attempted": len(results), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
