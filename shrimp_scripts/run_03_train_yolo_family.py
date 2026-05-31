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
from shrimp_scripts.models_yolo import list_yolo_family_runs, make_run_config as make_yolo_run_config, probe_yolo_availability, train_yolo_with_fallback, verify_yolo_run_artifacts
from shrimp_scripts.progress import log_event
from shrimp_scripts.utils import ensure_dir, save_csv, stable_hash, validate_run_completion, write_json, write_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO classification family comparison runs.")
    parser.add_argument("--output_dir", default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--list_runs", action="store_true")
    parser.add_argument("--skip_probe", action="store_true")
    parser.add_argument("--only_model", "--model", dest="only_model", default=None, help="Optional exact model filter for smoke/debug runs, e.g. yolo26m-cls.")
    parser.add_argument("--only_condition", default=None, help="Optional exact condition_key filter, e.g. asl_no_randaugment.")
    parser.add_argument("--only_loss", "--loss", dest="only_loss", default=None, help="Optional exact loss_key filter, e.g. asl_single_label.")
    parser.add_argument("--validate_resume", action="store_true", help="Validate skip/fresh-rerun decisions without training.")
    parser.add_argument("--verify_artifacts", action="store_true", help="Verify completed run artifacts and checkpoint loadability for selected runs without training.")
    parser.add_argument("--skip_checkpoint_load", action="store_true", help="Skip YOLO checkpoint load checks during --verify_artifacts.")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    return parser.parse_args()


def filter_runs(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = rows
    if args.only_model:
        selected = [row for row in selected if row["model"] == args.only_model]
    if args.only_condition:
        selected = [row for row in selected if row["condition_key"] == args.only_condition]
    if args.only_loss:
        selected = [row for row in selected if row["loss_key"] == args.only_loss]
    return selected


def condition_from_row(row: dict) -> dict:
    return {"condition_key": row["condition_key"], "loss_key": row["loss_key"], "randaugment": row["randaugment"]}


def stage03_summary(planned_rows: list[dict], results: list[dict]) -> dict:
    completed = []
    skipped = []
    failed = []
    other = []
    result_by_run_id = {str(result.get("run_id", "")): result for result in results}
    for row in planned_rows:
        result = result_by_run_id.get(row["run_id"], {"run_id": row["run_id"], "status": "not_attempted"})
        status = str(result.get("status", "completed"))
        item = {"run_id": row["run_id"], "model": row["model"], "condition": row["condition_key"], "loss_key": row["loss_key"], "status": status}
        if status == "completed":
            completed.append(item)
        elif status.startswith("skipped"):
            skipped.append(item)
        elif status == "failed":
            item["error"] = result.get("error", result.get("errors", ""))
            failed.append(item)
        else:
            other.append(item)
    return {
        "planned_count": len(planned_rows),
        "attempted_count": len(results),
        "completed_count": len(completed),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "other_count": len(other),
        "completed_runs": completed,
        "skipped_runs": skipped,
        "failed_runs": failed,
        "other_runs": other,
    }


def flatten_verification_row(row: dict) -> dict:
    flattened = {key: value for key, value in row.items() if key not in {"checks", "errors"}}
    for check_name, passed in row.get("checks", {}).items():
        flattened[f"check_{check_name}"] = bool(passed)
    flattened["errors"] = " | ".join(str(error) for error in row.get("errors", []))
    return flattened


def verify_selected_runs(rows: list[dict], output_dir: Path, load_checkpoints: bool, progress_enabled: bool) -> dict:
    verification_rows = []
    for row in rows:
        condition = condition_from_row(row)
        if progress_enabled:
            log_event("Verifying YOLO run artifacts.", run_id=row["run_id"], output_dir=output_dir, extra={"load_checkpoints": load_checkpoints})
        verification = verify_yolo_run_artifacts(row["model"], condition, output_dir, load_checkpoints=load_checkpoints)
        verification_rows.append(verification)
        if progress_enabled:
            level = "INFO" if verification["verification_status"] == "passed" else "ERROR"
            log_event("YOLO run artifact verification finished.", level=level, run_id=row["run_id"], output_dir=output_dir, extra=verification)
    summary = {
        "verified_count": len(verification_rows),
        "passed_count": sum(1 for row in verification_rows if row["verification_status"] == "passed"),
        "failed_count": sum(1 for row in verification_rows if row["verification_status"] != "passed"),
        "load_checkpoints": load_checkpoints,
        "runs": verification_rows,
    }
    write_json(output_dir / "stage03_yolo_artifact_verification.json", summary)
    save_csv(pd.DataFrame(flatten_verification_row(row) for row in verification_rows), output_dir / "stage03_yolo_artifact_verification.csv")
    return summary


def validate_resume_rows(rows: list[dict], output_dir: Path, smoke_test: bool) -> dict:
    flat_rows = []
    detailed = []
    for row in rows:
        condition = condition_from_row(row)
        validations = []
        for batch in config.YOLO_BATCH_FALLBACKS:
            run_config = make_yolo_run_config(row["model"], condition, output_dir, smoke_test=smoke_test, batch=batch)
            validation = validate_run_completion(row["run_id"], output_dir, stable_hash(run_config), "ultralytics")
            validations.append({"batch": batch, **validation})
        valid = next((item for item in validations if item["validation_status"] == "completed_with_valid_final_metrics"), None)
        if valid:
            validation_status = "completed_with_valid_final_metrics"
            resume_decision = "skipped_completed_with_final_metrics"
            selected_batch = valid["batch"]
            errors = []
        else:
            validation_status = "failed" if any(item["validation_status"] == "failed" for item in validations) else "skipped" if any(item["validation_status"] == "skipped" for item in validations) else "incomplete_missing_final_metrics"
            resume_decision = "will_fresh_rerun"
            selected_batch = ""
            errors = sorted({str(error) for item in validations for error in item.get("errors", [])})
        flat_rows.append({
            "run_id": row["run_id"],
            "backend": "ultralytics",
            "model": row["model"],
            "condition": row["condition_key"],
            "loss_key": row["loss_key"],
            "validation_status": validation_status,
            "resume_decision": resume_decision,
            "valid_batch": selected_batch,
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
    write_json(output_dir / "stage03_resume_validation.json", {"summary": summary, "runs": detailed})
    save_csv(pd.DataFrame(flat_rows), output_dir / "stage03_resume_validation.csv")
    return {"summary": summary, "runs": flat_rows}


def main() -> None:
    args = parse_args()
    all_rows = list_yolo_family_runs(smoke_test=args.smoke_test)
    rows = filter_runs(all_rows, args)
    if args.list_runs:
        print(json.dumps({"run_count": len(rows), "available_before_filter": len(all_rows), "runs": rows}, indent=2))
        return
    output_dir = ensure_dir(args.output_dir)
    if args.validate_resume:
        summary = validate_resume_rows(rows, output_dir, smoke_test=args.smoke_test)
        print(json.dumps({"resume_validation": summary}, indent=2, default=str))
        return
    if args.verify_artifacts:
        summary = verify_selected_runs(rows, output_dir, load_checkpoints=not args.skip_checkpoint_load, progress_enabled=args.progress)
        print(json.dumps({"artifact_verification": summary}, indent=2, default=str))
        if summary["failed_count"] > 0:
            raise SystemExit(1)
        return
    if args.progress:
        log_event("Starting YOLO family training.", output_dir=output_dir, extra={"planned_runs": len(rows), "available_before_filter": len(all_rows), "skip_probe": args.skip_probe, "only_model": args.only_model, "only_condition": args.only_condition, "only_loss": args.only_loss})
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
        condition = condition_from_row(row)
        result = train_yolo_with_fallback(row["model"], condition, yolo_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test, progress_enabled=args.progress)
        results.append(result)
        if args.progress:
            log_event("Finished planned YOLO family run.", run_id=row["run_id"], output_dir=output_dir, extra={"status": result.get("status", "completed")})
    summary = stage03_summary(rows, results)
    write_json(output_dir / "stage03_yolo_family_summary.json", summary)
    save_csv(pd.DataFrame(summary["completed_runs"] + summary["skipped_runs"] + summary["failed_runs"] + summary["other_runs"]), output_dir / "stage03_yolo_family_summary.csv")
    if summary["failed_count"] > 0 and args.progress:
        log_event("Stage 03 completed with failed YOLO runs.", level="ERROR", output_dir=output_dir, extra={"failed_count": summary["failed_count"], "failed_runs": summary["failed_runs"]})
    if args.progress:
        log_event("YOLO family script completed.", output_dir=output_dir, extra=summary)
    print(json.dumps({"completed_or_attempted": len(results), "summary": summary, "results": results}, indent=2, default=str))
    if summary["failed_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
