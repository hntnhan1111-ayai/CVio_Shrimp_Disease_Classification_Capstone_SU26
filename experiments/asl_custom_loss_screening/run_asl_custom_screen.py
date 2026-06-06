"""Run ASL custom-loss screening for YOLO and top lightweight Torch models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shrimp_scripts.dataset import load_split_manifest, load_yolo_manifest
from shrimp_scripts import config
from shrimp_scripts.models_torch import make_run_config as make_torch_run_config, train_torch_with_fallback
from shrimp_scripts.models_yolo import make_run_config as make_yolo_run_config, train_yolo_with_fallback, yolo_custom_loss_self_check
from shrimp_scripts.progress import log_event
from shrimp_scripts.utils import ensure_dir, save_csv, stable_hash, validate_run_completion, write_json

from experiments.asl_custom_loss_screening.screening_lib import (
    DEFAULT_OUTPUT_DIR,
    collect_screening_results,
    condition_from_row,
    filter_screening_runs,
    list_screening_runs,
    verify_screening_rows,
    write_plan_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASL custom-loss screening.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--backend", choices=["torch", "yolo", "all"], default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--loss", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--list_runs", action="store_true")
    parser.add_argument("--validate_resume", action="store_true", help="Validate skip/fresh-rerun decisions without training.")
    parser.add_argument("--verify_artifacts", action="store_true")
    parser.add_argument("--self_check_yolo_loss", action="store_true", help="Run a lightweight YOLO custom-loss plumbing check without training.")
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    return parser.parse_args()


def selected_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    all_rows = list_screening_runs(smoke_test=args.smoke_test)
    rows = filter_screening_runs(
        all_rows,
        backend=args.backend,
        model=args.model,
        loss=args.loss,
        run_id=args.run_id,
        start=args.start,
        limit=args.limit,
    )
    return all_rows, rows


def add_progress_indices(rows: list[dict]) -> list[dict]:
    model_order = list(dict.fromkeys(row["model"] for row in rows))
    loss_order = list(dict.fromkeys(row["loss_key"] for row in rows))
    model_total = len(model_order)
    loss_total = len(loss_order)
    indexed = []
    for run_index, row in enumerate(rows, start=1):
        item = dict(row)
        item["run_index"] = run_index
        item["run_total"] = len(rows)
        item["model_index"] = model_order.index(row["model"]) + 1
        item["model_total"] = model_total
        item["loss_index"] = loss_order.index(row["loss_key"]) + 1
        item["loss_total"] = loss_total
        indexed.append(item)
    return indexed


def validate_resume_rows(rows: list[dict], output_dir: Path, smoke_test: bool) -> dict:
    flat_rows = []
    detailed = []
    for row in rows:
        condition = condition_from_row(row)
        validations = []
        if row["screen_backend"] == "yolo":
            for batch in config.YOLO_BATCH_FALLBACKS:
                run_config = make_yolo_run_config(row["model"], condition, output_dir, smoke_test=smoke_test, batch=batch)
                validations.append({"batch": batch, **validate_run_completion(row["run_id"], output_dir, stable_hash(run_config), "ultralytics")})
        else:
            for micro_batch in config.TORCH_MICRO_BATCH_FALLBACKS:
                run_config = make_torch_run_config(row["model_key"], row["model"], condition, output_dir, smoke_test=smoke_test, micro_batch=micro_batch)
                validations.append({"micro_batch": micro_batch, **validate_run_completion(row["run_id"], output_dir, stable_hash(run_config), row["backend"])})
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
            "screen_backend": row["screen_backend"],
            "backend": row["backend"],
            "model": row["model"],
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
    write_json(output_dir / "asl_custom_screening_resume_validation.json", {"summary": summary, "runs": detailed})
    save_csv(pd.DataFrame(flat_rows), output_dir / "asl_custom_screening_resume_validation.csv")
    return {"summary": summary, "runs": flat_rows}


def main() -> None:
    args = parse_args()
    all_rows, rows = selected_rows(args)
    if args.list_runs:
        print(json.dumps({
            "available_before_filter": len(all_rows),
            "selected_count": len(rows),
            "backend": args.backend,
            "model": args.model,
            "loss": args.loss,
            "run_id": args.run_id,
            "start": args.start,
            "limit": args.limit,
            "runs": rows,
        }, indent=2))
        return

    output_dir = ensure_dir(args.output_dir)
    write_plan_files(output_dir, all_rows, rows)
    rows = add_progress_indices(rows)

    if args.validate_resume:
        summary = validate_resume_rows(rows, output_dir, smoke_test=args.smoke_test)
        print(json.dumps({"resume_validation": summary}, indent=2, default=str))
        return

    if args.self_check_yolo_loss:
        loss_keys = list(dict.fromkeys(row["loss_key"] for row in rows if row["screen_backend"] == "yolo"))
        summary = yolo_custom_loss_self_check(loss_keys=loss_keys, output_dir=output_dir)
        print(json.dumps({"yolo_custom_loss_self_check": summary}, indent=2, default=str))
        if summary["status"] != "passed":
            raise SystemExit(1)
        return

    if args.verify_artifacts:
        summary = verify_screening_rows(rows, output_dir, load_checkpoints=not args.skip_checkpoint_load)
        collect_screening_results(output_dir, rows=rows)
        print(json.dumps({"artifact_verification": summary}, indent=2, default=str))
        if summary["failed_count"] > 0:
            raise SystemExit(1)
        return

    if args.progress:
        log_event("Starting ASL custom-loss screening.", output_dir=output_dir, extra={
            "selected_runs": len(rows),
            "available_before_filter": len(all_rows),
            "backend": args.backend,
            "model": args.model,
            "loss": args.loss,
            "run_id": args.run_id,
            "resume": args.resume,
            "smoke_test": args.smoke_test,
        })

    needs_torch = any(row["screen_backend"] == "torch" for row in rows)
    needs_yolo = any(row["screen_backend"] == "yolo" for row in rows)
    split_manifest = load_split_manifest(output_dir) if needs_torch else None
    yolo_manifest = load_yolo_manifest(output_dir) if needs_yolo else None

    results: list[dict] = []
    for row in rows:
        condition = condition_from_row(row)
        if args.progress:
            log_event("Starting ASL screening run.", run_id=row["run_id"], output_dir=output_dir, extra={
                "run_index": row["run_index"],
                "run_total": row["run_total"],
                "model_index": row["model_index"],
                "model_total": row["model_total"],
                "loss_index": row["loss_index"],
                "loss_total": row["loss_total"],
                "backend": row["screen_backend"],
                "model": row["model"],
                "loss": row["loss_key"],
                "randaugment": row["randaugment"],
            })
        if row["screen_backend"] == "yolo":
            assert yolo_manifest is not None
            result = train_yolo_with_fallback(row["model"], condition, yolo_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test, progress_enabled=args.progress)
        else:
            assert split_manifest is not None
            result = train_torch_with_fallback(row["model_key"], row["model"], condition, split_manifest, output_dir, resume=args.resume, smoke_test=args.smoke_test, progress_enabled=args.progress)
        results.append(result)
        if args.progress:
            log_event("Finished ASL screening run.", run_id=row["run_id"], output_dir=output_dir, extra={"status": result.get("status", "completed")})

    write_json(output_dir / "asl_custom_screening_run_results.json", {"results": results})
    save_csv(pd.DataFrame(results), output_dir / "asl_custom_screening_run_results.csv")
    summary, failures, ranked = collect_screening_results(output_dir, rows=rows)
    failed_results = [result for result in results if result.get("status") not in {"completed", "skipped_completed", "skipped_completed_with_final_metrics"}]
    if args.progress:
        log_event("ASL custom-loss screening completed.", output_dir=output_dir, extra={
            "attempted": len(results),
            "completed_rows": len(summary),
            "failed_or_missing_rows": len(failures),
            "ranked_rows": len(ranked),
            "failed_results": len(failed_results),
        })
    print(json.dumps({"completed_or_attempted": len(results), "results": results}, indent=2, default=str))
    if failed_results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
