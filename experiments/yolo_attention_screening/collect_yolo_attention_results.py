"""Collect and rank YOLOv26m-cls attention screening results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shrimp_scripts import config
from shrimp_scripts.utils import read_json, read_json_safely, save_csv, validate_run_completion, stable_hash, write_json
from shrimp_scripts.models_yolo import make_run_config as make_yolo_run_config

from experiments.yolo_attention_screening.screening_lib import (
    ATTENTION_RANKING_COLUMNS,
    DEFAULT_OUTPUT_DIR,
    YOLO_ATTENTION_SCREEN_EPOCHS,
    attention_audit_valid,
    condition_for_attention,
    list_attention_runs,
    write_attention_plan,
)


RESULT_COLUMNS = [
    "run_id",
    "model",
    "attention_key",
    "condition_key",
    "loss_key",
    "randaugment",
    "epochs",
    "status",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "cohen_kappa",
    "val_macro_f1",
    "latency_ms_image",
    "fps",
    "params_m",
    "model_size_mb",
    "checkpoint_path",
    "class_order_audit_path",
    "attention_module_audit_path",
    "attention_status",
    "attention_inserted",
    "implementation_source",
    "implementation_variant",
    "params_added",
    "is_exact_official_implementation",
]
FAILED_COLUMNS = ["run_id", "model", "attention_key", "condition_key", "status", "errors"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect YOLOv26m-cls attention screening metrics and rankings.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=YOLO_ATTENTION_SCREEN_EPOCHS)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def completed_row(row: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    run_dir = output_dir / "runs" / row["run_id"]
    errors: list[str] = []
    valid_validation = None
    for batch in config.YOLO_BATCH_FALLBACKS:
        condition = condition_for_attention(row["attention_key"], epochs=int(row["epochs"]))
        run_config = make_yolo_run_config(row["model"], condition, output_dir, smoke_test=False, batch=batch)
        validation = validate_run_completion(row["run_id"], output_dir, stable_hash(run_config), "ultralytics")
        if validation["validation_status"] == "completed_with_valid_final_metrics":
            valid_validation = validation
            break
        errors.extend(str(error) for error in validation.get("errors", []))
    audit_ok, audit_errors, attention_audit = attention_audit_valid(run_dir, row["attention_key"])
    if not audit_ok:
        errors.extend(f"attention_audit:{error}" for error in audit_errors)
    metrics = read_json(run_dir / "metrics.json", default={}) if (run_dir / "metrics.json").is_file() else {}
    explicit_errors: list[str] = []
    for metric_name in ["test_macro_f1", "cohen_kappa", "test_accuracy"]:
        if not finite(metrics.get(metric_name)):
            explicit_errors.append(f"invalid_or_missing_metric:{metric_name}")
    class_order_audit, class_order_error = read_json_safely(run_dir / "class_order_audit.json")
    if class_order_error:
        explicit_errors.append(f"class_order_audit:{class_order_error}")
    else:
        if class_order_audit.get("audit_passed") is not True:
            explicit_errors.append("class_order_audit:audit_passed_not_true")
        swap = class_order_audit.get("swap_diagnostic", {})
        if class_order_audit.get("failed_due_to_swap_diagnostic") is True:
            explicit_errors.append("class_order_audit:swap_diagnostic_failed")
        if isinstance(swap, dict) and swap.get("failed_due_to_swap_diagnostic") is True:
            explicit_errors.append("class_order_audit:swap_diagnostic_failed")
    if not audit_ok:
        explicit_errors.extend(f"attention_audit:{error}" for error in audit_errors)
    errors.extend(explicit_errors)
    if valid_validation and audit_ok and not explicit_errors:
        out = {
            "run_id": row["run_id"],
            "model": row["model"],
            "attention_key": row["attention_key"],
            "condition_key": row["condition_key"],
            "loss_key": row["loss_key"],
            "randaugment": bool(row["randaugment"]),
            "epochs": int(row["epochs"]),
            "status": "completed",
            "test_accuracy": metrics.get("test_accuracy"),
            "test_macro_precision": metrics.get("test_macro_precision"),
            "test_macro_recall": metrics.get("test_macro_recall"),
            "test_macro_f1": metrics.get("test_macro_f1"),
            "cohen_kappa": metrics.get("cohen_kappa"),
            "val_macro_f1": metrics.get("val_macro_f1"),
            "latency_ms_image": metrics.get("test", {}).get("latency_ms_image") if isinstance(metrics.get("test"), dict) else None,
            "fps": metrics.get("test", {}).get("fps") if isinstance(metrics.get("test"), dict) else None,
            "params_m": metrics.get("params_m"),
            "model_size_mb": metrics.get("model_size_mb"),
            "checkpoint_path": metrics.get("checkpoint_path", ""),
            "class_order_audit_path": metrics.get("class_order_audit_path", ""),
            "attention_module_audit_path": metrics.get("attention_module_audit_path", str((run_dir / "attention_module_audit.json").resolve())),
            "attention_status": attention_audit.get("status", ""),
            "attention_inserted": bool(attention_audit.get("inserted", False)),
            "implementation_source": attention_audit.get("implementation_source", ""),
            "implementation_variant": attention_audit.get("implementation_variant", ""),
            "params_added": attention_audit.get("params_added"),
            "is_exact_official_implementation": bool(attention_audit.get("is_exact_official_implementation", False)),
        }
        return out, None
    status = read_json(run_dir / "status.json", default={}).get("status", "missing") if run_dir.exists() else "missing"
    failed = {
        "run_id": row["run_id"],
        "model": row["model"],
        "attention_key": row["attention_key"],
        "condition_key": row["condition_key"],
        "status": status,
        "errors": " | ".join(sorted(set(errors))) or "missing_or_invalid_run",
    }
    return None, failed


def rank_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["rank", *RESULT_COLUMNS])
    ranked = results.copy()
    for column, ascending in ATTENTION_RANKING_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = float("nan")
    ranked = ranked.sort_values([column for column, _ascending in ATTENTION_RANKING_COLUMNS], ascending=[ascending for _column, ascending in ATTENTION_RANKING_COLUMNS], na_position="last").reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return ""
    selected = frame[columns] if columns else frame
    headers = list(selected.columns)
    rows = []
    for item in selected.fillna("").astype(str).to_dict(orient="records"):
        rows.append([item.get(header, "") for header in headers])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _header in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, planned: list[dict[str, Any]], results: pd.DataFrame, ranking: pd.DataFrame, failed: pd.DataFrame) -> Path:
    lines = [
        "# YOLOv26m-cls Attention Screening Reproducibility Report",
        "",
        "Screening design: YOLOv26m-cls, cross-entropy only, RandAugment disabled, fixed paper class order and dataset split.",
        "",
        f"Planned variants: {len(planned)}",
        f"Completed valid variants: {len(results)}",
        f"Failed/skipped/missing variants: {len(failed)}",
        "",
        "Ranking rule: sort by test_macro_f1 descending, then cohen_kappa descending, test_accuracy descending, latency_ms_image ascending, fps descending, model_size_mb ascending, params_m ascending.",
        "",
    ]
    if not ranking.empty:
        keep = ["rank", "attention_key", "test_macro_f1", "cohen_kappa", "test_accuracy", "latency_ms_image", "fps", "model_size_mb", "params_m"]
        lines.extend(["## Ranking", "", markdown_table(ranking, keep), ""])
    if not failed.empty:
        lines.extend(["## Failed Or Skipped", "", markdown_table(failed), ""])
    report_path = output_dir / "attention_reproducibility_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    planned = list_attention_runs(epochs=args.epochs)
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "planned_count": len(planned), "runs": planned}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_attention_plan(output_dir, planned)
    result_rows = []
    failed_rows = []
    for row in planned:
        completed, failed = completed_row(row, output_dir)
        if completed:
            result_rows.append(completed)
        if failed:
            failed_rows.append(failed)
    results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    failed = pd.DataFrame(failed_rows, columns=FAILED_COLUMNS)
    ranking = rank_results(results)
    save_csv(results, output_dir / "attention_screening_results.csv")
    save_csv(ranking, output_dir / "attention_ranking.csv")
    save_csv(failed, output_dir / "attention_failed_or_skipped.csv")
    write_json(output_dir / "attention_screening_collection_summary.json", {
        "planned_count": len(planned),
        "completed_count": len(results),
        "failed_or_skipped_count": len(failed),
        "baseline_completed": bool((not results.empty) and results["attention_key"].eq("none_baseline").any()),
        "ranking_empty": bool(ranking.empty),
        "ranking_empty_reason": "no_valid_completed_runs" if ranking.empty else "",
        "ranking_rule": [f"{column}:{'asc' if ascending else 'desc'}" for column, ascending in ATTENTION_RANKING_COLUMNS],
    })
    report_path = write_report(output_dir, planned, results, ranking, failed)
    print(json.dumps({"results": len(results), "failed_or_skipped": len(failed), "report": str(report_path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
