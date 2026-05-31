"""Shared helpers for the ASL custom-loss screening experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from shrimp_scripts import config
from shrimp_scripts.losses import LOSS_CONFIG
from shrimp_scripts.models_torch import torch_run_id
from shrimp_scripts.models_yolo import verify_yolo_run_artifacts, yolo_run_id
from shrimp_scripts.utils import ensure_dir, read_json, required_outputs_exist, save_csv, stable_hash, write_json


EXPERIMENT_KEY = "asl_custom_screening"
EXPERIMENT_GROUP = "asl_custom_loss_screening"
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/shrimp_outputs_asl_custom_screening")
TORCH_SCREENING_MODELS = ["convnext_tiny_in22k", "mobilenet_v3_large", "efficientnet_b0"]
BASELINE_LOSS_KEYS = ["baseline_ce", "asl_single_label"]
CUSTOM_ASL_LOSS_KEYS = [
    "class_weighted_asl",
    "coinfection_weighted_asl",
    "boundary_weighted_asl",
    "confusion_aware_negative_asl",
    "soft_target_coinfection_asl",
    "attribute_projection_asl",
    "adaptive_gamma_asl",
    "asl_ldam_margin",
    "dangerous_confidence_penalty_asl",
    "coinfection_logit_adjusted_asl",
]
SCREENING_LOSS_KEYS = BASELINE_LOSS_KEYS + CUSTOM_ASL_LOSS_KEYS


def torch_backend_for_model(model_key: str) -> str:
    return "torchvision" if model_key in {"mobilenet_v3_large", "shufflenet_v2_x1_0", "squeezenet1_1"} else "timm"


def loss_role(loss_key: str) -> str:
    return "baseline" if loss_key in BASELINE_LOSS_KEYS else "custom"


def condition_for_loss(loss_key: str, randaugment: bool) -> dict[str, Any]:
    suffix = "randaugment" if randaugment else "no_randaugment"
    return {
        "condition_key": f"{loss_key}_{suffix}",
        "loss_key": loss_key,
        "randaugment": bool(randaugment),
        "experiment_key": EXPERIMENT_KEY,
        "experiment_group": EXPERIMENT_GROUP,
        "loss_role": loss_role(loss_key),
    }


def condition_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return condition_for_loss(row["loss_key"], bool(row["randaugment"]))


def list_screening_runs(smoke_test: bool = False) -> list[dict[str, Any]]:
    torch_models = TORCH_SCREENING_MODELS[:1] if smoke_test else TORCH_SCREENING_MODELS
    yolo_models = [config.YOLO_CORE_MODEL] if smoke_test else config.YOLO_FAMILY_MODELS
    rows: list[dict[str, Any]] = []
    for model_key in torch_models:
        for loss_key in SCREENING_LOSS_KEYS:
            condition = condition_for_loss(loss_key, randaugment=False)
            rows.append({
                "run_id": torch_run_id(model_key, condition),
                "screen_backend": "torch",
                "backend": torch_backend_for_model(model_key),
                "model": model_key,
                "model_key": model_key,
                "loss_key": loss_key,
                "loss": LOSS_CONFIG[loss_key]["name"],
                "loss_role": loss_role(loss_key),
                **condition,
            })
    for model_name in yolo_models:
        for loss_key in SCREENING_LOSS_KEYS:
            condition = condition_for_loss(loss_key, randaugment=True)
            rows.append({
                "run_id": yolo_run_id(model_name, condition),
                "screen_backend": "yolo",
                "backend": "ultralytics",
                "model": model_name,
                "model_key": model_name.replace("-", "_"),
                "loss_key": loss_key,
                "loss": LOSS_CONFIG[loss_key]["name"],
                "loss_role": loss_role(loss_key),
                **condition,
            })
    return rows


def filter_screening_runs(
    rows: list[dict[str, Any]],
    *,
    backend: str = "all",
    model: str | None = None,
    loss: str | None = None,
    run_id: str | None = None,
    start: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = rows
    if backend != "all":
        selected = [row for row in selected if row["screen_backend"] == backend]
    if model:
        selected = [row for row in selected if row["model"] == model or row["model_key"] == model]
    if loss:
        selected = [row for row in selected if row["loss_key"] == loss]
    if run_id:
        selected = [row for row in selected if row["run_id"] == run_id]
    selected = selected[start:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def write_plan_files(output_dir: str | Path, all_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> None:
    output_dir = ensure_dir(output_dir)
    save_csv(pd.DataFrame(all_rows), output_dir / "asl_custom_screening_plan_full.csv")
    save_csv(pd.DataFrame(selected_rows), output_dir / "asl_custom_screening_plan_selected.csv")


def _truthy_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def verify_torch_run_artifacts(row: dict[str, Any], output_dir: str | Path, load_checkpoints: bool = True) -> dict[str, Any]:
    output_dir = Path(output_dir)
    run_dir = output_dir / "runs" / row["run_id"]
    result: dict[str, Any] = {
        "run_id": row["run_id"],
        "screen_backend": row["screen_backend"],
        "backend": row["backend"],
        "model": row["model"],
        "loss_key": row["loss_key"],
        "randaugment": bool(row["randaugment"]),
        "verification_status": "failed",
        "checks": {},
        "errors": [],
    }
    if not run_dir.exists():
        result["errors"].append("missing_run_directory")
        return result
    status = read_json(run_dir / "status.json", default={})
    audit = read_json(run_dir / "run_audit.json", default={})
    metrics = read_json(run_dir / "metrics.json", default={})
    run_config = read_json(run_dir / "run_config.json", default={})
    checks = result["checks"]
    checks["status_completed"] = status.get("status") == "completed"
    checks["audit_completed"] = audit.get("status") == "completed"
    checks["metrics_completed"] = metrics.get("status") == "completed"
    checks["run_config_exists"] = _truthy_file(run_dir / "run_config.json")
    checks["resolved_model_config_exists"] = _truthy_file(run_dir / "resolved_model_config.json")
    checks["resolved_transform_config_exists"] = _truthy_file(run_dir / "resolved_transform_config.json")
    checks["training_history_exists"] = _truthy_file(run_dir / "training_history.csv")
    checks["val_predictions_exists"] = _truthy_file(run_dir / "val_predictions.csv")
    checks["test_predictions_exists"] = _truthy_file(run_dir / "test_predictions.csv")
    checks["classification_report_exists"] = _truthy_file(run_dir / "classification_report.csv")
    checks["confusion_matrix_exists"] = _truthy_file(run_dir / "confusion_matrix.csv")
    checks["confusion_counts_exists"] = _truthy_file(run_dir / "confusion_counts.csv")
    checks["torch_randaugment_disabled"] = read_json(run_dir / "resolved_transform_config.json", default={}).get("randaugment") is False
    outputs_ok, missing_outputs = required_outputs_exist(run_dir, row["backend"])
    checks["required_outputs_exist"] = outputs_ok
    result["missing_outputs"] = missing_outputs
    if run_config and audit.get("config_hash"):
        checks["config_hash_matches"] = audit.get("config_hash") == stable_hash(run_config)
    else:
        checks["config_hash_matches"] = False
    checkpoint_path = run_dir / "checkpoint_best.pt"
    result["checkpoint_path"] = str(checkpoint_path)
    checks["checkpoint_best_exists"] = _truthy_file(checkpoint_path)
    if load_checkpoints and checks["checkpoint_best_exists"]:
        try:
            import torch
            torch.load(checkpoint_path, map_location="cpu")
            checks["checkpoint_loadable"] = True
        except Exception as exc:
            checks["checkpoint_loadable"] = False
            result["errors"].append(f"checkpoint_load_failed:{repr(exc)}")
    elif load_checkpoints:
        checks["checkpoint_loadable"] = False
    for check_name, passed in checks.items():
        if not passed:
            result["errors"].append(check_name)
    if not missing_outputs and all(bool(value) for value in checks.values()):
        result["verification_status"] = "passed"
    return result


def verify_screening_rows(rows: list[dict[str, Any]], output_dir: str | Path, load_checkpoints: bool = True) -> dict[str, Any]:
    verification_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["screen_backend"] == "yolo":
            verification = verify_yolo_run_artifacts(row["model"], condition_from_row(row), output_dir, load_checkpoints=load_checkpoints)
            verification["screen_backend"] = "yolo"
            verification["backend"] = "ultralytics"
        else:
            verification = verify_torch_run_artifacts(row, output_dir, load_checkpoints=load_checkpoints)
        verification_rows.append(verification)
    summary = {
        "verified_count": len(verification_rows),
        "passed_count": sum(1 for item in verification_rows if item["verification_status"] == "passed"),
        "failed_count": sum(1 for item in verification_rows if item["verification_status"] != "passed"),
        "load_checkpoints": load_checkpoints,
        "runs": verification_rows,
    }
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "asl_custom_screening_artifact_verification.json", summary)
    flat_rows = []
    for item in verification_rows:
        flat = {key: value for key, value in item.items() if key not in {"checks", "errors"}}
        for check_name, passed in item.get("checks", {}).items():
            flat[f"check_{check_name}"] = bool(passed)
        flat["errors"] = " | ".join(str(error) for error in item.get("errors", []))
        flat_rows.append(flat)
    save_csv(pd.DataFrame(flat_rows), output_dir / "asl_custom_screening_artifact_verification.csv")
    return summary


def _metric_record_from_file(row: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics.get("test", {})
    val = metrics.get("val", {})
    record = {
        "run_id": row["run_id"],
        "status": "completed",
        "screen_backend": row["screen_backend"],
        "backend": metrics.get("backend", row["backend"]),
        "model": row["model"],
        "model_key": row["model_key"],
        "loss_key": row["loss_key"],
        "loss": metrics.get("loss", row["loss"]),
        "loss_role": row["loss_role"],
        "condition": metrics.get("condition", row["condition_key"]),
        "randaugment": bool(metrics.get("randaugment", row["randaugment"])),
        "seed": metrics.get("seed"),
        "repeat": metrics.get("repeat"),
        "split_seed": metrics.get("split_seed"),
        "val_macro_f1": val.get("macro_f1"),
        "test_accuracy": test.get("accuracy"),
        "test_macro_precision": test.get("macro_precision"),
        "test_macro_recall": test.get("macro_recall"),
        "test_macro_f1": test.get("macro_f1"),
        "cohen_kappa": test.get("cohen_kappa"),
        "BG->WSSV_BG": test.get("BG->WSSV_BG"),
        "WSSV->WSSV_BG": test.get("WSSV->WSSV_BG"),
        "WSSV_BG->BG": test.get("WSSV_BG->BG"),
        "WSSV_BG->WSSV": test.get("WSSV_BG->WSSV"),
        "training_time_s": metrics.get("training_time_s"),
        "latency_ms_image": test.get("latency_ms_image"),
        "fps": test.get("fps"),
        "params_m": metrics.get("params_m"),
        "model_size_mb": metrics.get("model_size_mb"),
        "checkpoint_path": metrics.get("checkpoint_path"),
        "checkpoint_sha256": metrics.get("checkpoint_sha256"),
    }
    record["coinfection_boundary_errors"] = sum(int(record.get(col) or 0) for col in ["BG->WSSV_BG", "WSSV->WSSV_BG", "WSSV_BG->BG", "WSSV_BG->WSSV"])
    per_class = test.get("per_class", {})
    for class_name, stats in per_class.items():
        for metric_name, value in stats.items():
            record[f"{class_name}_{metric_name}"] = value
    return record


def collect_screening_results(output_dir: str | Path, rows: list[dict[str, Any]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    rows = rows or list_screening_runs(smoke_test=False)
    completed_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(output_dir) / "runs" / row["run_id"]
        status = read_json(run_dir / "status.json", default={}) if run_dir.exists() else {"status": "missing_run_directory"}
        metrics_path = run_dir / "metrics.json"
        if status.get("status") == "completed" and metrics_path.exists():
            completed_records.append(_metric_record_from_file(row, read_json(metrics_path)))
        else:
            failure_records.append({
                "run_id": row["run_id"],
                "screen_backend": row["screen_backend"],
                "backend": row["backend"],
                "model": row["model"],
                "loss_key": row["loss_key"],
                "loss_role": row["loss_role"],
                "randaugment": bool(row["randaugment"]),
                "status": status.get("status", "missing_status"),
                "error": status.get("error", status.get("errors", "")),
                "exception_type": status.get("exception_type", ""),
                "exception_message": status.get("exception_message", ""),
            })
    summary = pd.DataFrame(completed_records)
    failures = pd.DataFrame(failure_records)
    ranked = rank_screening_results(summary)
    save_csv(summary, Path(output_dir) / "asl_custom_screening_summary.csv")
    save_csv(failures, Path(output_dir) / "asl_custom_screening_failures.csv")
    save_csv(ranked, Path(output_dir) / "asl_custom_screening_ranked.csv")
    write_json(Path(output_dir) / "asl_custom_screening_summary.json", {
        "planned_count": len(rows),
        "completed_count": len(summary),
        "failed_or_missing_count": len(failures),
        "ranked_count": len(ranked),
        "output_dir": str(Path(output_dir)),
    })
    return summary, failures, ranked


def rank_screening_results(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    ranked = summary.copy()
    for col in ["test_macro_f1", "cohen_kappa", "coinfection_boundary_errors", "latency_ms_image", "model_size_mb"]:
        if col not in ranked.columns:
            ranked[col] = None
    ranked["beats_ce_and_asl_same_model"] = False
    ranked["clear_coinfection_error_benefit"] = False
    ranked["screening_candidate"] = False
    for idx, row in ranked.iterrows():
        if row.get("loss_role") != "custom":
            continue
        same_model = ranked[(ranked["screen_backend"] == row["screen_backend"]) & (ranked["model"] == row["model"])]
        ce = same_model[same_model["loss_key"] == "baseline_ce"]
        asl = same_model[same_model["loss_key"] == "asl_single_label"]
        if ce.empty or asl.empty:
            continue
        ce_f1 = float(ce.iloc[0].get("test_macro_f1") or 0.0)
        asl_f1 = float(asl.iloc[0].get("test_macro_f1") or 0.0)
        best_base_f1 = max(ce_f1, asl_f1)
        best_base_errors = min(float(ce.iloc[0].get("coinfection_boundary_errors") or 0.0), float(asl.iloc[0].get("coinfection_boundary_errors") or 0.0))
        current_f1 = float(row.get("test_macro_f1") or 0.0)
        current_errors = float(row.get("coinfection_boundary_errors") or 0.0)
        beats_both = current_f1 > ce_f1 and current_f1 > asl_f1
        error_benefit = current_errors < best_base_errors and current_f1 >= best_base_f1 - 0.01
        ranked.at[idx, "beats_ce_and_asl_same_model"] = beats_both
        ranked.at[idx, "clear_coinfection_error_benefit"] = error_benefit
        ranked.at[idx, "screening_candidate"] = beats_both or error_benefit
    return ranked.sort_values(
        ["test_macro_f1", "cohen_kappa", "coinfection_boundary_errors", "latency_ms_image", "model_size_mb"],
        ascending=[False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
