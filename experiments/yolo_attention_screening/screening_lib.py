"""Shared helpers for the YOLOv26m-cls attention screening experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from shrimp_scripts import config
from shrimp_scripts.attention import attention_is_baseline, attention_keys, attention_metadata, module_unit_test, normalize_attention_key
from shrimp_scripts.models_yolo import make_run_config as make_yolo_run_config
from shrimp_scripts.models_yolo import paper_yolo_names, train_yolo_with_fallback, yolo_run_id
from shrimp_scripts.utils import ensure_dir, read_json, save_csv, stable_hash, validate_run_completion, write_json, write_status


DEFAULT_OUTPUT_DIR = Path("/kaggle/working/shrimp_outputs_yolo_attention_screening")
YOLO_ATTENTION_SCREEN_EPOCHS = 8
EXPERIMENT_KEY = "yolo_attention_screening"
EXPERIMENT_GROUP = "yolo_attention_screening"
ATTENTION_RANKING_COLUMNS = [
    ("test_macro_f1", False),
    ("cohen_kappa", False),
    ("test_accuracy", False),
    ("latency_ms_image", True),
    ("fps", False),
    ("model_size_mb", True),
    ("params_m", True),
]


def condition_for_attention(attention_key: str, epochs: int = YOLO_ATTENTION_SCREEN_EPOCHS) -> dict[str, Any]:
    key = normalize_attention_key(attention_key)
    return {
        "condition_key": f"attention_{key}_ce_no_randaugment",
        "loss_key": "baseline_ce",
        "randaugment": False,
        "attention_key": key,
        "epochs": int(epochs),
        "experiment_key": EXPERIMENT_KEY,
        "experiment_group": EXPERIMENT_GROUP,
        "screening_note": "YOLOv26m-cls attention screening; CE only; RandAugment disabled.",
    }


def list_attention_runs(epochs: int = YOLO_ATTENTION_SCREEN_EPOCHS) -> list[dict[str, Any]]:
    rows = []
    for key in attention_keys():
        condition = condition_for_attention(key, epochs=epochs)
        rows.append({
            "run_id": yolo_run_id(config.YOLO_CORE_MODEL, condition),
            "backend": "ultralytics",
            "model": config.YOLO_CORE_MODEL,
            **condition,
        })
    return rows


def write_attention_plan(output_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_dir = ensure_dir(output_dir)
    return save_csv(pd.DataFrame(rows), output_dir / "attention_screening_plan.csv")


def attention_audit_valid(run_dir: str | Path, expected_key: str) -> tuple[bool, list[str], dict[str, Any]]:
    run_dir = Path(run_dir)
    audit_path = run_dir / "attention_module_audit.json"
    if not audit_path.is_file() or audit_path.stat().st_size == 0:
        return False, ["missing_or_empty:attention_module_audit.json"], {}
    audit = read_json(audit_path, default={})
    errors: list[str] = []
    key = normalize_attention_key(audit.get("attention_key") or audit.get("requested_attention_key"))
    expected = normalize_attention_key(expected_key)
    if key != expected:
        errors.append(f"attention_key_mismatch:{key}!={expected}")
    for required_key in ["implementation_source", "implementation_variant", "params_added", "is_exact_official_implementation"]:
        if required_key not in audit:
            errors.append(f"missing_attention_audit_field:{required_key}")
    if attention_is_baseline(expected):
        trainer = audit.get("trainer_attention_audit", audit)
        if trainer.get("status") != "baseline_no_attention":
            errors.append(f"baseline_status_not_baseline_no_attention:{trainer.get('status')}")
        if bool(trainer.get("inserted", False)):
            errors.append("baseline_inserted_true")
    else:
        trainer = audit.get("trainer_attention_audit", audit)
        if trainer.get("status") != "inserted":
            errors.append(f"attention_status_not_inserted:{trainer.get('status')}")
        if not bool(trainer.get("inserted", False)):
            errors.append("attention_inserted_false")
        if not trainer.get("target_module_class"):
            errors.append("missing_target_module_class")
        if not trainer.get("verified_output_shape"):
            errors.append("missing_verified_output_shape")
    return not errors, errors, audit


def validate_attention_resume_rows(rows: list[dict[str, Any]], output_dir: str | Path, smoke_test: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    flat_rows = []
    detailed = []
    for row in rows:
        condition = condition_for_attention(row["attention_key"], epochs=int(row["epochs"]))
        validations = []
        for batch in config.YOLO_BATCH_FALLBACKS:
            run_config = make_yolo_run_config(row["model"], condition, output_dir, smoke_test=smoke_test, batch=batch)
            validation = validate_run_completion(row["run_id"], output_dir, stable_hash(run_config), "ultralytics")
            audit_ok, audit_errors, _audit = attention_audit_valid(output_dir / "runs" / row["run_id"], row["attention_key"])
            if not audit_ok:
                validation["errors"].extend(f"attention_audit:{error}" for error in audit_errors)
                if validation["validation_status"] == "completed_with_valid_final_metrics":
                    validation["validation_status"] = "incomplete_missing_final_metrics"
                    validation["resume_decision"] = "will_fresh_rerun"
            validations.append({"batch": batch, **validation})
        valid = next((item for item in validations if item["validation_status"] == "completed_with_valid_final_metrics"), None)
        errors = sorted({str(error) for item in validations for error in item.get("errors", [])})
        flat_rows.append({
            "run_id": row["run_id"],
            "backend": "ultralytics",
            "model": row["model"],
            "attention_key": row["attention_key"],
            "condition_key": row["condition_key"],
            "validation_status": "completed_with_valid_final_metrics" if valid else "incomplete_missing_final_metrics",
            "resume_decision": "skipped_completed_with_final_metrics" if valid else "will_fresh_rerun",
            "valid_batch": valid["batch"] if valid else "",
            "errors": " | ".join(errors),
        })
        detailed.append({"run_id": row["run_id"], "validations": validations})
    summary = {
        "planned_count": len(rows),
        "completed_with_valid_final_metrics": sum(1 for row in flat_rows if row["validation_status"] == "completed_with_valid_final_metrics"),
        "will_fresh_rerun": sum(1 for row in flat_rows if row["resume_decision"] == "will_fresh_rerun"),
        "runs": flat_rows,
    }
    write_json(output_dir / "attention_resume_validation.json", {"summary": summary, "details": detailed})
    save_csv(pd.DataFrame(flat_rows), output_dir / "attention_resume_validation.csv")
    return summary


def run_attention_module_unit_test(output_dir: str | Path) -> dict[str, Any]:
    try:
        result = module_unit_test()
    except Exception as exc:
        result = {"status": "failed", "error": repr(exc), "rows": [], "errors": [repr(exc)]}
    write_json(Path(output_dir) / "attention_module_unit_test.json", result)
    return result


def run_model_injection_test(output_dir: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    result_rows = []
    errors = []
    try:
        import torch
        from ultralytics import YOLO
        from shrimp_scripts.attention import assert_valid_classification_output, inject_attention_before_classify
    except Exception as exc:
        result = {"status": "failed", "error": repr(exc), "rows": []}
        write_json(output_dir / "attention_model_injection_test.json", result)
        return result
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for row in rows:
        key = row["attention_key"]
        item = {"attention_key": key, "status": "failed", "device": str(device)}
        try:
            supported, support_reason = attention_supported_before_training(key)
            if not supported:
                item.update({
                    "status": "skipped_not_compatible",
                    "skip_reason": support_reason,
                    **attention_metadata(key),
                })
                result_rows.append(item)
                continue
            yolo = YOLO(config.YOLO_CORE_MODEL + ".pt")
            model = yolo.model.to(device)
            model.names = paper_yolo_names()
            audit = inject_attention_before_classify(model, key, img_size=config.IMG_SIZE, num_classes=config.NUM_CLASSES)
            model.eval()
            with torch.no_grad():
                output_shape = assert_valid_classification_output(
                    model(torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)),
                    config.NUM_CLASSES,
                )
            item.update({
                "status": "passed",
                "output_shape": tuple(int(value) for value in output_shape),
                "audit": audit,
                "module_tree_tail": [module.__class__.__name__ for module in list(model.model)[-5:]],
                **attention_metadata(key),
            })
        except Exception as exc:
            item["error"] = repr(exc)
            errors.append(f"{key}:{repr(exc)}")
        result_rows.append(item)
    result = {
        "status": "passed" if not errors else "failed",
        "rows": result_rows,
        "errors": errors,
    }
    write_json(output_dir / "attention_model_injection_test.json", result)
    save_csv(pd.DataFrame([{k: json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()} for row in result_rows]), output_dir / "attention_model_injection_test.csv")
    return result


def mark_attention_skipped(output_dir: str | Path, row: dict[str, Any], reason: str, batch: int | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    run_dir = ensure_dir(output_dir / "runs" / row["run_id"])
    condition = condition_for_attention(row["attention_key"], epochs=int(row["epochs"]))
    chosen_batch = int(batch or config.YOLO_BATCH_FALLBACKS[0])
    run_config = make_yolo_run_config(row["model"], condition, output_dir, smoke_test=False, batch=chosen_batch)
    run_hash = stable_hash(run_config)
    attention_audit = {
        "attention_key": row["attention_key"],
        "requested_attention_key": row["attention_key"],
        "status": "skipped_not_compatible",
        "inserted": False,
        "params_added": 0,
        **attention_metadata(row["attention_key"]),
        "reason": reason,
    }
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "attention_module_audit.json", attention_audit)
    write_json(run_dir / "run_audit.json", {
        **run_config,
        "config_hash": run_hash,
        "status": "skipped_not_compatible",
        "skip_reason": reason,
        "attention_module_audit_path": str((run_dir / "attention_module_audit.json").resolve()),
    })
    write_status(run_dir, "skipped", run_id=row["run_id"], error=reason)
    return {"run_id": row["run_id"], "status": "skipped_not_compatible", "error": reason}


def attention_supported_before_training(attention_key: str) -> tuple[bool, str]:
    key = normalize_attention_key(attention_key)
    if key != "c2psa_or_psa":
        return True, "no_preflight_required"
    try:
        from ultralytics.nn.modules.block import C2PSA  # noqa: F401
        return True, "ultralytics.nn.modules.block.C2PSA"
    except Exception as c2psa_exc:
        try:
            from ultralytics.nn.modules.block import PSA  # noqa: F401
            return True, "ultralytics.nn.modules.block.PSA"
        except Exception as psa_exc:
            return False, f"c2psa_or_psa_unavailable:c2psa={c2psa_exc!r};psa={psa_exc!r}"


def train_attention_row(row: dict[str, Any], yolo_manifest: pd.DataFrame, output_dir: str | Path, resume: bool, progress_enabled: bool) -> dict[str, Any]:
    condition = condition_for_attention(row["attention_key"], epochs=int(row["epochs"]))
    return train_yolo_with_fallback(
        row["model"],
        condition,
        yolo_manifest,
        output_dir,
        resume=resume,
        smoke_test=False,
        progress_enabled=progress_enabled,
    )
