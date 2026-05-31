"""Project-independent utilities for experiment scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, np.integer):
        return int(obj)
    if np is not None and isinstance(obj, np.floating):
        return float(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if default is not None and not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    return path


def save_csv(frame: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
    return path


def file_hash(path: str | Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str | Path) -> str:
    return file_hash(path, "md5")


def sha256_file(path: str | Path) -> str:
    return file_hash(path, "sha256")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unavailable"


def set_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        return {
            "seed": seed,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_warn_only": True,
        }
    except Exception as exc:
        return {"seed": seed, "torch_seed_status": repr(exc)}


def environment_versions(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit_learn": package_version("scikit-learn"),
        "pillow": package_version("pillow"),
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "timm": package_version("timm"),
        "ultralytics": package_version("ultralytics"),
        "grad_cam": package_version("grad-cam"),
        "opencv_python": package_version("opencv-python"),
        "kagglehub": package_version("kagglehub"),
        "openpyxl": package_version("openpyxl"),
    }
    try:
        import torch
        payload.update({
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": str(torch.version.cuda),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        })
    except Exception as exc:
        payload.update({"cuda_available": False, "torch_runtime_error": repr(exc)})
    if extra:
        payload.update(extra)
    return payload


def is_oom_error(exc: BaseException) -> bool:
    text = repr(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


FINAL_TEST_METRIC_FIELDS = (
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "cohen_kappa",
)


def truthy_file(path: str | Path) -> bool:
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def read_json_safely(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not truthy_file(path):
        return {}, f"missing_or_empty:{path.name}"
    try:
        payload = read_json(path)
    except Exception as exc:
        return {}, f"invalid_json:{path.name}:{repr(exc)}"
    if not isinstance(payload, dict):
        return {}, f"json_not_object:{path.name}"
    return payload, ""


def finite_float(value: Any) -> bool:
    try:
        import math
        return math.isfinite(float(value))
    except Exception:
        return False


def extract_final_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics.get("test", {}) if isinstance(metrics.get("test", {}), dict) else {}
    return {
        "test_accuracy": metrics.get("test_accuracy", test.get("accuracy")),
        "test_macro_precision": metrics.get("test_macro_precision", test.get("macro_precision")),
        "test_macro_recall": metrics.get("test_macro_recall", test.get("macro_recall")),
        "test_macro_f1": metrics.get("test_macro_f1", test.get("macro_f1")),
        "cohen_kappa": metrics.get("cohen_kappa", test.get("cohen_kappa")),
    }


def final_test_metrics_valid(metrics: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    values = extract_final_test_metrics(metrics)
    missing = [name for name, value in values.items() if not finite_float(value)]
    return len(missing) == 0, missing, values


def required_outputs_exist(run_dir: str | Path, backend: str) -> tuple[bool, list[str]]:
    run_dir = Path(run_dir)
    required = [
        run_dir / "metrics.json",
        run_dir / "val_predictions.csv",
        run_dir / "test_predictions.csv",
        run_dir / "classification_report.csv",
        run_dir / "confusion_counts.csv",
    ]
    confusion_options = [run_dir / "confusion_matrix.csv", run_dir / "confusion_matrix.json"]
    missing = [str(path.name) for path in required if not truthy_file(path)]
    if not any(truthy_file(path) for path in confusion_options):
        missing.append("confusion_matrix.csv or confusion_matrix.json")
    if backend in {"torchvision", "timm", "torch"}:
        checkpoint = run_dir / "checkpoint_best.pt"
        if not truthy_file(checkpoint):
            missing.append("checkpoint_best.pt")
    if backend == "ultralytics":
        audit, _audit_error = read_json_safely(run_dir / "run_audit.json")
        metrics, _metrics_error = read_json_safely(run_dir / "metrics.json")
        best_text = str(audit.get("checkpoint_path") or metrics.get("checkpoint_path") or "").strip()
        best_path = Path(best_text) if best_text else run_dir / "best.pt"
        if not truthy_file(best_path):
            missing.append("best.pt")
        if not truthy_file(run_dir / "class_order_audit.json"):
            missing.append("class_order_audit.json")
    return len(missing) == 0, missing


def yolo_class_order_audit_valid(run_dir: str | Path) -> tuple[bool, list[str], dict[str, Any]]:
    audit, error = read_json_safely(Path(run_dir) / "class_order_audit.json")
    if error:
        return False, [error], audit
    errors: list[str] = []
    if audit.get("audit_passed") is not True:
        errors.append("audit_passed_not_true")
    swap = audit.get("swap_diagnostic", {})
    if isinstance(swap, dict) and swap.get("failed_due_to_swap_diagnostic") is True:
        errors.append("swap_diagnostic_failed")
    if audit.get("failed_due_to_swap_diagnostic") is True:
        errors.append("swap_diagnostic_failed")
    return len(errors) == 0, errors, audit


def validate_run_completion(run_id: str, output_dir: str | Path, expected_hash: str, backend: str) -> dict[str, Any]:
    run_dir = Path(output_dir) / "runs" / run_id
    result: dict[str, Any] = {
        "run_id": run_id,
        "backend": backend,
        "run_dir": str(run_dir),
        "validation_status": "incomplete_missing_final_metrics",
        "resume_decision": "will_fresh_rerun",
        "errors": [],
        "missing_outputs": [],
        "final_metrics": {},
    }
    if not run_dir.exists():
        result["errors"].append("missing_run_directory")
        return result
    status, status_error = read_json_safely(run_dir / "status.json")
    audit, audit_error = read_json_safely(run_dir / "run_audit.json")
    metrics, metrics_error = read_json_safely(run_dir / "metrics.json")
    if status_error:
        result["errors"].append(status_error)
    if audit_error:
        result["errors"].append(audit_error)
    if metrics_error:
        result["errors"].append(metrics_error)
    status_value = status.get("status")
    if status_value == "failed":
        result["validation_status"] = "failed"
    elif status_value == "skipped":
        result["validation_status"] = "skipped"
    elif status_value != "completed":
        result["errors"].append(f"status_not_completed:{status_value or 'missing'}")
    if audit and audit.get("config_hash") != expected_hash:
        result["errors"].append("config_hash_mismatch")
    outputs_ok, missing_outputs = required_outputs_exist(run_dir, backend)
    result["missing_outputs"] = missing_outputs
    if not outputs_ok:
        result["errors"].extend(f"missing_output:{name}" for name in missing_outputs)
    metrics_ok, missing_metrics, metric_values = final_test_metrics_valid(metrics)
    result["final_metrics"] = metric_values
    if not metrics_ok:
        result["errors"].extend(f"invalid_or_missing_metric:{name}" for name in missing_metrics)
    if backend == "ultralytics":
        audit_ok, audit_errors, class_order_audit = yolo_class_order_audit_valid(run_dir)
        result["class_order_audit"] = {
            "audit_passed": class_order_audit.get("audit_passed"),
            "class_order_match": class_order_audit.get("class_order_match"),
            "swap_diagnostic": class_order_audit.get("swap_diagnostic", {}),
        }
        if not audit_ok:
            result["errors"].extend(f"class_order_audit:{error}" for error in audit_errors)
    if status_value == "completed" and not result["errors"]:
        result["validation_status"] = "completed_with_valid_final_metrics"
        result["resume_decision"] = "skipped_completed_with_final_metrics"
    elif result["validation_status"] not in {"failed", "skipped"}:
        result["validation_status"] = "incomplete_missing_final_metrics"
    return result


def is_run_completed(run_id: str, output_dir: str | Path, expected_hash: str, backend: str) -> tuple[bool, str]:
    validation = validate_run_completion(run_id, output_dir, expected_hash, backend)
    if validation["validation_status"] == "completed_with_valid_final_metrics":
        return True, "completed_with_valid_final_metrics"
    errors = validation.get("errors", [])
    if errors:
        return False, ";".join(str(error) for error in errors)
    return False, str(validation["validation_status"])


def archive_run_dir_for_fresh_rerun(run_dir: str | Path, reason: str) -> Path | None:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(reason).strip())[:80] or "incomplete"
    archive_root = ensure_dir(run_dir.parent / "_archived_incomplete_runs")
    stamp = utc_now().replace(":", "").replace("+", "Z")
    target = archive_root / f"{run_dir.name}__{stamp}__{safe_reason}"
    counter = 1
    while target.exists():
        target = archive_root / f"{run_dir.name}__{stamp}__{safe_reason}_{counter}"
        counter += 1
    shutil.move(str(run_dir), str(target))
    return target


def completion_validation_report(planned_runs: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    rows = []
    for row in planned_runs:
        run_id = str(row["run_id"])
        backend = str(row.get("backend", ""))
        expected_hash = str(row.get("config_hash", ""))
        if not expected_hash:
            rows.append({
                "run_id": run_id,
                "backend": backend,
                "validation_status": "incomplete_missing_final_metrics",
                "resume_decision": "will_fresh_rerun",
                "errors": ["missing_expected_config_hash"],
            })
            continue
        rows.append(validate_run_completion(run_id, output_dir, expected_hash, backend))
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["validation_status"])
        counts[status] = counts.get(status, 0) + 1
        decision = str(row.get("resume_decision", ""))
        if decision:
            counts[decision] = counts.get(decision, 0) + 1
    return {"rows": rows, "counts": counts}



def write_status(run_dir: str | Path, status: str, **kwargs: Any) -> Path:
    payload = {"status": status, "timestamp_utc": utc_now()}
    payload.update(kwargs)
    return write_json(Path(run_dir) / "status.json", payload)


def cleanup_memory() -> None:
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def zip_directory(source_dir: str | Path, zip_path: str | Path, exclude_names: set[str] | None = None) -> Path:
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    ensure_dir(zip_path.parent)
    exclude_names = exclude_names or set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_dir.rglob("*"):
            if path.is_dir() or path.name in exclude_names:
                continue
            archive.write(path, path.relative_to(source_dir))
    return zip_path


def safe_unlink_tree(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def bool_arg(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}
