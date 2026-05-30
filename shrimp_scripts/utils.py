"""Project-independent utilities for experiment scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
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


def required_outputs_exist(run_dir: str | Path, backend: str) -> tuple[bool, list[str]]:
    run_dir = Path(run_dir)
    required = [
        run_dir / "metrics.json",
        run_dir / "test_predictions.csv",
        run_dir / "classification_report.csv",
    ]
    confusion_options = [run_dir / "confusion_matrix.csv", run_dir / "confusion_matrix.json"]
    missing = [str(path.name) for path in required if not path.is_file() or path.stat().st_size == 0]
    if not any(path.is_file() and path.stat().st_size > 0 for path in confusion_options):
        missing.append("confusion_matrix.csv or confusion_matrix.json")
    if backend in {"torchvision", "timm", "torch"}:
        checkpoint = run_dir / "checkpoint_best.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            missing.append("checkpoint_best.pt")
    if backend == "ultralytics":
        audit = read_json(run_dir / "run_audit.json", default={})
        best_path = Path(str(audit.get("checkpoint_path", ""))) if audit.get("checkpoint_path") else run_dir / "best.pt"
        if not best_path.is_file() or best_path.stat().st_size == 0:
            missing.append("best.pt")
    return len(missing) == 0, missing


def is_run_completed(run_id: str, output_dir: str | Path, expected_hash: str, backend: str) -> tuple[bool, str]:
    run_dir = Path(output_dir) / "runs" / run_id
    status_path = run_dir / "status.json"
    audit_path = run_dir / "run_audit.json"
    if not status_path.exists():
        return False, "missing_status"
    if not audit_path.exists():
        return False, "missing_run_audit"
    status = read_json(status_path, default={})
    audit = read_json(audit_path, default={})
    if status.get("status") != "completed":
        return False, "status_not_completed"
    if audit.get("config_hash") != expected_hash:
        return False, "config_hash_mismatch"
    ok, missing = required_outputs_exist(run_dir, backend)
    if not ok:
        return False, "missing_outputs:" + ",".join(missing)
    return True, "completed"


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

