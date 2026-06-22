#!/usr/bin/env python3
"""Print and write a machine-readable environment report.

Never trains. Mirrors `scripts/99_env_report.py` (kept for back-compat) but
adds the export-side packages and writes to `artifacts/metadata/env_report.json`.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

# Packages grouped by role. import name -> display label.
TRAIN_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "timm": "timm",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "yaml": "PyYAML",
    "kagglehub": "kagglehub",
    "PIL": "Pillow",
}
EXPORT_PACKAGES = {
    "tensorflow": "tensorflow",
    "tf_keras": "tf-keras",
    "onnx": "onnx",
    "onnx2tf": "onnx2tf",
    "onnxruntime": "onnxruntime",
    "onnxslim": "onnxslim",
    "onnx_graphsurgeon": "onnx-graphsurgeon",
    "sng4onnx": "sng4onnx",
    "ai_edge_litert": "ai-edge-litert",
}


def _version(import_name: str) -> str:
    try:
        module = importlib.import_module(import_name)
        return str(getattr(module, "__version__", "installed (version unknown)"))
    except Exception as exc:
        return f"not installed ({exc.__class__.__name__})"


def _cuda_report() -> tuple[bool, list[str]]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        names = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if available
            else []
        )
        return available, names
    except Exception:
        return False, []


def build_report() -> dict[str, Any]:
    project_root = Path.cwd()
    cuda_available, gpu_names = _cuda_report()
    data_root = Path("datasets/processed-images")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "project_root": str(project_root),
        "data_root": str(data_root),
        "cuda_available": cuda_available,
        "gpu_names": gpu_names,
        "training_packages": {label: _version(name) for name, label in TRAIN_PACKAGES.items()},
        "export_packages": {label: _version(name) for name, label in EXPORT_PACKAGES.items()},
        "known_good_export_environment": {
            "python": "3.12.2",
            "ultralytics": "8.4.75",
            "tensorflow": "2.19.1",
            "tf_keras": "2.19.0",
            "numpy": "2.1.3",
            "protobuf": "5.29.6",
            "onnx": "1.22.0",
            "onnxslim": "0.1.94",
            "onnxruntime": "1.27.0",
            "onnx2tf": "1.28.8",
            "sng4onnx": "2.0.1",
            "onnx_graphsurgeon": "0.6.1",
            "ai_edge_litert": "2.1.5",
        },
        "note": (
            "Python 3.13/base environments caused TensorFlow/tf-keras problems. "
            "Use a clean Python 3.12.2 venv for LiteRT/TFLite export."
        ),
    }


def main() -> None:
    report = build_report()
    out_json = Path("artifacts/metadata/env_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
