#!/usr/bin/env python3
"""Export a trained YOLO26m-cls ASL-LDAM model to LiteRT/TFLite (FP32 + FP16).

EXPORT-ONLY. This script never trains, never imports a training module's
`train` call, and never downloads the dataset.

Produces:
    <out-dir>/yolo26m_asl_ldam_simam_dcfr_fp32.tflite
    <out-dir>/yolo26m_asl_ldam_simam_dcfr_fp16.tflite
    <out-dir>/tflite_sanity_check.json
    <out-dir>/export_environment.json
    <out-dir>/export_log.txt

Usage:
    python scripts/06_export_litert_fp32_fp16.py \
        --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt \
        --out-dir export --imgsz 224

Recommended: use a clean Python 3.12.2 venv with requirements-export-litert.txt.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FP32_NAME = "yolo26m_asl_ldam_simam_dcfr_fp32.tflite"
FP16_NAME = "yolo26m_asl_ldam_simam_dcfr_fp16.tflite"


def _collect_environment() -> dict:
    report: dict = {"python_version": platform.python_version(), "platform": platform.platform()}
    for name in (
        "ultralytics",
        "tensorflow",
        "tf_keras",
        "onnx",
        "onnx2tf",
        "onnxruntime",
        "onnxslim",
        "onnx_graphsurgeon",
        "sng4onnx",
        "ai_edge_litert",
        "numpy",
        "protobuf",
    ):
        try:
            import importlib

            mod = importlib.import_module(name.replace("-", "_"))
            report[name] = str(getattr(mod, "__version__", "installed"))
        except Exception:
            report[name] = "not installed"
    report["known_good_reference"] = {
        "python": "3.12.2",
        "note": "Python 3.13/base caused TensorFlow/tf-keras problems; use 3.12.2.",
    }
    return report


def _log(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _export_fp32(weights: Path, imgsz: int, work_dir: Path, log: Path) -> Path:
    """Use Ultralytics export to produce an FP32 TFLite file."""
    from cvio_asl_ldam.attention.patch_yolo import register_checkpoint_safe_globals

    register_checkpoint_safe_globals()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    export_path = model.export(format="tflite", imgsz=imgsz, dynamic=False)
    export_path = Path(export_path)
    _log(log, f"Ultralytics exported FP32 TFLite: {export_path}")
    return export_path


def _quantize_fp16(fp32_path: Path, out_path: Path, log: Path) -> Path:
    """Quantize to FP16 by re-converting the SavedModel Ultralytics produced."""
    import tensorflow as tf

    saved_model_dir = fp32_path.parent
    saved_model = None
    for candidate in saved_model_dir.rglob("*_saved_model"):
        if candidate.is_dir():
            saved_model = candidate
            break
    if saved_model is None:
        for candidate in saved_model_dir.rglob("saved_model.pb"):
            saved_model = candidate.parent
            break
    if saved_model is None:
        raise FileNotFoundError(
            "Could not find a SavedModel folder alongside the FP32 TFLite to "
            "perform FP16 quantization. Re-run the FP32 export with Ultralytics."
        )
    _log(log, f"FP16 quantization source SavedModel: {saved_model}")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    fp16_bytes = converter.convert()
    out_path.write_bytes(fp16_bytes)
    _log(log, f"Wrote FP16 TFLite: {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--out-dir", default="export")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--format", default="tflite", help="Export format (tflite only).")
    parser.add_argument("--skip-if-present", action="store_true", help="Skip if both .tflite exist.")
    args = parser.parse_args()

    if args.format != "tflite":
        raise SystemExit("This script only supports --format tflite.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "export_log.txt"
    if log.exists():
        log.unlink()

    fp32_target = out_dir / FP32_NAME
    fp16_target = out_dir / FP16_NAME

    if args.skip_if_present and fp32_target.is_file() and fp16_target.is_file():
        _log(log, "Both TFLite files already present (--skip-if-present); skipping export.")

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}. Export cannot proceed without a checkpoint.")

    # ---- Environment record (written before export for diagnostics) ----
    env = _collect_environment()
    (out_dir / "export_environment.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not (args.skip_if_present and fp32_target.is_file()):
        work_dir = out_dir / "_export_work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        fp32_src = _export_fp32(weights, args.imgsz, work_dir, log)
        shutil.copy2(fp32_src, fp32_target)
        _log(log, f"Copied FP32 -> {fp32_target} ({fp32_target.stat().st_size} bytes)")
        # Cleanup intermediate artifacts.
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        _log(log, f"FP32 already present: {fp32_target}")

    if not (args.skip_if_present and fp16_target.is_file()):
        _quantize_fp16(fp32_target, fp16_target, log)
    else:
        _log(log, f"FP16 already present: {fp16_target}")

    # ---- Verify both files exist ----
    missing = [p for p in (fp32_target, fp16_target) if not p.is_file()]
    if missing:
        raise SystemExit(f"Export failed; missing TFLite files: {missing}")

    # ---- Sanity checks ----
    from cvio_asl_ldam.export import tflite_sanity_check

    sanity = {
        "fp32": tflite_sanity_check(fp32_target),
        "fp16": tflite_sanity_check(fp16_target),
    }
    # FP16 may keep float32 input/output while internal weights are float16.
    sanity["note"] = (
        "FP16 quantization converts internal weights to float16; input/output "
        "tensors may remain float32 depending on the converter/runtime."
    )
    (out_dir / "tflite_sanity_check.json").write_text(
        json.dumps(sanity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _log(log, f"Sanity check OK: input={sanity['fp32']['input_shape']} output={sanity['fp32']['output_shape']}")
    _log(log, "Export complete.")


if __name__ == "__main__":
    main()
