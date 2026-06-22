"""Validate the exported TFLite files if present."""

import pytest


from pathlib import Path

EXPORT_DIR = Path(__file__).resolve().parent.parent / "export"


def _tflite_files():
    if not EXPORT_DIR.is_dir():
        return []
    return sorted(EXPORT_DIR.glob("*.tflite"))


def test_tflite_files_exist_or_skip():
    files = _tflite_files()
    if not files:
        pytest.skip("No .tflite files in export/ (run scripts/06_export_litert_fp32_fp16.py)")


def test_tflite_files_are_large_enough():
    files = _tflite_files()
    if not files:
        pytest.skip("No .tflite files present")
    for path in files:
        size_mib = path.stat().st_size / (1024 * 1024)
        assert path.suffix == ".tflite"
        assert size_mib > 1.0, f"{path.name} is only {size_mib:.3f} MiB (expected > 1 MiB)"


def test_tflite_interpreter_sanity_check():
    files = _tflite_files()
    if not files:
        pytest.skip("No .tflite files present")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from cvio_asl_ldam.export import tflite_sanity_check
    except Exception:
        pytest.skip("ai_edge_litert/tensorflow not installed; cannot run interpreter check")
    for path in files:
        try:
            result = tflite_sanity_check(path)
        except ImportError:
            pytest.skip("ai_edge_litert/tensorflow not installed; cannot run interpreter check")
        assert result["status"] == "ok", f"{path.name} sanity check failed: {result}"
        assert len(result["output_shape"]) == 2
        assert result["output_shape"][1] == 4, f"{path.name} expected 4-class output, got {result['output_shape']}"
