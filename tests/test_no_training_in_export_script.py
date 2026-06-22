"""Ensure the export script never trains and does not import a train module."""

from pathlib import Path


EXPORT_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "06_export_litert_fp32_fp16.py"


def test_export_script_has_no_train_call():
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")
    assert ".train(" not in source, "Export script must not call .train("


def test_export_script_does_not_import_training_module():
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")
    # The export script must not import the yolo_train training entry point.
    assert "from cvio_asl_ldam.models import yolo_train" not in source
    assert "import cvio_asl_ldam.models.yolo_train" not in source


def test_export_script_is_export_only_by_docstring():
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")
    assert "EXPORT-ONLY" in source or "export-only" in source.lower()
    assert "never trains" in source.lower()
