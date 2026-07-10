"""Run all custom attention YOLO model sanity checks."""

from __future__ import annotations

from pathlib import Path
import sys
import traceback


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_attention._shared.sanity_check import sanity_check_yolo_seg_model  # noqa: E402
from custom_attention._shared.yaml_utils import PRIORITY_FOLDERS  # noqa: E402


def ordered_model_yamls() -> list[Path]:
    root = Path(__file__).resolve().parent
    found = [p for p in root.glob("*/model.yaml") if p.parent.name != "_shared"]
    priority = {folder: idx for idx, folder in enumerate(PRIORITY_FOLDERS)}
    return sorted(found, key=lambda p: (priority.get(p.parent.name, 10_000), p.parent.name))


def main() -> None:
    model_paths = ordered_model_yamls()
    results = []
    print(f"Found {len(model_paths)} model.yaml files.")
    for model_path in model_paths:
        module = model_path.parent.name
        print("\n" + "=" * 88)
        print(f"Sanity check: {module}")
        print(f"Model YAML: {model_path}")
        try:
            info = sanity_check_yolo_seg_model(str(model_path), imgsz=640, device="cpu")
            results.append((module, model_path, "PASS", "", info.get("params")))
        except Exception as exc:  # noqa: BLE001 - report every module failure and continue
            error = f"{exc.__class__.__name__}: {exc}"
            print(f"FAIL: {module}: {error}")
            traceback.print_exc()
            results.append((module, model_path, "FAIL", error, None))

    print("\n" + "=" * 88)
    print("Summary")
    print("| module | model path | status | params | error |")
    print("|---|---|---:|---:|---|")
    for module, path, status, error, params in results:
        params_text = f"{params:,}" if params is not None else ""
        print(f"| {module} | {path.as_posix()} | {status} | {params_text} | {error} |")
    passed = sum(1 for item in results if item[2] == "PASS")
    failed = len(results) - passed
    print(f"\nTotal modules: {len(results)} | PASS: {passed} | FAIL: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
