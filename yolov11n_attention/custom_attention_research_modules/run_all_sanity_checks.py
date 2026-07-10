"""Run sanity checks for all research attention YOLO model YAMLs."""

from __future__ import annotations

from pathlib import Path
import sys
import traceback


THIS_FILE = Path(__file__).resolve()
ATTENTION_ROOT = THIS_FILE.parents[1]
if str(ATTENTION_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTENTION_ROOT))

from custom_attention_research_modules._shared.sanity_check import sanity_check_yolo_seg_model  # noqa: E402


MODULES = (
    ("CoTE-Gate", Path("custom_attention_research_modules/cote_gate/cote_gate.yaml")),
    ("SCSG-Gate", Path("custom_attention_research_modules/scsg_gate/scsg_gate.yaml")),
    ("LPSC-Gate", Path("custom_attention_research_modules/lpsc_gate/lpsc_gate.yaml")),
)


def main() -> None:
    results = []
    for module, yaml_path in MODULES:
        print("\n" + "=" * 88)
        print(f"Sanity check: {module}")
        print(f"YAML: {yaml_path}")
        try:
            info = sanity_check_yolo_seg_model(str(yaml_path), imgsz=640, device="cpu")
            results.append((module, yaml_path, "PASS", "", info.get("params")))
        except Exception as exc:  # noqa: BLE001 - continue and report all failures
            error = f"{exc.__class__.__name__}: {exc}"
            print(f"FAIL: {module}: {error}")
            traceback.print_exc()
            results.append((module, yaml_path, "FAIL", error, None))

    print("\n" + "=" * 88)
    print("Summary")
    print("| module | yaml path | status | params | error |")
    print("|---|---|---:|---:|---|")
    for module, yaml_path, status, error, params in results:
        params_text = f"{params:,}" if params is not None else ""
        print(f"| {module} | {yaml_path.as_posix()} | {status} | {params_text} | {error} |")

    passed = sum(1 for item in results if item[2] == "PASS")
    failed = len(results) - passed
    print(f"\nTotal modules: {len(results)} | PASS: {passed} | FAIL: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
