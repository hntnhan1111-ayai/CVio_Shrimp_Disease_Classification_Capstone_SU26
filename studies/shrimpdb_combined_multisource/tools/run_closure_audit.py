import argparse, logging, subprocess, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("closure_audit")

STEPS = [
    ("Configuration validation", ["uv", "run", "pytest", "-q", "tests/test_config_integrity.py"]),
    ("Imports",                 ["uv", "run", "pytest", "-q", "tests/test_imports.py"]),
    ("Label mapping",           ["uv", "run", "pytest", "-q", "tests/test_label_mapping.py"]),
    ("Split protocol",          ["uv", "run", "pytest", "-q", "tests/test_split_protocol.py"]),
    ("Metric regression",       ["uv", "run", "pytest", "-q", "tests/test_metric_regression.py"]),
    ("Report links",            ["uv", "run", "pytest", "-q", "tests/test_report_links.py"]),
    ("Artifact tree",           ["uv", "run", "python", "tools/validate_artifact_tree.py"]),
    ("Checksum verification",   ["uv", "run", "python", "tools/verify_checksums.py"]),
    ("Source comparison",       ["uv", "run", "python", "tools/compare_source_snapshot.py", "--snapshot-root", "artifacts/source_snapshot"]),
]


def run_all(root: Path) -> int:
    failures = 0
    results = []
    for label, cmd in STEPS:
        LOGGER.info("=== %s ===", label)
        try:
            cp = subprocess.run(cmd, cwd=str(root), check=False)
            ok = (cp.returncode == 0)
        except FileNotFoundError as exc:
            LOGGER.error("Command failed to launch: %s", exc)
            ok = False
        results.append({"step": label, "command": cmd, "ok": ok})
        if not ok:
            failures += 1
    summary = {"results": results, "failures": failures, "status": "ok" if failures == 0 else "failed"}
    out = root / "artifacts/metadata/closure_audit.json"
    import json
    out.write_text(json.dumps(summary, indent=2) + chr(10), encoding="utf-8")
    LOGGER.info("Wrote %s (failures=%d)", out, failures)
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    return run_all(Path(args.root))


if __name__ == "__main__":
    sys.exit(main())
