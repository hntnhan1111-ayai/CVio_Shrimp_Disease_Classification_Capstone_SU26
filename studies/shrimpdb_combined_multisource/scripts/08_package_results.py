#!/usr/bin/env python3
"""Package key artifacts into ZIP and write registry files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any

from cvio_asl_ldam.utils.io import load_yaml, write_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")
EXCLUDE_DIRS = {"raw_datasets", ".venv", "__pycache__", ".git", "node_modules"}
MAX_FILE_SIZE_MB = 100


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_exclude(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    return bool(parts & EXCLUDE_DIRS)


def package_artifacts(output_zip: Path, root: Path = Path(".")) -> int:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if _should_exclude(path, root):
                continue
            if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                LOGGER.warning("Skipping large file: %s", path)
                continue
            arcname = str(path.relative_to(root))
            zf.write(path, arcname)
            LOGGER.debug("Packed %s", arcname)
    LOGGER.info("Wrote package: %s", output_zip)
    return 0


def write_registry(study: dict[str, Any]) -> int:
    model_registry: dict[str, Any] = {"checkpoints": {}}
    experiments = study.get("experiments", {})
    for exp_name, exp_cfg in experiments.items():
        run_name = exp_cfg.get("run_name", f"yolo26m_cls__asl_ldam_simam_dcfr__{exp_name}__seed42")
        best_pt = Path("runs") / run_name / "weights" / "best.pt"
        if best_pt.is_file():
            model_registry["checkpoints"][exp_name] = {
                "path": str(best_pt),
                "sha256": _sha256_file(best_pt),
                "classes": exp_cfg.get("class_names", []),
            }

    model_registry_path = Path("model_registry/checkpoints.json")
    model_registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(model_registry_path, model_registry)

    class_mapping: dict[str, Any] = {}
    for exp_name, exp_cfg in experiments.items():
        class_mapping[exp_name] = {
            "class_names": exp_cfg.get("class_names", []),
            "source_mapping": study.get("experiments", {}).get(exp_name, {}).get("source_mapping", {}),
        }
    write_json(Path("model_registry/class_mapping.json"), class_mapping)

    final_results: dict[str, Any] = {"experiments": {}}
    for exp_name in ("shrimpdb3", "combined4"):
        eval_dir = Path("evaluation") / exp_name / "best"
        metrics_path = eval_dir / "metrics_raw.json"
        if metrics_path.is_file():
            final_results["experiments"][exp_name] = json.loads(metrics_path.read_text(encoding="utf-8"))
    write_json(Path("FINAL_RESULTS.json"), final_results)
    LOGGER.info("Wrote model_registry and FINAL_RESULTS.json")
    return 0


def run(output_zip: Path = Path("artifacts/CVio_ShrimpDB_Combined_seed42_package.zip"), study_path: Path = STUDY_CONFIG) -> int:
    study = load_yaml(study_path)
    package_artifacts(output_zip)
    write_registry(study)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Package results")
    parser.add_argument("--output", default="artifacts/CVio_ShrimpDB_Combined_seed42_package.zip")
    parser.add_argument("--config", default=str(STUDY_CONFIG))
    args = parser.parse_args()
    try:
        return run(Path(args.output), Path(args.config))
    except Exception as exc:
        LOGGER.error("Packaging failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
