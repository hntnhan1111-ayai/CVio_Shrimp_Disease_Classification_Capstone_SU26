#!/usr/bin/env python3
"""Train YOLO26m-cls with ASL-LDAM + SimAM-DCFR on ShrimpDB-3."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cvio_asl_ldam.utils.io import load_yaml
from cvio_asl_ldam.utils.paths import ProjectPaths, resolve_device
from cvio_asl_ldam.utils.run_guard import decide_action, write_status
from cvio_asl_ldam.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")
DATASET_CONFIG = Path("configs/datasets/shrimpdb.yaml")


def run(
    study_path: str | Path = STUDY_CONFIG,
    dataset_path: str | Path = DATASET_CONFIG,
    device: str = "auto",
    resume: bool = False,
    force: bool = False,
    skip_if_complete: bool = True,
) -> int:
    study = load_yaml(study_path)
    dataset = load_yaml(dataset_path)
    training = study.get("training", {})
    seed = int(study.get("study", {}).get("seed", 42))
    seed_everything(seed)

    paths = ProjectPaths.from_environment()
    run_name = dataset.get("run_name", "yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42")
    run_dir = Path(study.get("training", {}).get("runs_dir", str(paths.output_dir))) / run_name

    action = decide_action(run_dir, skip_if_complete, force, resume)
    if action == "skip":
        LOGGER.info("Run already complete at %s, skipping.", run_dir)
        return 0

    from cvio_asl_ldam.models.yolo_train import run as _run

    class Args:
        data_dir = None
        output_dir = None
        model = training.get("model", "yolo26m-cls")
        seed = seed
        device = device
        resume = resume
        resume_weights = None
        smoke_test = False

    try:
        actual_run_dir = _run(study_path, Args())
        write_status(actual_run_dir, "ok", experiment="shrimpdb3", seed=seed)
        LOGGER.info("ShrimpDB-3 training completed: %s", actual_run_dir)
        return 0
    except Exception as exc:
        write_status(run_dir, "error", error=str(exc))
        LOGGER.error("ShrimpDB-3 training failed: %s", exc)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ShrimpDB-3 experiment")
    parser.add_argument("--config", default=str(STUDY_CONFIG))
    parser.add_argument("--dataset-config", default=str(DATASET_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip if complete")
    args = parser.parse_args()
    return run(
        study_path=args.config,
        dataset_path=args.dataset_config,
        device=args.device,
        resume=args.resume,
        force=args.force,
        skip_if_complete=not args.no_skip,
    )


if __name__ == "__main__":
    sys.exit(main())
