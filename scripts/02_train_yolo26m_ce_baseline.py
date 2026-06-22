#!/usr/bin/env python3
"""Train the YOLO26m-cls cross-entropy baseline (seed-42).

Trains ONLY the CE baseline. Never exports. Run-overwrite guards prevent
silent overwrites of reviewer results:

  --skip-if-complete  skip if status.json reports ok and best.pt exists (default)
  --force             delete the run directory before training
  --resume            resume from last.pt
  --run-name          use a custom run directory name
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvio_asl_ldam.models import yolo_train
from cvio_asl_ldam.utils.io import load_yaml
from cvio_asl_ldam.utils.run_guard import decide_action, is_run_complete, write_status

CONFIG = PROJECT_ROOT / "configs" / "train" / "yolo26m_ce.yaml"


def _run_dir(args: argparse.Namespace) -> Path:
    config = load_yaml(args.config)
    seed = int(args.seed or config["seed"])
    model = args.model or str(config["model"])
    method = str(config.get("method", "baseline_ce"))
    name = args.run_name or f"{model.replace('-', '_')}__{method}__seed{seed}"
    out_root = Path(args.output_dir or "runs")
    return out_root / "training" / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--data", default="runs/prepared_seed42", help="Prepared split root.")
    parser.add_argument("--data-dir", help="Raw dataset root (if split not prepared).")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--model", default="yolo26m-cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-if-complete", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_dir = _run_dir(args)
    action = decide_action(run_dir, args.skip_if_complete, args.force, args.resume)

    if action == "skip":
        print(f"Run already complete, skipping: {run_dir}")
        return
    if action == "force":
        if run_dir.exists():
            print(f"--force: removing existing run directory {run_dir}")
            shutil.rmtree(run_dir)

    if args.resume and (run_dir / "weights" / "last.pt").is_file():
        # Ultralytics resume re-uses the run directory.
        args.resume_weights = str(run_dir / "weights" / "last.pt")
    else:
        args.resume = False

    print(f"[CE baseline] action={action} run_dir={run_dir}")
    write_status(run_dir, "running", action=action, method="baseline_ce", seed=args.seed)

    args.data_dir = args.data_dir or args.data
    completed = yolo_train.run(args.config, args)

    if is_run_complete(completed):
        write_status(
            completed,
            "ok",
            action=action,
            method="baseline_ce",
            seed=args.seed,
            best_pt=str(completed / "weights" / "best.pt"),
        )
        print(f"CE baseline training complete: {completed}")
    else:
        write_status(completed, "incomplete", action=action, method="baseline_ce", seed=args.seed)
        raise SystemExit(f"Training finished without a complete checkpoint at {completed}")


if __name__ == "__main__":
    main()
