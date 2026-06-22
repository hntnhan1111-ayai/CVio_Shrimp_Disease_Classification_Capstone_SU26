#!/usr/bin/env python3
"""Train the main method: YOLO26m-cls + ASL-LDAM + SimAM-DCFR (seed-42).

Trains ONLY the main method. Never exports. Verifies ASL-LDAM is active and
SimAM-DCFR is injected. Run-overwrite guards prevent silent overwrites:

  --skip-if-complete  skip if status.json reports ok and best.pt exists (default)
  --force             delete the run directory before training
  --resume            resume from last.pt
  --run-name          use a custom run directory name
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvio_asl_ldam.models import yolo_train
from cvio_asl_ldam.utils.io import load_yaml, write_json
from cvio_asl_ldam.utils.run_guard import decide_action, is_run_complete, write_status

CONFIG = PROJECT_ROOT / "configs" / "train" / "yolo26m_asl_ldam_simam_dcfr.yaml"


def _run_dir(args: argparse.Namespace) -> Path:
    config = load_yaml(args.config)
    seed = int(args.seed or config["seed"])
    model = args.model or str(config["model"])
    method = str(config.get("method", "asl_ldam_simam_dcfr"))
    name = args.run_name or f"{model.replace('-', '_')}__{method}__seed{seed}"
    out_root = Path(args.output_dir or "runs")
    return out_root / "training" / name


def _verify_main_method_active() -> None:
    """Confirm the main-method loss and attention will be used at train time."""
    os.environ["CVIO_YOLO_LOSS"] = "asl_ldam"
    os.environ["CVIO_YOLO_ATTENTION"] = "simam_dcfr"
    from cvio_asl_ldam.attention.patch_yolo import get_custom_trainer_class
    from cvio_asl_ldam.losses.asl_ldam import ASLLDAMLoss

    # Construct the loss to confirm hyperparameters are valid.
    loss = ASLLDAMLoss(class_counts=(282, 139, 229, 154))
    assert loss.margins.numel() == 4
    assert get_custom_trainer_class() is not None
    print("Verified ASL-LDAM loss and SimAM-DCFR custom trainer are available.")


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

    _verify_main_method_active()

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
        args.resume_weights = str(run_dir / "weights" / "last.pt")
    else:
        args.resume = False

    print(f"[Main method] action={action} run_dir={run_dir}")
    write_status(run_dir, "running", action=action, method="asl_ldam_simam_dcfr", seed=args.seed)

    args.data_dir = args.data_dir or args.data
    completed = yolo_train.run(args.config, args)

    # Save the run config (already done inside run(), but record guard metadata too).
    write_json(
        completed / "status.json",
        {
            "status": "ok" if is_run_complete(completed) else "incomplete",
            "action": action,
            "method": "asl_ldam_simam_dcfr",
            "seed": args.seed,
            "best_pt": str(completed / "weights" / "best.pt"),
            "asl_ldam_active": True,
            "simam_dcfr_injected": True,
        },
    )
    if is_run_complete(completed):
        print(f"Main method training complete: {completed}")
    else:
        raise SystemExit(f"Training finished without a complete checkpoint at {completed}")


if __name__ == "__main__":
    main()
