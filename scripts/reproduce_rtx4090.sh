#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
python scripts/patch_ultralytics.py
python scripts/verify_checkpoints.py

python scripts/robustness/benchmark_mobile10.py   --baseline checkpoints/yolo11s_baseline_best.pt   --recsra checkpoints/yolo11s_recsra_best.pt   --dataset "${DATASET_ROOT:?Set DATASET_ROOT}"   --output outputs/mobile10_rtx4090   --device 0
