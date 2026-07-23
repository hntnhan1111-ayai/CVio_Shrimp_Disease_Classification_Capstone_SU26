#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
python scripts/capture_runtime.py --output outputs/kaggle_t4x2_runtime.json
python scripts/patch_ultralytics.py
python scripts/verify_checkpoints.py

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
[[ "$GPU_COUNT" -ge 2 ]] || { echo "Kaggle T4×2 target requires two visible CUDA GPUs" >&2; exit 1; }

python - <<'PYGPU'
import torch
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) < 2 or not all("T4" in name for name in names[:2]):
    raise SystemExit(f"Expected two Tesla T4 GPUs; found {names}")
print("[T4x2 VERIFIED]", names[:2])
PYGPU

# The matched pair is evaluated sequentially on T4 GPU 0 to avoid timing and
# memory interference between models. GPU 1 remains available for a separate
# process or future parallelized implementation.
PYTHONPATH=src python scripts/robustness/benchmark_mobile10.py   --baseline checkpoints/yolo11s_baseline_best.pt   --recsra checkpoints/yolo11s_recsra_best.pt   --dataset "${DATASET_ROOT:?Set DATASET_ROOT}"   --output outputs/mobile10_kaggle_t4x2   --device 0
