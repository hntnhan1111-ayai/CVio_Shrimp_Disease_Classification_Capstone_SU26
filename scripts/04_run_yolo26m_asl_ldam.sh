#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/datasets/processed-images}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
export PROJECT_ROOT DATA_DIR OUTPUT_DIR SEED DEVICE
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m cvio_asl_ldam.models.yolo_train \
  --config configs/train/yolo26m_asl_ldam_simam_dcfr.yaml \
  --seed "$SEED" \
  --device "$DEVICE" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR"
