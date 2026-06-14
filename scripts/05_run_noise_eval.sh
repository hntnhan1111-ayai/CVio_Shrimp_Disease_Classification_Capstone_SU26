#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/datasets/processed-images}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
WEIGHTS="${WEIGHTS:-$OUTPUT_DIR/training/yolo26m_cls__asl_ldam_simam_dcfr__seed${SEED}/weights/best.pt}"
export PROJECT_ROOT DATA_DIR OUTPUT_DIR SEED DEVICE
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m cvio_asl_ldam.evaluation.robustness \
  --weights "$WEIGHTS" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --device "$DEVICE"
