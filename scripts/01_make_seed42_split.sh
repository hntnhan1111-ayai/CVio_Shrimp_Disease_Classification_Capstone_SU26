#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/datasets/processed-images}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
export PROJECT_ROOT DATA_DIR OUTPUT_DIR SEED DEVICE
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m cvio_asl_ldam.data.make_split \
  --config configs/dataset/shrimpdiseasebd_seed42.yaml \
  --data-dir "$DATA_DIR" \
  --output-dir artifacts/manifests \
  --seed "$SEED"
