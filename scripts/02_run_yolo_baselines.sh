#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/datasets/processed-images}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
export PROJECT_ROOT DATA_DIR OUTPUT_DIR SEED DEVICE
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${MODEL:-}" ]]; then
  MODELS=("$MODEL")
else
  MODELS=(
    yolov8n-cls yolov8s-cls yolov8m-cls yolov8l-cls yolov8x-cls
    yolo11n-cls yolo11s-cls yolo11m-cls yolo11l-cls yolo11x-cls
    yolo26n-cls yolo26s-cls yolo26m-cls yolo26l-cls yolo26x-cls
  )
fi

for model in "${MODELS[@]}"; do
  python -m cvio_asl_ldam.models.yolo_train \
    --config configs/train/yolo26m_ce.yaml \
    --model "$model" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"
done
