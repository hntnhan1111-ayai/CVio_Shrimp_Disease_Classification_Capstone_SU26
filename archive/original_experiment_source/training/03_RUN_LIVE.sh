#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/drnguyenvinh/notebooks/SLDRA54_EV015Fair200
source /home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/.venv/bin/activate
export PYTHONPATH="/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/third_party/ultralytics:/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo:$PWD:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
OUT=/home/drnguyenvinh/notebooks/shrimp_yolo11s_full_method_zoo/sldra54_ev015fair200
mkdir -p "$OUT/logs"
python -u 04_run_all.py --start-id "${START_ID:-0}" --end-id "${END_ID:-53}" --epochs "${EPOCHS:-200}" --seed "${SEED:-42}" 2>&1 | tee -a "$OUT/logs/sldra54_seed${SEED:-42}.log"
