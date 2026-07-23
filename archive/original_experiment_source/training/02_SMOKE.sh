#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/drnguyenvinh/notebooks/SLDRA54_EV015Fair200
source /home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/.venv/bin/activate
export PYTHONPATH="/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/third_party/ultralytics:/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo:$PWD:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
python -u 04_run_all.py --start-id 0 --end-id 2 --epochs 1 --out /home/drnguyenvinh/notebooks/shrimp_yolo11s_full_method_zoo/sldra54_smoke
