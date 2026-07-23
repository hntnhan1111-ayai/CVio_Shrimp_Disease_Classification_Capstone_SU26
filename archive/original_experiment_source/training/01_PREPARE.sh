#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/drnguyenvinh/notebooks/SLDRA54_EV015Fair200
source /home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/.venv/bin/activate
export PYTHONPATH="/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/third_party/ultralytics:/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo:$PWD:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
python 01_patch_ultralytics.py
rm -rf configs
python 02_generate_configs.py
python -m py_compile *.py
python 06_preflight.py
