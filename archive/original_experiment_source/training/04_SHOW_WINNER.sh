#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/drnguyenvinh/notebooks/SLDRA54_EV015Fair200
source /home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/.venv/bin/activate
python 05_show_winner.py
