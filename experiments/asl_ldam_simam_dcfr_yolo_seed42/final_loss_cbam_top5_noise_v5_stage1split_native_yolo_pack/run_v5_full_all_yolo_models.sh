#!/usr/bin/env bash
set -u
set +e
cd /home/drnguyenvinh/notebooks
MODELS="${MODELS:-yolov8n-cls yolov8s-cls yolov8m-cls yolo11n-cls yolo11s-cls yolo11m-cls yolo26n-cls yolo26s-cls yolo26m-cls}"
SEEDS="${SEEDS:-42 1 2}"
echo "=== FULL ALL YOLO MODELS INTERACTIVE ==="
echo "MODELS=$MODELS"
echo "SEEDS=$SEEDS"
echo "This is long. It runs sequentially with live logs, not background."
for M in $MODELS; do
  echo
  echo "################################################################################"
  echo "FULL MODEL: $M"
  echo "################################################################################"
  bash /home/drnguyenvinh/notebooks/run_v5_full_one_model.sh "$M" "$SEEDS"
  RC=$?
  if [ "$RC" != "0" ]; then
    echo "STOP: full run failed for $M with exit=$RC"
    exit "$RC"
  fi
  echo "PASS FULL: $M"
done
