#!/usr/bin/env bash
set -u
set +e
cd /home/drnguyenvinh/notebooks
MODELS="${MODELS:-yolov8n-cls yolov8s-cls yolov8m-cls yolo11n-cls yolo11s-cls yolo11m-cls yolo26n-cls yolo26s-cls yolo26m-cls}"
SEED="${SEED:-42}"
echo "=== SMOKE ALL YOLO MODELS INTERACTIVE ==="
echo "MODELS=$MODELS"
echo "SEED=$SEED"
echo "Each model prints live Ultralytics logs in terminal via tee. No background run."
for M in $MODELS; do
  echo
  echo "################################################################################"
  echo "SMOKE MODEL: $M"
  echo "################################################################################"
  bash /home/drnguyenvinh/notebooks/run_v5_smoke_one_model.sh "$M" "$SEED"
  RC=$?
  if [ "$RC" != "0" ]; then
    echo "STOP: smoke failed for $M with exit=$RC"
    exit "$RC"
  fi
  echo "PASS SMOKE: $M"
done
