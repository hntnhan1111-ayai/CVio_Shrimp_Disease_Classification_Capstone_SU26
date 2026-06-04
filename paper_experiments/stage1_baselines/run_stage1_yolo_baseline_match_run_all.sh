#!/usr/bin/env bash
set -euo pipefail
cd /home/drnguyenvinh/notebooks

# Optional but recommended: stop unrelated GPU jobs before running.
pgrep -af "qwen3_bm|nbconvert|pt_data_worker" || true
nvidia-smi || true

/opt/miniconda3/bin/python stage1_yolo_baseline_match_run_all.py \
  2>&1 | tee stage1_yolo_baseline_match_run_all_live.log

zip -r stage1_yolo_baseline_match_run_all_for_review.zip \
  stage1_yolo_baseline_match_run_all.py \
  stage1_yolo_baseline_match_run_all_live.log \
  stage1_yolo_baseline_match_runall_outputs \
  -x "*.pt" "*.pth" "*.png" "*.jpg" "*.jpeg" "*.webp" \
  -x "*/yolo_dataset/*" "*/weights/*" "*/ultralytics_train/*" "*/__pycache__/*"

echo "Upload: /home/drnguyenvinh/notebooks/stage1_yolo_baseline_match_run_all_for_review.zip"
