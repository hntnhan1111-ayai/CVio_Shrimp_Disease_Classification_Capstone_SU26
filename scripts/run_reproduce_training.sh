#!/usr/bin/env bash
# Reproduce the full training workflow (split -> train main -> train baseline -> eval).
set -euo pipefail
export PYTHONPATH="D:\ASL-LDAM A Class-Imbalance-Aware Loss for Robust Shrimp Disease Image Classification Under Noisy Imaging Conditions/src"

echo "[1/5] Environment check"
python scripts/00_check_env.py

echo "[2/5] Prepare seed-42 split"
python scripts/01_prepare_dataset_and_split.py --seed 42 --skip-if-complete

echo "[3/5] Train main method (ASL-LDAM + SimAM-DCFR)"
python scripts/03_train_yolo26m_asl_ldam_simam_dcfr.py --device "" --skip-if-complete

echo "[4/5] (Optional) Train CE baseline"
python scripts/02_train_yolo26m_ce_baseline.py --device "" --skip-if-complete

echo "[5/5] Evaluate clean test set"
WEIGHTS="runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt"
python scripts/04_eval_clean.py --weights ""
echo "Done."
