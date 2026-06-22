#!/usr/bin/env bash
# Lightweight smoke tests that do not require the full dataset or GPU.
set -euo pipefail
export PYTHONPATH="D:\ASL-LDAM A Class-Imbalance-Aware Loss for Robust Shrimp Disease Image Classification Under Noisy Imaging Conditions/src"

echo "[1/3] Env report"
python scripts/00_check_env.py

echo "[2/3] pytest"
python -m pytest tests -q

echo "[3/3] Verify export script does not train"
if grep -q '\.train(' scripts/06_export_litert_fp32_fp16.py; then
  echo "FAIL: export script contains .train(" >&2
  exit 1
fi
echo "Smoke tests passed."
