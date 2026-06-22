#!/usr/bin/env bash
# Export-only workflow. NEVER trains.
set -euo pipefail
export PYTHONPATH="D:\ASL-LDAM A Class-Imbalance-Aware Loss for Robust Shrimp Disease Image Classification Under Noisy Imaging Conditions/src"

WEIGHTS=""
if [ ! -f "" ]; then
  echo "ERROR: weights not found at " >&2
  echo "Provide trained best.pt via WEIGHTS env var or train first." >&2
  exit 1
fi
python scripts/06_export_litert_fp32_fp16.py --weights "" --out-dir export --imgsz 224 --skip-if-present
echo "Export done. See export/README.md."
