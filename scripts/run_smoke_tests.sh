#!/usr/bin/env bash
# Lightweight smoke tests that do not require the full dataset or GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

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
