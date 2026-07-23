#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT/.venv/bin/python}"
OUT="${OUT:-/home/drnguyenvinh/notebooks/recsra_mobile10_output_rtx4090}"

BASELINE_PT="${BASELINE_PT:-/home/drnguyenvinh/notebooks/shrimp_yolo11s_full_method_zoo/baseline_reuse_lock/yolo11s_baseline_best.pt}"
RECSRA_PT="${RECSRA_PT:-/home/drnguyenvinh/notebooks/shrimp_yolo11s_full_method_zoo/sldra54_ev015fair200/seed_42/runs_train/SD001_A_RCCM_EV015FAIR_200E_SEED42/weights/best.pt}"
DATASET_ROOT="${DATASET_ROOT:-/home/drnguyenvinh/notebooks/shrimp_yolo_canonical_bg_wssv_2cls_split70_15_15}"

[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$BASELINE_PT" ]] || { echo "[FATAL] Baseline checkpoint missing: $BASELINE_PT" >&2; exit 1; }
[[ -f "$RECSRA_PT" ]] || { echo "[FATAL] RECSRA checkpoint missing: $RECSRA_PT" >&2; exit 1; }
[[ -f "$DATASET_ROOT/data.yaml" ]] || { echo "[FATAL] Dataset data.yaml missing: $DATASET_ROOT/data.yaml" >&2; exit 1; }

export PYTHONPATH="$PROJECT/third_party/ultralytics:$PROJECT:$HERE:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

EXPECTED_BASELINE_SHA="b9e30aa76f819126c7e6e9f3d3007a74d0839ad1f95941978a6a214484eb219d"
EXPECTED_RECSRA_SHA="c1652101bb870a0b174b569ac47a70bb8cfafe119577e1ca2fe2de87f57824ff"
ACTUAL_BASELINE_SHA="$(sha256sum "$BASELINE_PT" | awk '{print $1}')"
ACTUAL_RECSRA_SHA="$(sha256sum "$RECSRA_PT" | awk '{print $1}')"
[[ "$ACTUAL_BASELINE_SHA" == "$EXPECTED_BASELINE_SHA" ]] || { echo "[FATAL] Baseline SHA mismatch" >&2; exit 1; }
[[ "$ACTUAL_RECSRA_SHA" == "$EXPECTED_RECSRA_SHA" ]] || { echo "[FATAL] RECSRA SHA mismatch" >&2; exit 1; }

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('[FATAL] CUDA is unavailable')
name = torch.cuda.get_device_name(0)
print('[GPU]', name)
if '4090' not in name:
    raise SystemExit(f'[FATAL] Expected RTX 4090 as visible GPU 0, found: {name}')
PY

"$PYTHON_BIN" "$HERE/patch_ultralytics.py"
"$PYTHON_BIN" -m py_compile \
  "$HERE/patch_ultralytics.py" \
  "$HERE/mobile_corruptions.py" \
  "$HERE/eval_worker.py" \
  "$HERE/benchmark.py" \
  "$HERE/sldra_block.py"

mkdir -p "$OUT"

"$PYTHON_BIN" "$HERE/benchmark.py" \
  --baseline "$BASELINE_PT" \
  --recsra "$RECSRA_PT" \
  --dataset "$DATASET_ROOT" \
  --output "$OUT"

printf '\n[FINISHED]\nOutput: %s\nZIP: %s\nSHA: %s\n' \
  "$OUT" \
  "$OUT/RECSRA_mobile10_RTX4090_final_results.zip" \
  "$OUT/RECSRA_mobile10_RTX4090_final_results.zip.sha256"
