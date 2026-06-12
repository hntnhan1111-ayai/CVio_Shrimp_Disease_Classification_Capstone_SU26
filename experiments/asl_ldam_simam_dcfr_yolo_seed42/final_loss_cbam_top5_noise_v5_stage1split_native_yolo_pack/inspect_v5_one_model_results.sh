#!/usr/bin/env bash
set +e
OUT="${1:-}"
LOG="${2:-}"
ZIP="${3:-}"
if [ -z "$OUT" ]; then echo "Usage: bash inspect_v5_one_model_results.sh OUT LOG ZIP"; exit 2; fi

echo
echo "=== ERRORS ==="
if [ -f "$LOG" ]; then
  grep -nEi "SyntaxError|unrecognized arguments|KeyError|missing_checkpoint|FAILED TRAIN|failed_train|Traceback|RuntimeError|FileNotFoundError|ValueError|AttributeError|CUDA out of memory|No space left" "$LOG" | tail -200 || true
else
  echo "missing log: $LOG"
fi

echo
echo "=== TRAIN LOG SIGNAL ==="
if [ -f "$LOG" ]; then
  grep -nE "TRAIN BASELINE|TRAIN CUSTOM VARIANT|PROJECT=|YOLO.*-cls summary|Starting training|top1_acc|Saved final YOLO metrics|FINAL COMPACT SUMMARY|ZIP=" "$LOG" | tail -260 || true
fi

echo
echo "=== COMPACT SUMMARY ==="
CSV="$OUT/final_reports/compact_summary_table.csv"
test -f "$CSV" && column -s, -t "$CSV" | head -120 || echo "missing $CSV"

echo
echo "=== ROBUSTNESS RANKING ==="
CSV2="$OUT/final_reports/robustness_ranked_summary.csv"
test -f "$CSV2" && column -s, -t "$CSV2" | head -120 || echo "missing $CSV2"

echo
echo "=== STAGE1 SPLIT VERIFY ==="
bash /home/drnguyenvinh/notebooks/verify_v5_stage1_split.sh "$OUT" || true

echo
echo "=== REVIEW ZIP ==="
ls -lh "$ZIP" 2>/dev/null || true

echo
echo "=== FINAL REPORT FILES ==="
find "$OUT/final_reports" -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" 2>/dev/null | sort || true
