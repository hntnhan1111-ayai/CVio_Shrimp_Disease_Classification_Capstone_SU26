#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs}"

mkdir -p \
  "$PROJECT_ROOT/artifacts/tables" \
  "$PROJECT_ROOT/artifacts/figures" \
  "$PROJECT_ROOT/artifacts/predictions" \
  "$PROJECT_ROOT/artifacts/confusion_matrices" \
  "$PROJECT_ROOT/artifacts/xai" \
  "$PROJECT_ROOT/artifacts/logs_sample"

while IFS= read -r -d '' file; do
  name="$(basename "$file")"
  case "$name" in
    *prediction*.csv) cp -f "$file" "$PROJECT_ROOT/artifacts/predictions/$name" ;;
    *confusion_matrix*.csv|*confusion_matrix*.png) cp -f "$file" "$PROJECT_ROOT/artifacts/confusion_matrices/$name" ;;
    *xai*.csv|*gradcam*.png) cp -f "$file" "$PROJECT_ROOT/artifacts/xai/$name" ;;
    *.csv|*.json) cp -f "$file" "$PROJECT_ROOT/artifacts/tables/$name" ;;
    *.png) cp -f "$file" "$PROJECT_ROOT/artifacts/figures/$name" ;;
  esac
done < <(find "$OUTPUT_DIR" -type f \
  ! -path '*/weights/*' \
  ! -name '*.pt' ! -name '*.pth' ! -name '*.onnx' \
  \( -name '*.csv' -o -name '*.json' -o -name '*.png' \) -print0)

echo "Collected small paper outputs under $PROJECT_ROOT/artifacts"
