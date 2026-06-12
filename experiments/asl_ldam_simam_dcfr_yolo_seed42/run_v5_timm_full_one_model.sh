#!/usr/bin/env bash
set -u
set +e
cd /home/drnguyenvinh/notebooks

MODEL="${1:-convnext_tiny}"
SEED="${2:-42}"
DATASET="${DATASET:-/home/drnguyenvinh/notebooks/processed-images/processed_images}"
RUNNER="${TIMM_RUNNER:-/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_stage1split_timm_runner_v5.py}"
OUT="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_${MODEL}_seed${SEED}_outputs"
LOG="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_${MODEL}_seed${SEED}_live.log"
ZIP="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_${MODEL}_seed${SEED}_RESULTS_ONLY_for_review.zip"

echo "=== V5 TIMM FULL ONE MODEL ==="
echo "MODEL=$MODEL"
echo "SEED=$SEED"
echo "DATASET=$DATASET"
echo "RUNNER=$RUNNER"
echo "OUT=$OUT"
echo "LOG=$LOG"
echo "ZIP=$ZIP"

test -f "$RUNNER" || { echo "MISSING runner: $RUNNER"; exit 2; }
test -d "$DATASET" || { echo "MISSING dataset: $DATASET"; exit 3; }
/opt/miniconda3/bin/python -m py_compile "$RUNNER" || exit 4

if [ "${RESET:-0}" = "1" ]; then
  rm -rf "$OUT" "$ZIP" "$LOG"
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTHONPATH=/home/drnguyenvinh/notebooks:$PYTHONPATH

/opt/miniconda3/bin/python -u "$RUNNER" \
  --model "$MODEL" \
  --dataset_root "$DATASET" \
  --output_dir "$OUT" \
  --seed "$SEED" \
  --imgsz 224 \
  --epochs 30 \
  --patience 5 \
  --batch 32 \
  --eval_batch 128 \
  --workers 0 \
  --paper_batch 128 \
  --device 0 \
  --amp \
  --corruptions "impulse_noise,gaussian_noise,contrast_reduction,defocus_blur,low_light" \
  --severities "1,2,3" \
  --zip_path "$ZIP" \
  2>&1 | tee "$LOG"

RUN_EXIT=${PIPESTATUS[0]}
echo "RUN_EXIT=$RUN_EXIT"
echo "ZIP=$ZIP"

echo "=== Error search ==="
grep -nEi "KeyError|FAILED|failed|Traceback|RuntimeError|ValueError|CUDA out of memory|No space left|nan|nonfinite" "$LOG" | tail -120 || true
exit "$RUN_EXIT"
