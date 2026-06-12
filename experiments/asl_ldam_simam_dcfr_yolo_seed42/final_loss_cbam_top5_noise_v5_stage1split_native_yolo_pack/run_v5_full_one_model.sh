#!/usr/bin/env bash
set -u
set +e
cd /home/drnguyenvinh/notebooks

MODEL="${1:-yolo26m-cls}"
SEEDS="${2:-42 1 2}"
MODEL_SAFE="$(echo "$MODEL" | sed 's#[/ .]#_#g')"

ASL_PROJECT="${ASL_PROJECT:-/home/drnguyenvinh/notebooks/YOLO26M_CE_VS_ASL_LDAM_NOISY_TEST_EVAL_project}"
CE_PROJECT="${CE_PROJECT:-/home/drnguyenvinh/notebooks/YOLO26M_CE_TOP10_CUSTOM_LOSS_CBAM_WORK_project}"
DATASET="${DATASET:-/home/drnguyenvinh/notebooks/processed-images/processed_images}"
PATCH="${PATCH:-/home/drnguyenvinh/notebooks/patch_yolo_attention_injection_v2.py}"
RUNNER="${RUNNER:-/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_stage1split_native_yolo_runner_v5.py}"

/opt/miniconda3/bin/python -m py_compile "$RUNNER" || exit 4

echo "=== V5 FULL ONE MODEL ==="
echo "MODEL=$MODEL"
echo "SEEDS=$SEEDS"
echo "RESET=${RESET:-0} (set RESET=1 to delete previous full outputs for this model/seed)"
echo "Native YOLO requirement: CE baseline uses YOLO(...).train(); custom variants call project train_yolo_with_fallback(), preserving native Ultralytics classification training path."

for SEED in $SEEDS; do
  OUT="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_${MODEL_SAFE}_seed${SEED}_outputs"
  ZIP="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_${MODEL_SAFE}_seed${SEED}_RESULTS_ONLY_for_review.zip"
  LOG="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_${MODEL_SAFE}_seed${SEED}_live.log"
  if [ "${RESET:-0}" = "1" ]; then
    rm -rf "$OUT" "$ZIP" "$LOG"
  fi
  mkdir -p "$OUT"

  if [ -f "/home/drnguyenvinh/notebooks/${MODEL}.pt" ]; then
    WEIGHTS="/home/drnguyenvinh/notebooks/${MODEL}.pt"
  else
    WEIGHTS="${MODEL}.pt"
  fi

  echo
  echo "===================================================================================================="
  echo "FULL RUN MODEL=$MODEL SEED=$SEED"
  echo "OUT=$OUT"
  echo "LOG=$LOG"
  echo "ZIP=$ZIP"
  echo "===================================================================================================="

  export PYTHONUNBUFFERED=1
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  export TORCH_CUDNN_V8_API_ENABLED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

  /opt/miniconda3/bin/python -u "$RUNNER" \
    --model_name "$MODEL" \
    --yolo_weights "$WEIGHTS" \
    --project_dir "$ASL_PROJECT" \
    --asl_project_dir "$ASL_PROJECT" \
    --ce_project_dir "$CE_PROJECT" \
    --dataset_root "$DATASET" \
    --output_dir "$OUT" \
    --patch_script "$PATCH" \
    --epochs 30 \
    --patience 15 \
    --batch 32 \
    --workers 8 \
    --seed "$SEED" \
    --imgsz 224 \
    --device 0 \
    --eval_batch 32 \
    --lr0 0.00125 \
    --corruptions "impulse_noise,gaussian_noise,contrast_reduction,defocus_blur,low_light" \
    --severities "1,2,3" \
    --max_examples_per_corruption 8 \
    --zip_path "$ZIP" \
    2>&1 | tee "$LOG"

  RUN_EXIT=${PIPESTATUS[0]}
  echo "RUN_EXIT=$RUN_EXIT"
  echo "RESULT_ZIP=$ZIP"
  bash /home/drnguyenvinh/notebooks/inspect_v5_one_model_results.sh "$OUT" "$LOG" "$ZIP" || true
  if [ "$RUN_EXIT" != "0" ]; then
    echo "STOP: model=$MODEL seed=$SEED failed. Fix this before continuing."
    exit "$RUN_EXIT"
  fi
done
