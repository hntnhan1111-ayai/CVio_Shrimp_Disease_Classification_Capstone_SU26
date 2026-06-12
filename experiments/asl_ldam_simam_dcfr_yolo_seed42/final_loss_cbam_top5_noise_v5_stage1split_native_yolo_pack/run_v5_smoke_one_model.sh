#!/usr/bin/env bash
set -u
set +e
cd /home/drnguyenvinh/notebooks

MODEL="${1:-yolo26m-cls}"
SEED="${2:-42}"
MODEL_SAFE="$(echo "$MODEL" | sed 's#[/ .]#_#g')"

ASL_PROJECT="${ASL_PROJECT:-/home/drnguyenvinh/notebooks/YOLO26M_CE_VS_ASL_LDAM_NOISY_TEST_EVAL_project}"
CE_PROJECT="${CE_PROJECT:-/home/drnguyenvinh/notebooks/YOLO26M_CE_TOP10_CUSTOM_LOSS_CBAM_WORK_project}"
DATASET="${DATASET:-/home/drnguyenvinh/notebooks/processed-images/processed_images}"
PATCH="${PATCH:-/home/drnguyenvinh/notebooks/patch_yolo_attention_injection_v2.py}"
RUNNER="${RUNNER:-/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_stage1split_native_yolo_runner_v5.py}"

OUT="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_SMOKE_${MODEL_SAFE}_seed${SEED}_outputs"
ZIP="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_SMOKE_${MODEL_SAFE}_seed${SEED}_RESULTS_ONLY.zip"
LOG="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_SMOKE_${MODEL_SAFE}_seed${SEED}_live.log"

rm -rf "$OUT" "$ZIP" "$LOG"

echo "=== V5 SMOKE ONE MODEL ==="
echo "MODEL=$MODEL"
echo "SEED=$SEED"
echo "OUT=$OUT"
echo "LOG=$LOG"
echo "ZIP=$ZIP"
echo "NOTE=This is a 1-epoch smoke only. Do not use metrics for paper ranking."

for P in "$RUNNER" "$PATCH" "$ASL_PROJECT/shrimp_scripts/models_yolo.py" "$CE_PROJECT/shrimp_scripts/models_yolo.py"; do
  test -e "$P" || { echo "MISSING: $P"; exit 2; }
done
test -d "$DATASET" || { echo "MISSING DATASET: $DATASET"; exit 3; }

/opt/miniconda3/bin/python -m py_compile "$RUNNER" || exit 4

if [ -f "/home/drnguyenvinh/notebooks/${MODEL}.pt" ]; then
  WEIGHTS="/home/drnguyenvinh/notebooks/${MODEL}.pt"
else
  WEIGHTS="${MODEL}.pt"
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
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
  --epochs 1 \
  --patience 1 \
  --batch 32 \
  --workers 8 \
  --seed "$SEED" \
  --imgsz 224 \
  --device 0 \
  --eval_batch 32 \
  --lr0 0.00125 \
  --corruptions "gaussian_noise" \
  --severities "1" \
  --max_examples_per_corruption 4 \
  --force \
  --force_corruptions_rebuild \
  --zip_path "$ZIP" \
  2>&1 | tee "$LOG"

RUN_EXIT=${PIPESTATUS[0]}
echo "SMOKE_EXIT=$RUN_EXIT"
echo "RESULT_ZIP=$ZIP"
echo
bash /home/drnguyenvinh/notebooks/inspect_v5_one_model_results.sh "$OUT" "$LOG" "$ZIP" || true
exit "$RUN_EXIT"
