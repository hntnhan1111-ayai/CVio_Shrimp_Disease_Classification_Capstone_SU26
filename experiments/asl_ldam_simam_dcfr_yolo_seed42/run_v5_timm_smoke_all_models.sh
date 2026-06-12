#!/usr/bin/env bash
set -u
set +e
cd /home/drnguyenvinh/notebooks
MODELS=${MODELS:-"mobilenetv3_large_100 efficientnet_b0 tf_efficientnetv2_s shufflenet_v2_x1_0 ghostnetv2_100 mobilevit_s efficientvit_m1 fastvit_t8 convnext_tiny"}
SEED=${SEED:-42}
for M in $MODELS; do
  echo
  echo "===================================================================================================="
  echo "TIMM SMOKE MODEL: $M seed$SEED"
  echo "===================================================================================================="
  bash /home/drnguyenvinh/notebooks/run_v5_timm_smoke_one_model.sh "$M" "$SEED"
  OUT="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_SMOKE_${M}_seed${SEED}_outputs"
  LOG="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_SMOKE_${M}_seed${SEED}_live.log"
  ZIP="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_SMOKE_${M}_seed${SEED}_RESULTS_ONLY.zip"
  bash /home/drnguyenvinh/notebooks/inspect_v5_timm_one_model_results.sh "$OUT" "$LOG" "$ZIP"
  read -p "Continue to next TIMM smoke model? Press Enter to continue, Ctrl+C to stop. " _
done
