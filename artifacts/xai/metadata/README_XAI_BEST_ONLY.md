# XAI best-method-only

Model: `yolo26m-cls`

Seed: `42`

Variant: `asl_ldam_simam_dcfr_top1`

Checkpoint: `${PROJECT_ROOT}/final_loss_cbam_top5_noise_v5_stage1split_yolo26m-cls_seed42_outputs/runs/ultralytics_train/ultralytics_yolo26m_cls_asl_ldam_simam_dcfr_top1_seed42_repeat1/weights/best.pt`

Checkpoint source: `target_method_checkpoint`

Run dir: `${PROJECT_ROOT}/final_loss_cbam_top5_noise_v5_stage1split_yolo26m-cls_seed42_outputs/evaluations/asl_ldam_simam_dcfr_top1`

Generated one representative image per class.

Important: if checkpoint_source is fallback_checkpoint_no_target_method_weights_found, the result folders did not contain a separate target-method checkpoint. The XAI image is generated using the available fallback checkpoint and should be reported carefully.
