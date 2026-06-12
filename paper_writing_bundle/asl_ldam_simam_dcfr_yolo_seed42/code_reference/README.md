# final_loss_cbam_top5_noise_v5_stage1split_native_yolo_pack

Purpose: run native-Ultralytics YOLO classification experiments using the ShrimpXNet/Stage-1-style fixed random stratified image-level 70/15/15 split (seed 42 style, 804/172/173), then evaluate clean test and top-5 noisy test corruptions.

## Methods per YOLO model

- `baseline_ce_native_ultralytics`: native `YOLO(...).train(...)` classification CE baseline.
- `asl_ldam_margin`: custom loss through the ASL project native YOLO fallback training path.
- `asl_ldam_margin_cbam`: ASL-LDAM + CBAM through native YOLO fallback training path.
- `asl_ldam_simam_dcfr_top1`: ASL-LDAM + `attention_key=simam_gated_residual__dcfr_texture`.
- `ce_effective_num_weighted`: CE effective-number weighting through the CE project native YOLO fallback training path.
- `ce_effective_num_weighted_cbam`: CE effective-number weighting + CBAM.

The runner preserves native Ultralytics classification training for YOLO. It does not use a pure custom PyTorch YOLO loop.

## Top-5 noise protocol

Full runs use fixed top-5 corruptions selected from CE baseline weakness by mean Macro-F1 across severity 1-3:

1. impulse_noise
2. gaussian_noise
3. contrast_reduction
4. defocus_blur
5. low_light

Smoke runs use only `gaussian_noise` severity 1 for fast validation.

## Main commands

1. Unzip pack into `/home/drnguyenvinh/notebooks`.
2. Smoke each model first using `run_v5_smoke_one_model.sh` or `run_v5_smoke_all_yolo_models.sh`.
3. Inspect results.
4. Run full only after smoke passes.

No scripts run in the background. All scripts stream live logs to terminal with `tee`.
