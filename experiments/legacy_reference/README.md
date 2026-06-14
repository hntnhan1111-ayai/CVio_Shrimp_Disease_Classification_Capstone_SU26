# Legacy Experiment Runners

These two runners are retained as source provenance for the original experiment
implementation:

- `final_loss_cbam_top5_noise_stage1split_native_yolo_runner_v5.py`
- `final_loss_cbam_top5_noise_stage1split_timm_runner_v5.py`

They are not the supported reproduction entry points. Use the portable commands
under `scripts/` and the package under `src/cvio_asl_ldam/` for the seed-42 paper
workflow.

The native-YOLO legacy runner expects historical project copies and a patch
script that are not distributed in this repository. Its path defaults now use
`PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`, `ASL_PROJECT_DIR`, and
`CE_PROJECT_DIR` instead of private workstation paths.

Retention of these files does not extend the paper claims beyond the fixed
seed-42 Stage-1 split. No multi-seed mean plus/minus standard deviation or
statistical-significance claim is supported.
