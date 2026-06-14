# Artifact Index

The paths below are the curated, committed seed-42 paper artifacts.

## Manifests and Logs

- `artifacts/manifests/split_manifest_seed42.csv`
- `artifacts/logs_sample/environment_report.json`
- `artifacts/logs_sample/README.md`

## Baseline Tables

- `artifacts/tables/baselines/coverage_timm_seed42.csv`
- `artifacts/tables/baselines/coverage_yolo_seed42.csv`
- `artifacts/tables/baselines/timm_baseline_comparison_seed42.csv`
- `artifacts/tables/baselines/yolo_baseline_comparison_seed42.csv`

## Main and Class-Wise Tables

- `artifacts/tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv`
- `artifacts/tables/improvements/timm_best_method_vs_baseline_seed42.csv`
- `artifacts/tables/improvements/yolo_best_method_vs_baseline_seed42.csv`
- `artifacts/tables/classwise/best_method_classwise_metrics_seed42.csv`
- `artifacts/tables/classwise/best_method_classwise_overall_metrics_seed42.json`

## Dataset and Configuration Tables

- `artifacts/tables/config/kaggle_reference_configuration_table.csv`
- `artifacts/tables/dataset/dataset_class_distribution_total_seed42.csv`
- `artifacts/tables/dataset/dataset_split_distribution_long_seed42.csv`
- `artifacts/tables/dataset/dataset_split_distribution_table_seed42.csv`
- `artifacts/tables/sample_images/sample_image_selection.csv`

## Noise and Efficiency Tables

- `artifacts/tables/noise/best_method_top5_noise_by_severity.csv`
- `artifacts/tables/noise/best_method_top5_noise_mean_summary.csv`
- `artifacts/tables/noise/top5_noise_baseline_vs_best_mean_summary.csv`
- `artifacts/tables/noise/top5_noise_selection_from_ce_baseline.csv`
- `artifacts/tables/efficiency/model_efficiency_summary_seed42.csv`
- `artifacts/tables/efficiency/model_params_size_latency_fps_seed42.csv`

## Confusion Matrices

- `artifacts/confusion_matrices/baseline_ce_clean_confusion_matrix.csv`
- `artifacts/confusion_matrices/best_method_clean_confusion_matrix.csv`
- `artifacts/confusion_matrices/best_method_impulse_noise_s3_confusion_matrix.csv`
- PNG renderings under `artifacts/figures/confusion_matrices/`

## Figures

- `artifacts/figures/samples/sample_no_bg_4class_panel.png`
- `artifacts/figures/baselines/timm_baseline_macro_f1_seed42.png`
- `artifacts/figures/baselines/yolo_baseline_macro_f1_seed42.png`
- `artifacts/figures/improvements/timm_delta_macro_f1_seed42.png`
- `artifacts/figures/improvements/yolo_delta_macro_f1_seed42.png`
- `artifacts/figures/improvements/yolo26m_key_result_macro_f1_seed42.png`
- `artifacts/figures/noise/best_method_top5_noise_macro_f1_by_severity.png`
- `artifacts/figures/noise/top5_noise_baseline_vs_best_mean_macro_f1.png`
- `artifacts/figures/training_curves/best_method_training_curves_top1_only.png`
- `artifacts/figures/xai/xai_best_method_4class_grid.png`
- `artifacts/figures/source_evidence/timm_baseline_previous_seed42_table_screenshot.png`

## XAI

- Four per-class panels under `artifacts/xai/panels/`
- `artifacts/xai/metadata/README_XAI_BEST_ONLY.md`
- `artifacts/xai/metadata/xai_summary.csv`
- `artifacts/xai/metadata/xai_summary.json`

## Compatibility Note

These files were migrated from the historical
`paper_artifacts/asl_ldam_simam_dcfr_yolo_seed42/` and
`paper_writing_bundle/asl_ldam_simam_dcfr_yolo_seed42/` layouts. Git history
retains the original paths. Large data, weights, checkpoints, and archives are
not part of the curated layout.
