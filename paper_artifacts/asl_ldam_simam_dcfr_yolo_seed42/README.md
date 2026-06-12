# Seed-42 ASL-LDAM SimAM-DCFR Paper Artifacts

This folder contains the seed-42 artifact package for the paper:

**ASL-LDAM with SimAM-DCFR Attention for Robust YOLO-Based Shrimp Disease Image Classification Under Noisy Imaging Conditions**

Short title: **Robust YOLO-Based Shrimp Disease Classification**

## Scope

- Fixed seed-42 Stage-1 split only.
- No mean ± std across multiple seeds.
- No model weights/checkpoints.
- No raw dataset or corrupted image dataset folders.
- Intended for paper writing, artifact review, and reproducibility code organization.

## Contents

### Tables

- `tables/dataset/dataset_split_distribution_table_seed42.csv` — dataset split summary
- `tables/baselines/yolo_baseline_comparison_seed42.csv` — YOLO native baseline ranking
- `tables/baselines/timm_baseline_comparison_seed42.csv` — TIMM baseline ranking
- `tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv` — key YOLO26m result
- `tables/improvements/yolo_best_method_vs_baseline_seed42.csv` — YOLO improvement deltas
- `tables/improvements/timm_best_method_vs_baseline_seed42.csv` — TIMM improvement deltas
- `tables/noise/top5_noise_selection_from_ce_baseline.csv` — top-5 noise selection rationale
- `tables/noise/top5_noise_baseline_vs_best_mean_summary.csv` — best-method noise summary
- `tables/classwise/best_method_classwise_metrics_seed42.csv` — class-wise precision/recall/F1
- `tables/efficiency/model_params_size_latency_fps_seed42.csv` — model size and efficiency
- `tables/config/kaggle_reference_configuration_table.csv` — training and environment config

### Figures

- `figures/samples/sample_no_bg_4class_panel.png` — sample panel (Healthy, BG, WSSV, WSSV_BG)
- `figures/baselines/yolo_baseline_macro_f1_seed42.png` — YOLO baseline Macro-F1
- `figures/baselines/timm_baseline_macro_f1_seed42.png` — TIMM baseline Macro-F1
- `figures/improvements/yolo26m_key_result_macro_f1_seed42.png` — YOLO26m key result
- `figures/improvements/yolo_delta_macro_f1_seed42.png` — YOLO Macro-F1 deltas
- `figures/improvements/timm_delta_macro_f1_seed42.png` — TIMM Macro-F1 deltas
- `figures/noise/top5_noise_baseline_vs_best_mean_macro_f1.png` — baseline vs best across top-5 noise
- `figures/noise/best_method_top5_noise_macro_f1_by_severity.png` — best method noise by severity
- `figures/training_curves/best_method_training_curves_top1_only.png` — training curves for best method
- `figures/confusion_matrices/best_method_clean_confusion_matrix.png` — clean test confusion matrix
- `figures/confusion_matrices/baseline_ce_clean_confusion_matrix.png` — baseline CE clean confusion matrix
- `figures/confusion_matrices/best_method_impulse_noise_s3_confusion_matrix.png` — noise example at severity 3
- `figures/xai/xai_best_method_4class_grid.png` — XAI visualization grid

### Code

- `experiments/asl_ldam_simam_dcfr_yolo_seed42/` — runner scripts, TIMM runner, result collectors

### Metadata

- `PACKAGE_VALIDATION.json` — artifact package checksum/integrity
- `PACKAGE_MANIFEST.csv` — manifest of included files
- `paper_ready_summary/` — additional documentation and corrections

## Key Result (YOLO26m-cls)

| Metric | Baseline | Best (ASL-LDAM + SimAM-DCFR) | Delta |
|---|---|---|---|
| Macro-F1 | 0.890200 | 0.910137 | +0.019937 |
| Accuracy | 0.890200 | 0.913295 | +0.023095 |
| Cohen's Kappa | 0.850500 | 0.881593 | +0.031093 |

Source: `tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv`

## Top-5 Noise Robustness

| Noise | Baseline Mean Macro-F1 | Best Mean Macro-F1 | Delta |
|---|---|---|---|
| impulse_noise | 0.4850 | 0.5582 | +0.0732 |
| gaussian_noise | 0.7479 | 0.7850 | +0.0371 |
| contrast_reduction | 0.8207 | 0.8810 | +0.0603 |
| defocus_blur | 0.8300 | 0.8866 | +0.0567 |
| low_light | 0.8315 | 0.9022 | +0.0707 |

Source: `tables/noise/top5_noise_baseline_vs_best_mean_summary.csv`

## Usage Notes

- Link tables and figures in papers using relative paths from this folder.
- Do not present results as mean ± std over multiple seeds.
- Do not claim statistical significance across seeds.
- The fixed seed-42 Stage-1 split should be referenced explicitly.
