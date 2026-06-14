# Results

All reported results use the fixed seed-42 Stage-1 split. They are not
multi-seed averages.

## Main YOLO26m Result

| Metric | Baseline CE | ASL-LDAM + SimAM-DCFR | Delta |
|---|---:|---:|---:|
| Macro-F1 | 0.890200 | 0.910137 | +0.019937 |
| Accuracy | 0.890200 | 0.913295 | +0.023095 |
| Cohen's Kappa | 0.850500 | 0.881593 | +0.031093 |

Source:
[`paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv`](../artifacts/tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv)

## Class-Wise Clean Test

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 1.000 | 0.869 | 0.930 | 61 |
| BG | 0.900 | 0.931 | 0.915 | 29 |
| WSSV | 0.900 | 0.900 | 0.900 | 50 |
| WSSV_BG | 0.800 | 0.970 | 0.877 | 33 |

Source:
[`best_method_classwise_metrics_seed42.csv`](../artifacts/tables/classwise/best_method_classwise_metrics_seed42.csv)

## YOLO Baselines

The full 15-model comparison is in
[`yolo_baseline_comparison_seed42.csv`](../artifacts/tables/baselines/yolo_baseline_comparison_seed42.csv).
YOLO26m-cls ranked first with Macro-F1 0.8902.

## TIMM Baselines

| Model | Macro-F1 | Accuracy | Kappa |
|---|---:|---:|---:|
| convnext_tiny_in22k | 0.8416 | 0.8522 | 0.7690 |
| mobilenet_v3_large | 0.8010 | 0.8261 | 0.6114 |
| efficientnet_b0 | 0.7701 | 0.7913 | 0.5940 |
| repvgg_a0 | 0.7688 | 0.7739 | 0.5514 |
| efficientnet_v2_s | 0.7678 | 0.7739 | 0.6627 |

Source:
[`timm_baseline_comparison_seed42.csv`](../artifacts/tables/baselines/timm_baseline_comparison_seed42.csv)

## Top-5 Noise Robustness

| Noise | Baseline Mean Macro-F1 | Best Mean Macro-F1 | Delta |
|---|---:|---:|---:|
| impulse_noise | 0.4850 | 0.5582 | +0.0732 |
| gaussian_noise | 0.7479 | 0.7850 | +0.0371 |
| contrast_reduction | 0.8207 | 0.8810 | +0.0603 |
| defocus_blur | 0.8300 | 0.8866 | +0.0567 |
| low_light | 0.8315 | 0.9022 | +0.0707 |

Sources:

- [`top5_noise_selection_from_ce_baseline.csv`](../artifacts/tables/noise/top5_noise_selection_from_ce_baseline.csv)
- [`top5_noise_baseline_vs_best_mean_summary.csv`](../artifacts/tables/noise/top5_noise_baseline_vs_best_mean_summary.csv)
- [`best_method_top5_noise_by_severity.csv`](../artifacts/tables/noise/best_method_top5_noise_by_severity.csv)

## Efficiency and Confusion Matrices

- [`model_efficiency_summary_seed42.csv`](../artifacts/tables/efficiency/model_efficiency_summary_seed42.csv)
- [`baseline_ce_clean_confusion_matrix.csv`](../artifacts/confusion_matrices/baseline_ce_clean_confusion_matrix.csv)
- [`best_method_clean_confusion_matrix.csv`](../artifacts/confusion_matrices/best_method_clean_confusion_matrix.csv)
- [`best_method_impulse_noise_s3_confusion_matrix.csv`](../artifacts/confusion_matrices/best_method_impulse_noise_s3_confusion_matrix.csv)
