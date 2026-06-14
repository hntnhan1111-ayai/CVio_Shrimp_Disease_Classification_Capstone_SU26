# Paper Writing Brief

## Recommended Title
ASL-LDAM with SimAM-DCFR Attention for Robust YOLO-Based Shrimp Disease Image Classification Under Noisy Imaging Conditions
(Robust YOLO-Based Shrimp Disease Classification)

## Scope
Fixed seed-42 Stage-1 split; 4 classes: Healthy, BG, WSSV, WSSV_BG. Results reflect one split only. No raw dataset or model weights included.

## Dataset and Split
Train 804 / val 172 / test 173 stratified.

## Experimental Setup
Kaggle T4x2, Python 3.12.3, 30 epochs, image size 224. YOLO native Ultralytics classification baseline and TIMM baselines. Top-5 noisy evaluations using severities 1-3.

## Main Method
YOLO26m-cls + ASL-LDAM + SimAM-DCFR (top-1 variant).

## Main Corrected YOLO26m Result (from tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv)
| Metric | Baseline | Best | Delta |
|---|---|---|---|
| Macro-F1 | 0.890200 | 0.910137 | +0.019937 |
| Accuracy | 0.890200 | 0.913295 | +0.023095 |
| Cohen's Kappa | 0.850500 | 0.881593 | +0.031093 |

## YOLO Baseline Comparison
Use tables/baselines/yolo_baseline_comparison_seed42.csv; 15 models sorted by Macro-F1 descending, top model yolo26m-cls 0.8902.

## TIMM Baseline Comparison
Use tables/baselines/timm_baseline_comparison_seed42.csv; top model convnext_tiny_in22k Macro-F1 0.8416, test_accuracy 0.8522, test_macro_f1 0.8416, fps 49.2, latency_ms 20.34.

## Improvement vs Baseline
Use tables/improvements/yolo_best_method_vs_baseline_seed42.csv and timm_best_method_vs_baseline_seed42.csv.

## Top-5 Noise Robustness
Use tables/noise/top5_noise_baseline_vs_best_mean_summary.csv.

## Classwise Performance
Use tables/classwise/best_method_classwise_metrics_seed42.csv.

## XAI Evidence
Use xai panels for Healthy, BG, WSSV, WSSV_BG.

## Efficiency Evidence
Use tables/efficiency/model_efficiency_summary_seed42.csv.

## Recommended Main Paper Figures
sample_no_bg_4class_panel.png, yolo_baseline_macro_f1_seed42.png, yolo26m_key_result_macro_f1_seed42.png, top5_noise_baseline_vs_best_mean_macro_f1.png, best_method_training_curves_top1_only.png, xai_best_method_4class_grid.png.

## Recommended Main Paper Tables
dataset_split_distribution_table_seed42.csv, yolo_baseline_comparison_seed42.csv, timm_baseline_comparison_seed42.csv, paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv, top5_noise_baseline_vs_best_mean_summary.csv, best_method_classwise_metrics_seed42.csv, model_efficiency_summary_seed42.csv.

## Safe Claims
On the fixed seed-42 Stage-1 split, YOLO26m-cls with ASL-LDAM and SimAM-DCFR improved over the CE baseline in Macro-F1, accuracy, and Cohen's Kappa. Top-5 noisy conditions were selected from the weakest CE baseline corruptions. XAI shows class-discriminative regions for all four classes.

## Claims to Avoid
statistically significant, state-of-the-art, robust across random seeds, globally best across all architectures, mean ± std over multiple seeds, mobile deployment validated.

## Notes for Related Work Search
Synthesize general background on YOLO classification heads, class-imbalanced loss (LDAM, ASL), and lightweight attention (SimAM/CBAM). Do not cite this single-seed result as a broad empirical law.
