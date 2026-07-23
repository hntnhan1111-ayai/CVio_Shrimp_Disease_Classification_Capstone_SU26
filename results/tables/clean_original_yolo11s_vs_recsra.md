# Clean test comparison — Original YOLO11s baseline vs YOLO11s-RECSRA

| Metric | YOLO11s baseline | YOLO11s-RECSRA | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| mAP50 | 12.900% | 16.036% | +3.136 pp | +24.31% |
| mAP50_95 | 3.800% | 4.657% | +0.857 pp | +22.54% |
| precision | 22.400% | 24.342% | +1.942 pp | +8.67% |
| recall | 21.800% | 24.695% | +2.895 pp | +13.28% |
| AP_BG_mAP50_95 | 4.000% | 5.373% | +1.373 pp | +34.33% |
| AP_WSSV_mAP50_95 | 3.500% | 3.940% | +0.440 pp | +12.56% |

> **Baseline context:** original_yolo11s_14_model_benchmark.
> **RECSRA experiment ID:** SD001_A_RCCM_EV015FAIR_200E_SEED42.
> Raw decimal metrics are preserved in `results/raw/clean_original_baseline_vs_recsra.json`.
