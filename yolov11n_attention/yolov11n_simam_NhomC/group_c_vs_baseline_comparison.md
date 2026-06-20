# Group C vs Clean YOLO11n-seg Baseline

Baseline source: `yolov11n_attention/yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb`

Group C sources:
- `bilinear_refine/bilinear_refine_summary.csv`
- `ca_dual/ca_dual_summary.csv`
- `simam_backbone_ca_head/simam_backbone_ca_head_summary.csv`
- `lka_head/lka_head_summary.csv`

Notes:
- Baseline full/labeled mask mAP50, miss rate, healthy FP rate, and healthy-aware score are read from notebook outputs.
- Baseline mAP50-95 values are rounded because the linked notebook exposes them in the Ultralytics log at three decimals.
- Baseline training time is derived from the notebook log: `60 epochs completed in 0.286 hours` = `17.16` minutes.

| Rank | Model | Train min | Epochs | Full mask mAP50 | Delta vs baseline | Full mask mAP50-95 | Labeled mask mAP50 | Delta vs baseline | Labeled mask mAP50-95 | Disease miss rate | Healthy FP rate | Healthy-aware labeled mAP50 | Delta vs baseline | Quick read |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `lka_head` | 52.81 | 75 | 0.5003 | +0.0593 | 0.1732 | 0.5249 | +0.0585 | 0.1812 | 7.95% | 36.59% | 0.4518 | +0.0584 | Best overall; strongest mask quality while keeping miss rate low. |
| 2 | `clean_light_aug_baseline` | 17.16 | 60 | 0.4409 | +0.0000 | ~0.1610 | 0.4664 | +0.0000 | ~0.1690 | 12.50% | 36.59% | 0.3934 | +0.0000 | Strong, fast baseline; only LKA clearly beats it. |
| 3 | `bilinear_refine` | 79.14 | 100 | 0.3881 | -0.0529 | 0.1070 | 0.4153 | -0.0512 | 0.1145 | 6.82% | 41.46% | 0.3399 | -0.0535 | Lowest miss rate, but over-predicts and trails baseline mask quality. |
| 4 | `simam_backbone_ca_head` | 54.41 | 76 | 0.3848 | -0.0561 | 0.1213 | 0.4185 | -0.0480 | 0.1328 | 11.36% | 29.27% | 0.3480 | -0.0454 | Better FP control than baseline/LKA, but lower mask mAP. |
| 5 | `ca_dual` | 66.29 | 100 | 0.3647 | -0.0762 | 0.1190 | 0.3824 | -0.0840 | 0.1241 | 21.59% | 29.27% | 0.3010 | -0.0924 | Conservative; low FP but misses too many diseased images. |

## Summary

`lka_head` is the only Group C model that beats the clean baseline on the main segmentation metrics. It improves full test mask mAP50 by `+0.0593`, labeled-only mask mAP50 by `+0.0585`, and healthy-aware labeled mAP50 by `+0.0584` while reducing disease miss rate from `12.50%` to `7.95%`.

The tradeoff is runtime: baseline is much faster at about `17.16` minutes, while `lka_head` takes `52.81` minutes. Healthy FP rate is equal between baseline and LKA at `36.59%`, so LKA's gain comes mostly from better diseased-mask quality and recall rather than stricter healthy rejection.
