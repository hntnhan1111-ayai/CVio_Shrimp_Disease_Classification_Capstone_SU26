# Group B vs Clean YOLO11n-seg Baseline

Baseline source: `yolov11n_attention/yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb`

Group B sources:
- `yolov11n_simam_WIoUv3.ipynb`
- `yolov11n_simam_BoundaryLoss.ipynb`
- `yolov11n_simam_AsymFocalLoss.ipynb`

Notes:
- Ranking is by healthy-aware score.
- Baseline full/labeled mask mAP50, miss rate, healthy FP rate, and healthy-aware score are read from notebook outputs.
- Baseline mAP50-95 values are rounded because the linked notebook exposes them in the Ultralytics log at three decimals.
- Baseline training time is derived from the notebook log: `60 epochs completed in 0.286 hours` = `17.16` minutes.
- Group B full test mask mAP50-95 values are taken from the Ultralytics evaluation log, while labeled mask mAP50-95 values come from each notebook summary dataframe.

| Rank | Model | Loss variant | Train min | Epochs | Full mask mAP50 | Delta vs baseline | Full mask mAP50-95 | Labeled mask mAP50 | Delta vs baseline | Labeled mask mAP50-95 | Disease miss rate | Healthy FP rate | Pred masks | Count MAE | Healthy-aware score | Delta vs baseline | Quick read |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `simam_ca_wiouv3` | WIoU_v3 | 48.61 | 67 | 0.4314 | -0.0095 | ~0.1400 | 0.4539 | -0.0125 | 0.1473 | 12.50% | 17.07% | 119 | 0.4545 | 0.3954 | +0.0020 | Best healthy-aware score, mainly because healthy FP is much lower than baseline. |
| 2 | `clean_light_aug_baseline` | YOLO default loss | 17.16 | 60 | 0.4409 | +0.0000 | ~0.1610 | 0.4664 | +0.0000 | ~0.1690 | 12.50% | 36.59% | N/A | 0.3542 | 0.3934 | +0.0000 | Strong and fast reference; still best full mask mAP50 among Group B comparisons. |
| 3 | `simam_ca_boundary` | BCE + BoundaryIoU (weight=0.5) | 72.75 | 100 | 0.4215 | -0.0194 | ~0.1360 | 0.4699 | +0.0034 | 0.1529 | 15.91% | 31.71% | 111 | 0.4413 | 0.3922 | -0.0012 | Slightly improves labeled mAP50, but loses on full mAP and miss rate. |
| 4 | `simam_ca_asl` | ASL (gamma_neg=4, gamma_pos=0, clip=0.05) | 42.54 | 45 | 0.1917 | -0.2493 | ~0.0503 | 0.2116 | -0.2548 | 0.0563 | 42.05% | 43.90% | 58 | 0.5284 | 0.0782 | -0.3151 | Clear regression; under-predicts disease masks and raises healthy false positives. |

## Summary

`simam_ca_wiouv3` is the most useful Group B variant if the priority is reducing false positives on healthy images. It cuts healthy FP rate from `36.59%` to `17.07%`, keeps disease miss rate equal to baseline at `12.50%`, and narrowly beats baseline on healthy-aware score by `+0.0020`.

The tradeoff is mask quality: baseline still has better full mask mAP50 (`0.4409` vs `0.4314`) and better labeled mask mAP50 (`0.4664` vs `0.4539`). `simam_ca_boundary` is competitive because it slightly improves labeled mask mAP50 to `0.4699`, but its full mAP50 and miss rate are worse. `simam_ca_asl` should be rejected for this setup.
