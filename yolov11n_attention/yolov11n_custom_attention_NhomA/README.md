# YOLO11n Custom Attention - Nhom A

This folder implements Group A from `custom_attention_improvement_transfer_plan.md`.
The focus is evaluation and calibration, so the notebooks avoid architecture
changes unless they are loading existing custom-attention checkpoints.

## Notebooks

- `yolo11_cus_att_tta_inference.ipynb`: A1 TTA inference at the default `conf=0.25`, `iou=0.70`.
- `yolo11_cus_att_threshold_sweep.ipynb`: A2 sweep over `conf=[0.10..0.50]` and `iou=[0.50,0.60,0.70]`.
- `yolo11_cus_att_matched_baseline_strong_augmentation.ipynb`: A3 baseline YOLO11n-seg with the same strong augmentation policy used by CoTE/LPSC strong runs.

## Protocol Notes

- The baseline notebook is not modified.
- The grouped shrimp-level no-leakage dataset split is reused.
- Evaluation keeps the baseline healthy-aware score:

```text
score = diseased_mask_mAP50
        - 0.05 * mask_count_MAE
        - 0.15 * disease_miss_rate
        - 0.10 * healthy_mask_FP_rate
```

- A1/A2 auto-discover common checkpoint paths, but each notebook has
  `CHECKPOINT_OVERRIDES` for Kaggle sessions with different run names.
- The helper registers the same custom attention classes used by
  `custom_attention`, `custom_attention_research_modules`, and
  `custom_attention_multiseed`.
- Local preflight passed for the referenced custom YAMLs at `imgsz=640` with a
  CPU dummy forward: Triplet, CoTE clean/strong, LPSC clean/strong, CESA-Lite,
  and SGE-ECA.

## Expected Dataset

Run the baseline data-preparation cells first, or point `BASE_DATA_DIR` to a
prepared grouped split containing:

```text
train/images, train/labels
valid/images, valid/labels
test/images, test/labels
data.yaml
```
