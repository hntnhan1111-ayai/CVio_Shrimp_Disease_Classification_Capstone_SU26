# Candidate Behaviour on the Old and Expanded Datasets

## Scope

This report compares the three enabled candidate configurations evaluated on the expanded dataset with their matched configurations from the original dataset. The comparison uses the exact recorded paper rows, not rounded values copied from notebook displays.

The new-dataset candidates use the same model and main training contract as the new baseline: YOLO11n-seg, Ultralytics 8.4.62, seed 42, batch 16, image size 640, 100 requested epochs, patience 30, and the mixed grouped-plus-unmatched split. Fourier in M2 and M3 is W1: offline high-pass preprocessing with sigma 50 and alpha 0.1.

## Candidate Definitions

| Mode | Configuration | Fourier | Augmentation | Hook |
|---|---|---|---|---|
| M0 | New-dataset reference baseline | No | Clean-light | Enabled |
| M1 | Strong augmentation | No | Strong | Disabled |
| M2 | W1 with clean-light augmentation | W1 | Clean-light | Enabled |
| M3 | W1 with strong augmentation | W1 | Strong | Disabled |

M0 was the reference baseline and was not rerun in the candidate panel. M1, M2, and M3 were the enabled candidate runs.

## Exact Results

### Original dataset

| Configuration | Labeled test mAP50 | Labeled test mAP50-95 | Healthy FP rate | Disease miss rate | Healthy-aware test score |
|---|---:|---:|---:|---:|---:|
| M0: clean-light, no Fourier | 0.512002 | 0.167785 | 0.243902 | 0.079545 | 0.459581 |
| M1: strong, no Fourier | 0.595095 | 0.211258 | 0.341463 | 0.090909 | 0.532634 |
| M2: clean-light + W1 | 0.531771 | 0.170904 | 0.243902 | 0.102273 | 0.470165 |
| M3: strong + W1 | 0.577126 | 0.208817 | 0.317073 | 0.147727 | 0.501480 |

### Expanded dataset

| Configuration | Labeled test mAP50 | Labeled test mAP50-95 | Healthy FP rate | Disease miss rate | Healthy-aware test score |
|---|---:|---:|---:|---:|---:|
| M0: clean-light, no Fourier | 0.563676 | 0.233139 | 0.268293 | 0.091667 | 0.502055 |
| M1: strong, no Fourier | **0.612328** | 0.236280 | 0.463415 | 0.091667 | 0.534806 |
| M2: clean-light + W1 | 0.571258 | 0.240800 | 0.317073 | 0.091667 | 0.503162 |
| M3: strong + W1 | 0.598953 | **0.251300** | **0.292683** | **0.058333** | **0.546491** |

## Expanded-Dataset Change Relative to the Original Dataset

Positive changes in mAP and healthy-aware score are improvements. Lower false-positive and disease-miss rates are improvements.

| Configuration | mAP50 change | mAP50-95 change | Healthy FP change | Disease miss change | Healthy-aware change |
|---|---:|---:|---:|---:|---:|
| M0: clean-light, no Fourier | +0.051674 | +0.065354 | +0.024390 | +0.012122 | +0.042474 |
| M1: strong, no Fourier | +0.017233 | +0.025022 | +0.121952 | +0.000758 | +0.002172 |
| M2: clean-light + W1 | +0.039487 | +0.069896 | +0.073171 | -0.010606 | +0.032997 |
| M3: strong + W1 | +0.021827 | +0.042483 | **-0.024390** | **-0.089394** | **+0.045011** |

The baseline and all candidates improved their raw segmentation metrics on the expanded dataset. M3 improved on every listed practical metric, including both healthy false positives and disease misses.

This is an across-dataset comparison, not a causal estimate of the effect of adding data. The expanded dataset has more images and a different test composition, so the result should be described as performance behaviour under the new evaluation distribution.

## Fourier Interaction on the Expanded Dataset

The direct comparison between M1 and M3 isolates the W1 addition within the strong-augmentation regime:

| Metric | M1: strong, no Fourier | M3: strong + W1 | W1 change |
|---|---:|---:|---:|
| Labeled test mAP50 | 0.612328 | 0.598953 | -0.013375 |
| Labeled test mAP50-95 | 0.236280 | 0.251300 | **+0.015020** |
| Healthy FP rate | 0.463415 | 0.292683 | **-0.170732** |
| Disease miss rate | 0.091667 | 0.058333 | **-0.033333** |
| Healthy-aware test score | 0.534806 | 0.546491 | **+0.011685** |

On the expanded dataset, W1 still lowers raw mAP50 under strong augmentation, but it improves localization quality, healthy-image control, disease recall, and the healthy-aware score. This differs from the original strong-augmentation comparison, where W1 reduced both mAP50 and healthy-aware performance.

The clean-light comparison is weaker:

- New M0 to M2 mAP50: `0.563676 -> 0.571258` (`+0.007582`)
- New M0 to M2 mAP50-95: `0.233139 -> 0.240800` (`+0.007660`)
- New M0 to M2 healthy FP rate: `0.268293 -> 0.317073` (`+0.048780`)
- New M0 to M2 healthy-aware score: `0.502055 -> 0.503162` (`+0.001107`)

Therefore, W1 is more useful in the strong-augmentation regime than in the clean-light regime for the expanded dataset.

## Rankings on the Expanded Dataset

### Raw labeled mAP50

1. M1 strong augmentation, no Fourier: `0.612328`
2. M3 strong augmentation + W1: `0.598953`
3. M2 clean-light + W1: `0.571258`
4. M0 clean-light reference: `0.563676`

### Healthy-aware test score

1. M3 strong augmentation + W1: `0.546491`
2. M1 strong augmentation, no Fourier: `0.534806`
3. M2 clean-light + W1: `0.503162`
4. M0 clean-light reference: `0.502055`

### Practical error control

M3 is the strongest candidate: it has the best mAP50-95, lowest disease miss rate, and lowest healthy FP rate among the candidate runs. M1 remains the raw-mAP50 winner, but its healthy FP rate is substantially higher.

## Conclusions

1. Strong augmentation is the main contributor to raw mAP50 on the expanded dataset.
2. W1 does not maximize raw mAP50, but it improves the more balanced operating profile when combined with strong augmentation.
3. Clean-light + W1 provides only a marginal healthy-aware improvement over the new baseline and increases healthy false positives.
4. For continued evaluation, M3 should be retained as the primary Fourier candidate and M1 as the no-Fourier raw-mAP control.
5. Multi-seed confirmation should compare at least M0, M1, and M3 using the same expanded split protocol before making a final claim.

## Evidence

- New candidate paper rows: [new_dataset_candidate_transfer_panel_seed42_paper_row.csv](C:/Users/Admin/Downloads/new_dataset_candidate_transfer_panel_seed42_paper_row.csv)
- New baseline paper row: [seg_yolo11n_new_dataset_mixed_split_baseline_paper_row (1).csv](<C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/shrimp-leakage-aware-segmentation/train_log/seg_yolo11n_new_dataset_mixed_split_baseline_paper_row (1).csv>)
- New baseline execution log: [logyolo11n-new-dataset-mixed-split-baseline-kaggle.ipynb](C:/Users/Admin/Downloads/logyolo11n-new-dataset-mixed-split-baseline-kaggle.ipynb)
- Original-dataset metrics registry: [[hand-explanation]final_metrics_and_training_configs_registry.md](<C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/shrimp-leakage-aware-segmentation/ONLY_fourier/[hand-explanation]final_metrics_and_training_configs_registry.md>)
