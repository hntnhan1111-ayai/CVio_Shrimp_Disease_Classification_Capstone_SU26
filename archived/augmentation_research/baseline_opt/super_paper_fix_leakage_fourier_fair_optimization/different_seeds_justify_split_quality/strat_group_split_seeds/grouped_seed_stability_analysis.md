# Stratified Grouped-Specimen Seed-Stability Analysis

Source folder: `different_seeds_justify_split_quality/strat_group_split_seeds`

This run evaluates `yolo11n-seg.pt` with the same clean/light YOLO augmentation across three grouped-specimen split seeds (`42`, `123`, `3407`). All images from the same shrimp are kept in one split.

## Artifact Check

Present:

- `reports/grouped_seed_stability.csv`
- `reports/grouped_seed_stability_paper_table.csv`
- `reports/grouped_seed_stability_summary.csv`
- `reports/grouped_seed_stability_partial.csv`
- `reports/split_manifests/*`
- executed notebook
- `best.pt`, `last.pt`, `results.csv`, `results.png`, and plot artifacts for all 3 grouped runs

All downloaded split manifests reproduce the fingerprints recorded in the paper table.

## Main Result

| Seed | Train Images | Valid Images | Test Images | Train-Test Overlap | Labeled Test mAP50 | Full Test mAP50 | Healthy FP Rate | Disease Miss Rate | Healthy-Aware Test Score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 913 | 113 | 123 | 0 | 0.4449 | 0.4319 | 0.1220 | 0.0732 | 0.3986 |
| 42 | 905 | 115 | 129 | 0 | 0.5120 | 0.4815 | 0.2439 | 0.0795 | 0.4596 |
| 123 | 921 | 105 | 123 | 0 | 0.4619 | 0.4398 | 0.3415 | 0.0976 | 0.3881 |

## Stability Summary

| Metric | Mean | Std | Range |
|---|---:|---:|---:|
| Full test mask mAP50 | 0.4511 | 0.0267 | 0.0496 |
| Labeled test mask mAP50 | 0.4729 | 0.0349 | 0.0671 |
| Healthy false-positive rate | 0.2358 | 0.1100 | 0.2195 |
| Train-test specimen overlap | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

The grouped-specimen protocol removes the leakage confound completely: train-test and valid-test specimen overlap are zero for all seeds.

The mean fair grouped labeled-test mask mAP50 is `0.4729`, far below the image-level random split means from the companion seed-variance run:

- plain random image split mean labeled-test mAP50: `0.6766`
- stratified random image split mean labeled-test mAP50: `0.6592`
- stratified grouped-specimen split mean labeled-test mAP50: `0.4729`

This confirms that the performance drop under grouped evaluation is not a one-seed artifact.

Healthy false positives remain unstable and high under grouped evaluation, ranging from `0.1220` to `0.3415`. This is important for deployment: the fair protocol reveals that the model still confuses unseen healthy specimens with disease, even when mAP is moderate.

Paper-ready claim:

> Across three grouped-specimen seeds, specimen overlap was consistently zero and labeled-test mask mAP50 averaged 0.4729. Compared with image-level random splits, which averaged 0.6592-0.6766, the grouped protocol reveals a substantially lower but more realistic estimate of generalization to unseen shrimp specimens. The grouped runs also expose healthy false-positive rates that are hidden by leakage-prone random splits.

