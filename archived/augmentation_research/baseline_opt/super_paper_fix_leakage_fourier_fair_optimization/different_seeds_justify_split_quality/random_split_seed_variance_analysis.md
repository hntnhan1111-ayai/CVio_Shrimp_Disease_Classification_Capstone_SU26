# Random Split Seed-Variance Analysis

Source folder: `different_seeds_justify_split_quality`

This run evaluates `yolo11n-seg.pt` with the same clean/light YOLO augmentation across three seeds (`42`, `123`, `3407`) for two image-level split policies:

- deterministic naive random image split
- disease-stratified random image split

Both policies allow specimen leakage. The purpose is to test whether image-level random evaluation is stable enough to be a reliable benchmark.

## Artifact Check

Present:

- `random_split_seed_variance_paper_table.csv`
- `random_split_seed_variance_summary.csv`
- executed notebook
- split manifests and fingerprints for all 6 split-policy/seed combinations
- `best.pt`, `results.csv`, and `results.png` for all 6 runs

Missing:

- `random_split_seed_variance.csv`
- `random_split_seed_variance_partial.csv`

The missing full CSV is useful but not critical for the main paper table because the compact paper table and summary are present. If the Kaggle session is still alive, download `random_split_seed_variance.csv` as the richer 96-column record.

## Fingerprint Verification

All downloaded split manifests reproduce the fingerprints recorded in the paper table.

| Split Policy | Seed | Fingerprint Match |
|---|---:|---|
| plain random image | 42 | yes |
| plain random image | 123 | yes |
| plain random image | 3407 | yes |
| stratified random image | 42 | yes |
| stratified random image | 123 | yes |
| stratified random image | 3407 | yes |

## Main Result

| Split Policy | Runs | Labeled Test mAP50 Mean | Std | Range | Full Test mAP50 Mean | Std | Range | Train-Test Overlap Mean | Overlap Range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain random image | 3 | 0.6766 | 0.0127 | 0.0254 | 0.6711 | 0.0135 | 0.0270 | 98.7 | 2 |
| stratified random image | 3 | 0.6592 | 0.0619 | 0.1172 | 0.6510 | 0.0596 | 0.1150 | 94.0 | 11 |

## Per-Seed Results

| Split Policy | Seed | Labeled Test mAP50 | Full Test mAP50 | Healthy FP Rate | Disease Miss Rate | Train-Test Overlap |
|---|---:|---:|---:|---:|---:|---:|
| plain random image | 42 | 0.6641 | 0.6577 | 0.0682 | 0.0556 | 98 |
| plain random image | 3407 | 0.6759 | 0.6710 | 0.0227 | 0.0694 | 98 |
| plain random image | 123 | 0.6896 | 0.6847 | 0.0714 | 0.0568 | 100 |
| stratified random image | 123 | 0.7063 | 0.6994 | 0.0976 | 0.1333 | 87 |
| stratified random image | 3407 | 0.6821 | 0.6690 | 0.1220 | 0.0533 | 97 |
| stratified random image | 42 | 0.5891 | 0.5844 | 0.0732 | 0.0133 | 98 |

## Interpretation

The image-level random protocols are not stable enough to serve as the final generalization benchmark. Even after making the split construction deterministic, changing only the seed changes which related specimen images leak between train and test.

The stratified random image split is especially variable here: its labeled-test mAP50 ranges from `0.5891` to `0.7063`, a spread of `0.1172`. That spread is large relative to the fair grouped baseline and could change the conclusion of an experiment.

Stratification balances disease/healthy distributions better, but it does not prevent specimen leakage. In this run, stratified random has lower mean train-test overlap than plain random, but higher metric variance. This indicates that overlap count alone is insufficient; the identity and difficulty of leaked specimens also matter.

Paper-ready claim:

> Image-level random splitting is allocation-sensitive in this dataset. Across seeds, the same model and training configuration produced substantially different test mAP under random image-level splits, despite deterministic split construction. Therefore, image-level random metrics are not a reliable estimate of generalization to unseen shrimp specimens. Specimen-grouped splitting is required for a fair benchmark.

