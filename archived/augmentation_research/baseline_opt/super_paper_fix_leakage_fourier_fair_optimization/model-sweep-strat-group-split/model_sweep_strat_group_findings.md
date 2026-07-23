# Finding: YOLO Model Sweep Under Leakage-Free Grouped-Specimen Split

## Purpose

This experiment selects the YOLO segmentation backbone to use for later optimization. The comparison is intentionally run only under the stratified grouped-specimen split, because this split prevents the same shrimp specimen from appearing in both train and test sets.

The goal is not to maximize the easiest image-level score. The goal is to choose the model that performs best under the paper-facing fair evaluation setting.

## Experiment Contract

- Dataset: hand-labeled shrimp disease segmentation dataset.
- Split policy: `stratified_grouped_specimen`.
- Seed: `42`.
- Train/valid/test images: `905 / 115 / 129`.
- Train-test specimen overlap: `0`.
- Split fingerprint: `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`.
- Augmentation setting: clean/light YOLO augmentation enabled.
- Ultralytics Albumentations hook: enabled.
- Model candidates:
  - `yolov8n-seg.pt`
  - `yolov8s-seg.pt`
  - `yolo11n-seg.pt`
  - `yolo11s-seg.pt`
  - `yolo26n-seg.pt`
  - `yolo26s-seg.pt`

## Evidence Files

- Notebook log: `model-sweep-strat-group-split.ipynb`
- Full metric report: `reports (1)/split_policy_model_comparison.csv`
- Paper table: `reports (1)/split_policy_model_comparison_paper_table.csv`
- Split manifest: `reports (1)/split_manifests/stratified_grouped_specimen_seed42_manifest.csv`
- Split summary: `reports (1)/split_manifests/stratified_grouped_specimen_seed42_summary.csv`
- Split fingerprint: `reports (1)/split_manifests/stratified_grouped_specimen_seed42_fingerprint.txt`
- Training artifacts: `segment (1)/`

Each model run folder contains `weights/best.pt`, `weights/last.pt`, `results.csv`, `results.png`, and `args.yaml`.

## Main Results

| Rank | Model | Labeled test mask mAP50 | Full test mask mAP50 | Labeled test mask mAP50-95 | Healthy FP rate | Healthy-aware val score | Healthy-aware test score |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `yolo11n-seg.pt` | `0.5120` | `0.4815` | `0.1678` | `0.2439` | `0.4411` | `0.4596` |
| 2 | `yolo26s-seg.pt` | `0.4604` | `0.4170` | `0.1821` | `0.2683` | `0.4391` | `0.3972` |
| 3 | `yolov8n-seg.pt` | `0.4316` | `0.3951` | `0.1388` | `0.3415` | `0.4249` | `0.3527` |
| 4 | `yolov8s-seg.pt` | `0.4436` | `0.3914` | `0.1588` | `0.3415` | `0.4109` | `0.3643` |
| 5 | `yolo26n-seg.pt` | `0.4628` | `0.4301` | `0.1601` | `0.1951` | `0.3866` | `0.3930` |
| 6 | `yolo11s-seg.pt` | `0.3606` | `0.2971` | `0.1239` | `0.5122` | `0.3675` | `0.2503` |

## Backbone Selection

`yolo11n-seg.pt` is selected as the main optimization backbone.

This choice is justified because it has:

- The best validation healthy-aware score: `0.4411`.
- The best labeled-test mask mAP50: `0.5120`.
- The best full-test mask mAP50: `0.4815`.
- A substantially better healthy-aware test score than the closest validation competitor, `yolo26s-seg.pt`.

`yolo26s-seg.pt` is the closest competitor by validation healthy-aware score, but it generalizes worse on the test set:

- `yolo11n-seg.pt` healthy-aware test score: `0.4596`.
- `yolo26s-seg.pt` healthy-aware test score: `0.3972`.

`yolo26n-seg.pt` has the lowest healthy false positive rate, but its validation healthy-aware score is much lower. This suggests that it may be more conservative, but not the strongest model for overall segmentation quality.

## Paper Interpretation

This sweep supports the following paper claim:

> Under a leakage-free stratified grouped-specimen split, YOLO11n-seg provides the strongest trade-off between segmentation accuracy and healthy false positive control among YOLOv8, YOLO11, and YOLO26 n/s variants. Therefore, subsequent preprocessing and Fourier-domain optimization experiments use YOLO11n-seg as the fixed backbone.

The result is important because the backbone is selected under the same fair evaluation protocol used for the paper's main argument. This avoids selecting a model based on image-level leakage or unstable random split behavior.

## Notes For Next Experiments

Future Fourier and preprocessing ablations should keep the following fixed unless intentionally studying sensitivity:

- Model: `yolo11n-seg.pt`.
- Split policy: `stratified_grouped_specimen`.
- Seed: `42` for direct comparison with this sweep.
- Split fingerprint: `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`.
- Training augmentation: clean/light YOLO augmentation enabled.
- Model selection: validation healthy-aware score.
- Final reporting: test metrics, including full-test mAP, labeled-only mAP, healthy false positive rate, disease missed-image rate, mask count MAE, and healthy-aware score.

Once a promising Fourier configuration is found, rerun the selected baseline and Fourier method across multiple grouped-specimen seeds to confirm that the improvement is stable rather than split-specific.
