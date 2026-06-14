# Research Plan: Leakage-Safe Shrimp Disease Segmentation and Fourier-Based Fair Optimization

Working title:

> Specimen-Leakage-Aware Evaluation and Frequency-Domain Optimization for Shrimp Disease Segmentation

This document is the planning source for the ICIT-style paper in `icit-template/`. It records the paper story, expected contributions, experiment contract, missing code, and results that must be collected before writing the final manuscript.

## 1. Core Paper Claim

The strongest version of this paper is not simply that Fourier preprocessing improves YOLO segmentation. The stronger claim is:

> Image-level random splitting substantially overestimates shrimp disease segmentation performance when multiple images of the same shrimp specimen exist. We address this by creating a consistently hand-labeled segmentation dataset, enforcing specimen-grouped disease-stratified evaluation, and studying frequency-domain preprocessing under this fairer protocol.

The paper should frame Fourier transform as the improvement study after establishing a fair evaluation protocol. The leakage correction is the methodological anchor; Fourier is the optimization contribution.

## 2. Main Contributions

### Contribution 1: Clean Hand-Labeled Segmentation Dataset

We created a new segmentation dataset for shrimp disease from `shrimpdiseasebd`, relabeled by one annotator to reduce inconsistent mask style and incomplete annotation.

Why this matters:

- Previous labels were inconsistent or incomplete.
- Multi-person annotation introduced another source of inconsistency.
- Segmentation labels require consistent boundary decisions; noisy masks directly harm mAP50 and mAP50-95.

Evidence to collect:

- total image count
- healthy image count
- diseased/labeled image count
- mask instance count
- per-class mask count
- specimen count
- images per specimen distribution
- train/valid/test split counts under grouped-stratified split
- example annotation panels

### Contribution 2: Specimen-Grouped Disease-Stratified Split

Each physical shrimp specimen can have multiple images. All images from the same specimen must stay in one split.

Why this matters:

- Random image-level splitting can place the same shrimp in train and test.
- The model may learn specimen-specific body texture, lighting, pose, and background instead of disease-generalizable cues.
- The observed drop from random split mAP50 around `0.80` to grouped-stratified mAP50 around `0.466` is direct evidence that random split evaluation is inflated.

Paper framing:

> The lower grouped-split score is not a worse model; it is a more honest estimate of generalization to unseen shrimp.

Evidence to collect:

- random image-level split baseline on hand-labeled dataset
- specimen-grouped split baseline on the same hand-labeled dataset
- specimen-grouped disease-stratified split baseline on the same hand-labeled dataset
- optional: examples of visually similar same-specimen images that would leak under random splitting

### Contribution 3: Fair YOLO11n-Seg Baseline

The clean baseline uses:

- YOLO11n-seg
- grouped disease-stratified split
- clean-light YOLO augmentation
- hidden Ultralytics Albumentations disabled
- deterministic validation/test evaluation
- healthy-aware diagnostics

Why YOLO11n-seg:

- small model
- fast training/inference
- practical for future mobile/edge deployment
- good enough baseline after leakage-safe evaluation

Evidence to collect:

- YOLOv8n/s/m and YOLO11n/s/m comparison on the same split and training contract
- parameter count
- training time
- inference time if possible
- mAP and healthy false-positive metrics

### Contribution 4: Healthy-Aware Evaluation

Segmentation mAP is not enough. The model must detect disease while avoiding disease predictions on healthy shrimp.

Primary metrics:

- labeled-only diseased test mask mAP50
- full test mask mAP50
- healthy mask false-positive rate
- healthy false-positive masks per image
- missed diseased image rate
- mask count MAE
- confidence-threshold sweeps

Why this matters:

- A model with high mAP but frequent healthy false positives is not practically trustworthy.
- Fourier can improve disease sensitivity while increasing healthy false positives, so the tradeoff must be measured explicitly.

### Contribution 5: Fourier/Frequency-Domain Preprocessing Under Fair Evaluation

Fourier high-pass detail boosting enhances frequency components associated with edges and fine texture. In shrimp disease segmentation, this may make subtle disease patterns more visible.

Current evidence:

- clean-light grouped baseline labeled test mask mAP50: about `0.466`
- Fourier + clean-light best mAP-oriented result: about `0.512` to `0.516`
- absolute improvement: about `+0.046` to `+0.050`
- relative improvement: roughly `+10%`

Important caveat:

- Fourier train-only gave strong labeled mAP but high healthy false positives.
- Fourier all-splits settings may be more balanced.

Evidence to collect:

- top Fourier settings with 3 seeds
- confidence sweeps for top Fourier settings
- qualitative predictions showing where Fourier helps and where it creates false positives

## 3. Proposed Paper Structure

The ICIT/Springer `svproc` template is compact. The final paper should be concise and table-heavy.

### Abstract

70-150 words, following the template constraint. Include:

- dataset relabeling
- specimen leakage problem
- grouped-stratified split
- YOLO11n-seg fair baseline
- Fourier improvement
- healthy false-positive tradeoff

### 1. Introduction

Points to cover:

- shrimp disease monitoring needs reliable visual segmentation
- existing datasets are mostly classification or have inconsistent segmentation labels
- segmentation evaluation is vulnerable to specimen leakage
- random image-level split can strongly inflate mAP
- this paper proposes leakage-safe evaluation and frequency-domain optimization

End with contribution bullets.

### 2. Related Work

Subsections:

- shrimp disease image datasets and classification
- segmentation models for biological/agricultural disease imaging
- data leakage in visual/medical datasets
- frequency-domain preprocessing for visual recognition

Important references to add:

- subject/slice leakage in medical imaging
- near-duplicate leakage in computer vision datasets
- PlantVillage/leaf grouping or biological specimen grouping analogies
- YOLO segmentation reference
- Fourier/image frequency enhancement references

### 3. Dataset and Annotation

Subsections:

- source classification dataset from Mendeley: `https://data.mendeley.com/datasets/jhrtdj9txm/3`
- previous EWU/Roboflow segmentation dataset and caveats: `https://universe.roboflow.com/shirmpdiseasedtection/ewu_shrimp_disease`
- new hand-labeled dataset construction: `https://universe.roboflow.com/lets-try-this/shrimpdishandsegv2`
- annotation rules
- specimen identity/grouping
- distribution summary

Filename/specimen conventions:

| Dataset | Filename pattern | Specimen key |
|---|---|---|
| Mendeley classification | `<ShrimpDisease>-<ShrimpID>-img-<imgnum>.jpg` | `<ShrimpDisease>::<ShrimpID>` |
| Hand-labeled segmentation | `<ShrimpDisease>-<ShrimpID>-img-<imgnum>.jpg`, sometimes Roboflow-normalized | `<ShrimpDisease>::<ShrimpID>` |
| EWU segmentation | `Shrimp_<shrimpid>-<imgnum>.jpg`, `Shrimp_<shrimpid>-<imgnum>-.jpg`, or single-image `Shrimp_<shrimpid>.jpg` after Roboflow normalization | `shrimp::<shrimpid>` |

EWU caveat:

- EWU IDs do not appear to map directly to Mendeley shrimp IDs, for example `Shrimp_331`.
- EWU grouping must follow the leakage-fixed old-dataset parser: ignore the trailing image number and group by `Shrimp_<shrimpid>`.
- Treat EWU as an independently labeled segmentation resource unless a reliable metadata bridge is found.
- The EDA should quantify EWU label counts, image counts, images per inferred shrimp ID, and class/mask distribution before using it as an optimization benchmark.

Required table:

| Dataset | Task | Images | Classes | Masks | Healthy images | Specimen IDs | Known caveats |
|---|---:|---:|---|---:|---:|---:|---|

Required figure:

- sample healthy image
- sample diseased image
- sample mask overlay
- same-specimen multi-image example

### 4. Leakage-Safe Evaluation Protocol

Explain:

- why random image split is unfair
- specimen group definition
- disease-aware stratification
- train/valid/test split ratios
- leakage check/assertion

Required table:

| Split policy | Test mask mAP50 | Labeled test mask mAP50 | Healthy FP rate | Interpretation |
|---|---:|---:|---:|---|
| random image split | ~0.80 | TBD | TBD | inflated by specimen leakage |
| grouped split | TBD | TBD | TBD | fairer unseen-specimen evaluation |
| grouped-stratified split | ~0.466 baseline | ~0.466 | TBD | final protocol |

Use the random-vs-group drop as a core result.

### 5. Baseline Model and Evaluation Metrics

Baseline:

- YOLO11n-seg
- input size 640
- clean-light augmentation
- no mosaic/mixup/cutmix/copy-paste
- hidden Albumentations disabled
- patience and epoch settings from notebooks

Metrics:

- mask mAP50
- mask mAP50-95
- box mAP50
- full test metrics
- labeled-only diseased metrics
- healthy false-positive rate
- healthy FP masks per image
- missed diseased image rate
- mask count MAE
- healthy-aware validation score

Required table:

| Metric | Definition | Why important | Used for |
|---|---|---|---|

### 6. Fourier Preprocessing Method

Describe Fourier high-pass detail boosting:

```text
enhanced image = original image + alpha * high-frequency detail
```

Hyperparameters:

- `fourier_sigma`: controls low-pass filter width
- `fourier_alpha`: controls detail boost strength
- `fourier_mode`: `all_splits`, `train_only`, `train_copy`

Required equation:

Use compact math in LaTeX:

```text
I_enh = clip(I + alpha * (I - LP_sigma(I)))
```

where `LP_sigma` is a Gaussian low-pass approximation in the frequency domain.

Required table:

| Setting | Sigma | Alpha | Mode | Hypothesis |
|---|---:|---:|---|---|
| s50-a03 | 50 | 0.30 | all_splits | gentler version of strong Fourier |
| s50-a05 | 50 | 0.50 | all_splits | stronger mAP-oriented reference |
| s70-a03 | 70 | 0.30 | all_splits | more conservative high-pass |
| s50-a03 train-only | 50 | 0.30 | train_only | augmentation/regularization mode |

### 7. Experiments

Experiment groups:

1. Dataset distribution and annotation summary
2. Random split vs specimen-grouped split
3. YOLO model scale comparison
4. Clean-light baseline
5. Preprocessing ablation: CLAHE, Fourier, CLAHE+Fourier, LBP, RandAug
6. Fourier sweep
7. Confidence threshold analysis
8. Optional noise robustness

### 8. Results

Result tables to include:

#### Table A: Leakage Inflation

| Dataset | Split | Model | Labeled test mask mAP50 | Full test mask mAP50 | Healthy FP rate |
|---|---|---|---:|---:|---:|

#### Table B: Model Size Justification

| Model | Params | Time | Full mAP50 | Labeled mAP50 | Healthy FP |
|---|---:|---:|---:|---:|---:|

#### Table C: Preprocessing Ablation

| Method | Full mAP50 | Labeled mAP50 | Healthy FP | Missed disease | Count MAE |
|---|---:|---:|---:|---:|---:|

#### Table D: Fourier Sweep

| Fourier setting | Mode | Full mAP50 | Labeled mAP50 | Healthy FP | Selected? |
|---|---|---:|---:|---:|---|

#### Table E: Confidence Threshold Sweep

For top 2-3 models only.

| Model | Conf | Healthy FP | Missed disease | Mask count MAE |
|---|---:|---:|---:|---:|

### 9. Discussion

Key points:

- Random split strongly overestimates performance.
- Grouped-stratified split is justified and should be standard for this dataset type.
- Clean-light augmentation is a baseline necessity, not a luxury.
- Fourier improves disease segmentation but introduces a sensitivity/specificity tradeoff.
- Healthy-aware metrics expose behavior that mAP alone hides.
- Future work: wavelet, hard-negative mining, background removal, multi-seed validation, larger external dataset.

### 10. Limitations

Must state honestly:

- small dataset
- single annotator improves consistency but does not prove clinical/agricultural ground truth
- specimen IDs may be inferred from filenames/grouping conventions
- Fourier settings are empirical
- test set is still limited
- no external validation dataset yet
- healthy/diseased labels may have ambiguous borderline cases

## 4. Current Known Results

These are working notes and must be verified before final paper tables.

### Clean Baseline

| Setting | Labeled test mask mAP50 | Full test mask mAP50 | Healthy FP rate |
|---|---:|---:|---:|
| clean-light grouped baseline | ~0.466 | ~0.441 | ~0.366 |

### Fourier Results

| Setting | Mode | Labeled test mask mAP50 | Full test mask mAP50 | Healthy FP rate | Notes |
|---|---|---:|---:|---:|---|
| Fourier s50-a05 | all_splits | ~0.509 | ~0.469 | ~0.390 | strong confirmed mAP |
| Fourier s50-a03 | all_splits | ~0.504 | ~0.468 | TBD | likely balanced, needs FP sweep |
| Fourier s70-a03 | all_splits | ~0.443 | ~0.416 | ~0.317 | validation-selected conservative run |
| Fourier s50-a03 | train_only | ~0.516 | ~0.481 | ~0.512 | best mAP, high healthy FP |

Interpretation:

- Fourier is a real improvement signal.
- The train-only setting may be too sensitive for practical use unless thresholding/calibration reduces false positives.
- `s50-a03 all_splits` is currently the most important missing diagnostic because its healthy FP behavior is not fully collected.

## 5. Missing Results To Collect

Current status after local EDA:

- [x] EWU/hand-labeled dataset distribution table generated.
- [x] EWU/hand-labeled class and co-infection count table generated.
- [x] EWU/hand-labeled same-specimen visual annotation panels generated.
- [x] EWU/hand-labeled visual-hash similar-pair evidence generated.
- [x] Research findings document created with direct links to CSV reports and figures.

High priority:

- [x] hand-labeled dataset distribution table
- [ ] Mendeley classification dataset distribution table, optional for now
- [x] EWU/Roboflow segmentation dataset distribution and caveat summary
- [ ] stratified random image split vs stratified grouped-specimen split on the hand-labeled dataset
- [ ] clean-light baseline with 3 seeds
- [ ] top Fourier setting(s) with 3 seeds
- [ ] direct healthy FP threshold sweep for `fourier_s50_a03_light_yolo_aug`
- [ ] direct labeled diseased threshold sweep for `fourier_s50_a03_light_yolo_aug`
- [ ] YOLOv8n/s/m and YOLO11n/s/m comparison under the same grouped-stratified split

Medium priority:

- [ ] EWU original/random split vs grouped split if specimen grouping can be reconstructed
- [ ] preprocessing ablation table with CLAHE, Fourier, CLAHE+Fourier, LBP, RandAug
- [ ] qualitative prediction panels for baseline vs Fourier
- [ ] healthy false-positive visual examples
- [x] same-specimen annotation-quality illustration figure
- [ ] same-specimen leakage split illustration figure

Optional robustness:

- [ ] Gaussian noise robustness
- [ ] blur robustness
- [ ] brightness/contrast robustness
- [ ] JPEG compression robustness

## 6. Missing Code / Notebook Work

The current folder contains:

- clean baseline notebook
- preprocessing ablation notebook
- Fourier sweep notebook

The following code/notebooks are still missing or incomplete.

### Missing Code 1: Dataset Distribution Report

Needed output:

- CSV/Markdown table of image counts, healthy counts, diseased counts, mask instances, class counts, specimen counts.
- Works for Mendeley classification dataset, EWU segmentation dataset, and hand-labeled dataset when paths are provided.

Suggested artifact:

```text
dataset_eda_specimen_distribution.ipynb
reports/dataset_distribution_summary.csv
```

### Missing Code 2: Split-Policy And Model-Scale Comparison

Needed output:

- same model/training settings
- stratified random image-level split
- stratified grouped-specimen split
- YOLOv8n/s/m and YOLO11n/s/m variants controlled by a model list
- same test metrics and healthy-aware metrics

Suggested artifact:

```text
[AIP491_01]yolo_seg_split_policy_model_comparison_kaggle.ipynb
reports/split_policy_model_comparison.csv
```

Important:

- Random split result should be presented as leakage inflation evidence, not as a better model.
- This can be one runner, but the paper should interpret it as two questions:
  - leakage question: does stratified random image splitting inflate metrics compared with grouped-specimen splitting?
  - model-choice question: under the fair grouped-specimen split, which YOLO variant is the best optimization target?

### Missing Code 3: YOLO Model Scale Comparison

Needed output:

- `yolov8n-seg`
- `yolov8s-seg`
- `yolov8m-seg`
- `yolo11n-seg`
- `yolo11s-seg`
- `yolo11m-seg`
- same grouped-stratified split
- same clean-light augmentation
- same evaluation metrics

Suggested artifact:

```text
covered by [AIP491_01]yolo_seg_split_policy_model_comparison_kaggle.ipynb
reports/split_policy_model_comparison.csv
```

### Missing Code 4: Fourier s50-a03 Extra Diagnostics

Needed output:

- healthy FP threshold sweep
- labeled diseased threshold sweep
- optional visual examples

This is urgent because `s50-a03 all_splits` may be the best balanced Fourier setting.

Suggested artifact:

```text
reports/fourier_s50_a03_threshold_sweep.csv
```

### Missing Code 5: Multi-Seed Runner

Needed output:

- run key experiments with seeds such as `42`, `123`, `3407`
- report mean and standard deviation

Minimum experiments:

- clean-light baseline
- Fourier s50-a03 all_splits
- Fourier s50-a05 all_splits or best previous Fourier
- Fourier s50-a03 train_only if included as sensitivity-focused result

Suggested artifact:

```text
yolo_seg_multiseed_key_results.ipynb
reports/key_results_multiseed.csv
```

### Missing Code 5b: Weighted Model-Selection Function

Needed output:

- one documented validation score used to select a practical best model
- configurable weights for disease segmentation, healthy false positives, missed disease, and count error
- raw metrics still reported separately

Initial score:

```text
weighted_score =
    w_map * labeled_val_mask_map50
    - w_fp * healthy_val_mask_fp_rate
    - w_miss * labeled_val_disease_mask_miss_rate
    - w_count * labeled_val_mask_count_mae
```

Default weights should be stated as assumptions, not universal truth.

### Missing Code 6: Robustness Evaluation

Optional but useful.

Needed perturbations:

- Gaussian noise
- blur
- brightness shift
- contrast reduction
- JPEG compression

Purpose:

- Test whether Fourier improves robustness or overfits to sharpened texture.

Suggested artifact:

```text
yolo_seg_robustness_eval.ipynb
reports/robustness_eval.csv
```

## 7. Figures To Prepare

Must-have figures:

1. Dataset examples:
   - healthy image
   - diseased image
   - mask overlay
2. Leakage example:
   - multiple images from the same shrimp specimen
   - show why random split is unfair
3. Split protocol diagram:
   - image-level random split vs specimen-grouped split
4. Baseline vs Fourier prediction panel:
   - same image, ground truth, clean baseline prediction, Fourier prediction
5. Healthy false-positive panel:
   - healthy image where Fourier predicts false mask

Optional figures:

- Fourier preprocessing visualization
- confidence threshold tradeoff curve
- robustness curves

## 8. ICIT Manuscript Constraints

The provided template uses Springer `svproc`.

Practical implications:

- Keep abstract between 70 and 150 words.
- Keep paper compact.
- Prioritize tables over long narrative.
- Use only the strongest figures.
- Avoid too many ablations in the main text; move secondary results into discussion or appendix if allowed.

Suggested main-paper tables:

1. Dataset distribution
2. Split leakage comparison
3. Main baseline/Fourier results
4. Confidence threshold tradeoff

Suggested main-paper figures:

1. Specimen leakage diagram/example
2. Qualitative predictions baseline vs Fourier

## 9. Answered Assumptions For Code Generation

Dataset sources:

- Mendeley classification dataset: `https://data.mendeley.com/datasets/jhrtdj9txm/3`
- EWU Roboflow segmentation dataset: `https://universe.roboflow.com/shirmpdiseasedtection/ewu_shrimp_disease`
- Hand-labeled Roboflow segmentation dataset: `https://universe.roboflow.com/lets-try-this/shrimpdishandsegv2`

Specimen ID parsing:

- Mendeley and hand-labeled data use `<ShrimpDisease>-<ShrimpID>-img-<imgnum>.jpg`.
- EWU uses `Shrimp_<shrimpid>-<imgnum>.jpg`, sometimes with a trailing dash after the image number in Roboflow exports; grouping ignores `<imgnum>`.
- EWU shrimp IDs may be generated independently and should not be assumed to map to Mendeley IDs.

Model-sweep scope:

- Include YOLOv8 and YOLO11 families.
- Include only n/s/m variants.
- Do not include l/x variants.
- The model sweep is for selecting a baseline worth optimizing, not for exhaustively chasing the largest model.

Model-selection direction:

- Use a weighted function across metrics to select the practical best model.
- Report raw metrics separately so the weighting is transparent.

Still open:

- Local/Kaggle dataset paths must be filled in when running the EDA notebook.
- Robustness experiments can be added after the core leakage, dataset, baseline, and Fourier results are stable.

## 10. Immediate Next Steps

Recommended order:

1. Skip Mendeley classification EDA for now; keep it optional unless the manuscript needs a source-dataset distribution table.
2. Run one controlled split-policy/model runner on the hand-labeled dataset with only two split policies: stratified random image split and stratified grouped-specimen split.
3. Use the grouped-specimen rows from that runner to choose the YOLO variant worth optimizing.
4. After the best YOLO variant is chosen, run Fourier auto-tuning using validation-only healthy-aware selection.
5. Once the best baseline and Fourier settings look stable, run 3-seed confirmation.
6. Generate model prediction panels: ground truth, clean baseline prediction, Fourier prediction, and healthy false-positive examples.
7. Prepare final result tables and convert the plan into `icit-template/manuscript.tex`.

Near-term checklist for the next coding session:

- [x] Create `[AIP491_01]yolo_seg_split_policy_model_comparison_kaggle.ipynb`.
- [ ] Export `reports/split_policy_model_comparison.csv`.
- [ ] Add a reusable weighted model-selection score table.
- [ ] Create Fourier auto-tuning runner for the selected YOLO model.
- [ ] Add threshold-sweep export for top Fourier candidates only.
- [ ] Add qualitative prediction-panel generation from saved checkpoints.

## 11. Current Experiment Contract

### Split Policies For The Next Run

Three split policies are now supported in the paper-facing split notebook:

| Split policy key | Stratification unit | Leakage constraint | Paper role |
|---|---|---|---|
| `plain_random_image` | none | none; same specimen may appear across splits | reproduces the old baseline-style leakage condition |
| `stratified_random_image` | image disease label | none; same specimen may appear across splits | controlled leakage-inflated reference |
| `stratified_grouped_specimen` | specimen disease label | all images from a specimen stay in one split | fair evaluation protocol |

Do not include a plain non-stratified grouped split. It adds compute cost without answering the current paper question.

### YOLO Models For The Split/Model Runner

Use one configurable list:

```python
YOLO_MODELS = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
]
```

Recommended execution order:

1. Leakage ablation: `yolo11n-seg.pt` under `plain_random_image`, `stratified_random_image`, and `stratified_grouped_specimen`.
2. Fair model sweep: `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolo11n-seg.pt`, and `yolo11s-seg.pt` under `stratified_grouped_specimen` only.
3. Optional extended sweep: add `yolov8m-seg.pt` and `yolo11m-seg.pt` only if the smaller sweep shows the larger models are likely worth the compute.
4. Selection slice: choose the Fourier optimization target using grouped-specimen rows only.

### Metrics To Record For Paper Tables

Every run row should include:

| Column | Purpose |
|---|---|
| `split_policy` | separates leakage-inflated and fair protocols |
| `model` | YOLO variant |
| `seed` | needed for later 3-seed confirmation |
| `train_images`, `valid_images`, `test_images` | verifies comparable split sizes |
| `train_specimens`, `valid_specimens`, `test_specimens` | verifies grouped evaluation support |
| `train_test_group_overlap`, `valid_test_group_overlap` | direct leakage evidence; expected nonzero for random image split and zero for grouped split |
| `params_million` | model-size justification |
| `train_time_min` | practical cost |
| `epochs_ran`, `best_epoch_by_mask_map50` | convergence/early-stopping behavior |
| `full_test_mask_map50`, `full_test_mask_map50_95` | headline full-test segmentation metrics |
| `labeled_test_mask_map50`, `labeled_test_mask_map50_95` | diseased-image segmentation quality without healthy negatives diluting mAP interpretation |
| `healthy_test_mask_fp_rate` | primary safety/practicality metric for healthy shrimp |
| `healthy_test_fp_masks_per_image` | severity of false alarms when healthy errors occur |
| `labeled_test_disease_mask_miss_rate` | disease sensitivity at the chosen inference confidence |
| `labeled_test_mask_count_mae` | over/under-segmentation behavior |
| `healthy_aware_labeled_val_mask_map50` | validation-only model-selection score |
| `healthy_aware_labeled_test_mask_map50` | report-only sanity score, never used for selection |

Keep the existing inference confidence `PREDICT_CONF_FOR_COUNT = 0.25` for count, miss, and healthy-FP diagnostics in this matrix. Run confidence sweeps only for finalists, not for all twelve model/split runs.

### Model-Selection Rule

Report raw metrics first. Use the weighted score only to select a practical model:

```text
score =
    labeled_val_mask_map50
    - 0.05 * labeled_val_mask_count_mae
    - 0.15 * labeled_val_disease_mask_miss_rate
    - 0.10 * healthy_val_mask_fp_rate
```

This matches the existing baseline and Fourier notebooks, so old and new logs remain comparable.

### Fourier Auto-Tuning Plan

Do Fourier tuning only after selecting the YOLO variant from the grouped-specimen model comparison.

Use validation-only selection:

- objective: `healthy_aware_labeled_val_mask_map50`
- test set: evaluate only the clean baseline and final top Fourier candidates
- first-stage epochs: shorter tuning budget such as 40-60 epochs
- final-stage epochs: full baseline budget

Search space:

| Hyperparameter | Range | Notes |
|---|---:|---|
| `fourier_sigma` | 25 to 90 | larger values produce gentler low-pass separation |
| `fourier_alpha` | 0.15 to 0.65 | higher values boost frequency detail more aggressively |
| `fourier_mode` | `all_splits`, `train_only` | tune separately; `all_splits` is deployable preprocessing, `train_only` is augmentation/regularization |

Preferred implementation:

- use Optuna TPE if available on Kaggle
- fall back to a fixed candidate grid if Optuna is unavailable
- save every trial to `reports/fourier_auto_tuning_trials.csv`
- save final selected rows to `reports/fourier_auto_tuning_selected.csv`
