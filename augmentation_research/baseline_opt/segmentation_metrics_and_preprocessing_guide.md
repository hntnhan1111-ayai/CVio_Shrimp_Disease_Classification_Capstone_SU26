# Shrimp Disease Segmentation: Metrics and Preprocessing Guide

This guide documents the evaluation metrics and preprocessing/augmentation methods used in the leakage-safe YOLO segmentation experiments for the hand-labeled `shrimpdiseasebd` dataset.

The goal is to make the experiment background understandable to a new reader, and to give future contributors a practical path for improving the model without accidentally optimizing the wrong thing.

## Experiment Context

The current segmentation task uses hand-labeled shrimp disease masks. A single shrimp specimen can appear in multiple images, so train/validation/test splitting must be specimen-grouped. If images of the same shrimp appear in multiple splits, the model can memorize specimen-specific appearance and produce inflated metrics.

The current clean baseline is:

- YOLO segmentation model: `yolo11n-seg.pt`
- grouped split by shrimp specimen
- disease-aware stratification across split groups
- clean-light YOLO augmentation enabled
- hidden Ultralytics Albumentations disabled
- deterministic validation/test evaluation

Relevant notebooks:

- Clean baseline: `[AIP491_01]yolo_seg_11n_clean_baseline_fix_leakage.ipynb`
- Preprocessing ablation: `[AIP491_01]yolo_seg_11n_preproc_ablation_group_stratified.ipynb`
- Fourier sweep: `[AIP491_01]yolo_seg_11n_fourier_sweep_group_stratified.ipynb`

## Metric Priority

The highest-priority metric is not a single number. The task has two competing needs:

1. Detect and segment diseased shrimp regions accurately.
2. Avoid predicting disease on healthy shrimp.

For research reporting, segmentation mAP is the headline metric. For practical deployment, healthy false positives and missed diseased images become equally important, because an aquaculture user needs both useful disease detection and trust that healthy animals are not constantly flagged.

Recommended priority:

| Priority | Metric | Why it matters |
|---:|---|---|
| 1 | Labeled-only diseased test mask mAP50 | Measures segmentation performance on images that actually contain disease masks. |
| 1 | Healthy test mask false-positive rate | Measures whether the model invents disease masks on healthy shrimp. |
| 2 | Full test mask mAP50 | Measures overall test performance including healthy images. |
| 2 | Missed diseased image rate | Practical recall proxy: how often a diseased image gets no mask. |
| 2 | Mask count MAE / predicted masks per GT | Practical count quality and over/under-prediction behavior. |
| 3 | mAP50-95 | Stricter mask localization quality across IoU thresholds. |
| 3 | Train/validation loss gap | Diagnoses overfitting and regularization problems. |

## Core Metrics

### Mask mAP50

Mask mAP50 is mean average precision for segmentation masks at IoU threshold 0.50.

It asks: when the model predicts a disease mask, does that mask overlap the ground-truth mask by at least 50%, and is the confidence ranking good?

Why it matters:

- This is the main segmentation success metric.
- It reflects both detection and mask quality.
- It is easier to improve and interpret than mAP50-95 on small datasets.

Limitations:

- It does not directly tell us how often healthy shrimp are falsely flagged.
- It can hide poor behavior at deployment thresholds because mAP evaluates across confidence thresholds.

How to improve:

- better specimen-safe split quality and class balance
- clean-light augmentation to reduce overfitting
- preprocessing that improves disease texture visibility, such as Fourier detail boosting
- model size scaling if compute allows
- better labels for small or ambiguous disease regions

### Mask mAP50-95

Mask mAP50-95 averages mask AP over IoU thresholds from 0.50 to 0.95.

It asks: are predicted masks not only roughly correct, but tightly aligned with the disease region?

Why it matters:

- It is a stricter segmentation quality metric.
- It is useful if precise disease area measurement matters.

Limitations:

- It is often low for small datasets and noisy masks.
- It can punish reasonable predictions when labels are coarse or boundary ambiguity is high.

How to improve:

- improve mask boundary consistency
- increase image resolution if disease patterns are small
- tune augmentation so geometry is not distorted
- consider larger YOLO segmentation variants

### Box mAP50

Box mAP50 evaluates bounding boxes around disease regions.

Why it matters:

- It shows whether the model can localize disease roughly, even if masks are imperfect.
- A model with strong box mAP but weak mask mAP may need better mask-specific training or cleaner mask labels.

How to improve:

- similar to mask mAP improvements
- ensure polygon masks are valid and not overly fragmented
- tune image preprocessing for region contrast

### Full Test Metrics

Full test metrics are computed on the complete test set, including labeled diseased images and healthy images.

Why it matters:

- This reflects the real test distribution.
- It includes the effect of healthy images on evaluation.

Limitations:

- If many healthy images have no labels, full test mAP alone does not fully explain false positives.
- It should be reported with healthy false-positive metrics.

### Labeled-Only Diseased Metrics

Labeled-only metrics are computed on test images that contain at least one disease mask.

Why it matters:

- They isolate disease segmentation ability.
- They prevent healthy images from obscuring whether the model can segment actual disease.

Limitations:

- They do not measure healthy false alarms.
- A model can score well here and still be bad in practice if it frequently predicts disease on healthy shrimp.

### Healthy Mask False-Positive Rate

Healthy mask false-positive rate is the fraction of healthy images where the model predicts at least one disease mask.

Formula:

```text
healthy mask FP rate = healthy images with >=1 predicted mask / total healthy images
```

What it represents:

- How often the model falsely flags healthy shrimp as diseased.
- A direct trust and usability metric.

Why it is significant:

- In shrimp health screening, false positives can cause unnecessary alarm, manual rechecking, or incorrect treatment decisions.
- A model that detects disease well but constantly flags healthy shrimp is not practically reliable.
- In this project, Fourier preprocessing improved disease mAP, but healthy false positives became a key tradeoff.

Priority:

- Very high, especially for deployment.
- It should be evaluated alongside labeled-only diseased mAP.

How to improve:

- tune inference confidence threshold
- include enough healthy images in train/validation/test
- use healthy-aware validation scoring
- tune Fourier strength to avoid amplifying normal shell texture as disease-like texture
- add hard-negative mining: retrain with healthy images where the model falsely predicted masks
- calibrate confidence after training

### Healthy False-Positive Masks Per Image

This is the average number of false-positive masks predicted per healthy image.

Formula:

```text
healthy FP masks per image = total predicted masks on healthy images / total healthy images
```

Why it matters:

- The FP rate only says whether an image has any false positive.
- FP masks per image says how severe the false-positive behavior is.

Example:

- FP rate 0.30 and 0.30 masks/image means most affected images have one false mask.
- FP rate 0.30 and 1.50 masks/image means affected healthy images may be heavily over-segmented.

How to improve:

- increase confidence threshold
- reduce aggressive texture enhancement
- train with harder healthy negatives
- add regularization or clean augmentation

### Healthy Average FP Confidence

This is the average confidence of false-positive detections on healthy images.

Why it matters:

- Low-confidence false positives may be manageable with threshold tuning.
- High-confidence false positives indicate a deeper model confusion.

How to improve:

- confidence calibration
- hard-negative training
- better preprocessing strength
- add healthy examples that resemble diseased texture but are truly normal

### Missed Diseased Image Rate

Missed diseased image rate is the fraction of labeled diseased images where the model predicts no masks.

Formula:

```text
missed diseased image rate = diseased images with 0 predicted masks / diseased images
```

Why it matters:

- It is a practical recall proxy.
- A high value means the model may fail to alert users when disease is present.

How to improve:

- lower inference confidence threshold
- use preprocessing that increases disease visibility
- improve class balance
- add examples of subtle disease patterns
- tune model size or training duration

### Mask Count MAE

Mask count MAE is the mean absolute error between the number of ground-truth masks and predicted masks per image.

Formula:

```text
mask count MAE = mean(abs(predicted mask count - ground-truth mask count))
```

Why it matters:

- It measures over-prediction and under-prediction at the image level.
- It is useful when each image can contain multiple disease regions.

How to improve:

- tune confidence threshold
- reduce false positives
- improve annotations for fragmented disease regions
- use hard-negative mining

### Predicted Masks Per Ground Truth

This is the total predicted masks divided by total ground-truth instances on labeled diseased images.

Formula:

```text
pred masks per GT = total predicted masks / total ground-truth masks
```

Interpretation:

- Around 1.0: count is balanced.
- Above 1.0: model tends to over-predict.
- Below 1.0: model tends to under-predict or miss disease regions.

### Healthy-Aware Labeled Validation Score

The ablation notebooks use a custom validation score to select the best run:

```text
healthy-aware score =
    labeled_val_mask_map50
    - COUNT_PENALTY_WEIGHT * labeled_val_mask_count_mae
    - DISEASE_MISS_PENALTY_WEIGHT * labeled_val_disease_box_miss_rate
    - HEALTHY_FP_PENALTY_WEIGHT * healthy_val_mask_fp_rate
```

Why it matters:

- It prevents selecting a model purely because it has good mAP while being unusable on healthy images.
- It makes the validation selection closer to the real project objective.

Limitations:

- The penalty weights are hand-chosen.
- Different deployment priorities may require different weights.

How to improve:

- tune penalty weights based on real cost of false positives vs missed disease
- report both raw metrics and the composite score
- avoid using test metrics to choose checkpoints

### Train/Validation Segmentation Loss Gap

The segmentation loss gap is:

```text
last validation segmentation loss - last training segmentation loss
```

Why it matters:

- A very large positive gap suggests overfitting.
- In earlier bare-baseline runs, turning off clean augmentation caused strong overfitting and poor test performance.

How to improve:

- use clean-light augmentation
- increase patience only if validation is still improving
- use stronger regularization carefully
- add more data or more diverse train examples

## Confidence Thresholds

mAP is not the same as deployment prediction quality. YOLO mAP is evaluated across confidence thresholds, while deployment uses a chosen confidence threshold such as 0.25 or 0.40.

Observed behavior from the Fourier run:

- Lower threshold improves disease recall but increases healthy false positives.
- Higher threshold reduces healthy false positives but increases missed diseased images.

Recommended practice:

- Report mAP using standard YOLO validation.
- Separately report threshold sweeps for healthy false positives and missed diseased images.
- Choose deployment confidence based on the acceptable tradeoff.

Example operating points:

| Threshold | Behavior |
|---:|---|
| 0.25 | Better recall, more false positives. Good if missing disease is costly. |
| 0.40 | More conservative, fewer healthy false positives, more missed disease. |
| 0.50+ | Often too strict for this dataset unless false positives are much more costly than misses. |

## Methods

### Clean-Light YOLO Augmentation

Clean-light augmentation is now treated as part of the baseline because the bare model overfit and performed worse.

Current settings:

```python
fliplr = 0.5
hsv_h = 0.01
hsv_s = 0.35
hsv_v = 0.20
translate = 0.05
scale = 0.20
```

Disabled settings:

```python
mosaic = 0.0
mixup = 0.0
cutmix = 0.0
copy_paste = 0.0
erasing = 0.0
auto_augment = None
```

Why it matters:

- It regularizes the model without heavily changing the segmentation task.
- It better reflects a fair baseline than a fully bare training run.

Risk:

- Too much augmentation could create unrealistic shrimp appearances or damage mask alignment.

Improvement approach:

- keep as baseline unless a future run proves a better default
- tune only after preprocessing experiments are understood

### Fourier High-Pass Detail Boost

Fourier transform decomposes an image into frequency components. Low frequencies represent smooth color and illumination. High frequencies represent edges, texture, and fine detail.

In this project, Fourier high-pass detail boost works by:

1. Transforming each image channel to the frequency domain.
2. Estimating the low-frequency component with a Gaussian low-pass filter.
3. Subtracting that low-frequency component from the original channel to obtain high-frequency detail.
4. Adding a scaled version of that detail back to the image.

Conceptually:

```text
enhanced image = original image + alpha * high-frequency detail
```

Why it is useful here:

- Shrimp disease signs can be texture-like or contrast-subtle.
- Frequency boosting may make disease patterns more visible to YOLO.
- The experiments showed Fourier + clean augmentation improved mask mAP over the clean-light baseline.

Current key hyperparameters:

```python
fourier_sigma
fourier_alpha
fourier_mode
```

#### `fourier_sigma`

Controls the Gaussian low-pass filter size.

Interpretation:

- Smaller sigma: stronger/broader detail extraction.
- Larger sigma: more conservative detail extraction.

Suggested sweep:

```python
fourier_sigma = 35
fourier_sigma = 50
fourier_sigma = 70
```

#### `fourier_alpha`

Controls how much high-frequency detail is added back.

Interpretation:

- Lower alpha: weaker enhancement, likely fewer artifacts.
- Higher alpha: stronger enhancement, possibly higher mAP but more false positives.

Suggested sweep:

```python
fourier_alpha = 0.30
fourier_alpha = 0.40
fourier_alpha = 0.50
```

#### `fourier_mode`

Controls where Fourier preprocessing is applied.

Current modes:

| Mode | Description | Research meaning |
|---|---|---|
| `all_splits` | Apply Fourier to train, validation, and test images. | Tests whether a Fourier-transformed image domain improves segmentation. |
| `train_only` | Apply Fourier only to train images. | Tests Fourier as regularization/augmentation while evaluating natural images. |
| `train_copy` | Keep original train images and add Fourier-enhanced copies. | Tests Fourier as data augmentation while preserving the natural image domain. |

Recommended next sweep:

```python
sigma=50, alpha=0.30, mode=all_splits
sigma=50, alpha=0.40, mode=all_splits
sigma=70, alpha=0.30, mode=all_splits
sigma=50, alpha=0.30, mode=train_only
```

Why this sweep:

- The previous `sigma=50, alpha=0.50` result was strong but had healthy false-positive concerns.
- Lower alpha may keep the mAP improvement while reducing normal texture amplification.
- `train_only` tests whether Fourier can help without requiring transformed validation/test images.

Fourier failure modes:

- Over-enhancing normal shrimp shell texture.
- Increasing false positives on healthy shrimp.
- Creating a domain shift if train and test preprocessing differ.
- Boosting label noise or irrelevant image artifacts.

How to improve Fourier results:

- reduce `alpha`
- increase `sigma`
- use train-copy augmentation instead of replacing all images
- combine with healthy-aware validation selection
- add healthy hard negatives from Fourier false-positive cases

### CLAHE

CLAHE stands for Contrast Limited Adaptive Histogram Equalization. It improves local contrast while limiting contrast amplification.

In this project, CLAHE is applied to the luminance channel in LAB color space.

Key hyperparameters:

```python
clahe_clip_limit
clahe_tile_grid_size
```

#### `clahe_clip_limit`

Controls how strongly local contrast is enhanced.

Suggested values:

```python
1.5
2.0
3.0
```

Lower values are safer. Higher values may expose disease regions but can also exaggerate noise.

#### `clahe_tile_grid_size`

Controls the local region size for contrast equalization.

Common value:

```python
(8, 8)
```

Why it is useful:

- Disease regions may be low contrast.
- CLAHE can make local discoloration easier to identify.

Observed behavior:

- CLAHE alone was more conservative on healthy false positives.
- CLAHE did not beat Fourier on segmentation mAP.
- CLAHE + Fourier did not beat Fourier alone in the current run.

How to improve:

- try weak CLAHE with lower clip limit
- avoid combining strong CLAHE with strong Fourier
- evaluate healthy false positives carefully

### LBP Texture Blend

LBP means Local Binary Pattern. It encodes local texture by comparing each pixel with neighboring pixels.

In this project, LBP is blended into the luminance channel:

```text
new luminance = (1 - alpha) * original luminance + alpha * LBP texture map
```

Key hyperparameter:

```python
LBP_BLEND_ALPHA
```

Why it might help:

- Some disease signs may be texture patterns rather than color patterns.

Observed behavior:

- The current LBP blend underperformed.

Possible reasons:

- LBP may be too artificial for YOLO pretraining assumptions.
- It may destroy useful color/luminance cues.
- It may amplify irrelevant shell texture.

Recommendation:

- Low priority for now.
- Only revisit with weaker blend values or train-copy mode.

Suggested values if revisited:

```python
LBP_BLEND_ALPHA = 0.05
LBP_BLEND_ALPHA = 0.10
LBP_BLEND_ALPHA = 0.15
```

### Photometric RandAug Train-Only Copies

RandAug creates additional training images with randomized photometric transformations. The current idea is to add augmented train-only copies while leaving validation and test natural.

Why it might help:

- More variation in brightness, contrast, and color can improve robustness.

Observed behavior:

- The previous RandAug run increased healthy false positives.

Recommendation:

- Low priority unless made milder.
- Avoid aggressive photometric distortion.
- Do not use transformations that create disease-like artifacts on healthy shrimp.

How to improve:

- reduce augmentation magnitude
- apply only to labeled diseased images or only to healthy hard negatives depending on failure mode
- inspect augmented images visually before full training

## Recommended Improvement Workflow

1. Keep the clean-light grouped-stratified baseline fixed.
2. Run Fourier parameter sweep with the same split and model.
3. Select candidates using validation healthy-aware score, not test metrics.
4. Report final test metrics only after selection.
5. For top candidates, run confidence sweeps:
   - healthy false-positive rate
   - healthy FP masks per image
   - missed diseased image rate
   - mask count MAE
6. Inspect visual predictions on:
   - true diseased images
   - healthy false-positive images
   - subtle disease images
7. Use error cases for the next iteration:
   - add hard negatives
   - tune Fourier strength
   - consider confidence calibration

## Reporting Template

For each experiment, report:

```text
Experiment name:
Split policy:
Preprocessing:
Train augmentation:
Model:
Epochs/patience:

Validation:
- labeled_val_mask_map50
- healthy_val_mask_fp_rate
- healthy_aware_labeled_val_mask_map50

Test:
- full_test_mask_map50
- labeled_test_mask_map50
- healthy_test_mask_fp_rate at conf=0.25
- missed diseased image rate at conf=0.25
- mask count MAE at conf=0.25

Notes:
- visual failure modes
- whether improvement is mAP-focused, FP-focused, or balanced
```

## Current Practical Takeaways

- Clean-light augmentation should be treated as the baseline, not a separate improvement.
- Fourier detail boosting is the strongest current improvement signal for disease segmentation mAP.
- Fourier also increases the importance of healthy false-positive monitoring.
- CLAHE is safer for healthy false positives but weaker for segmentation mAP.
- LBP and RandAug are lower priority based on current evidence.
- The next best research step is a Fourier sweep over `sigma`, `alpha`, and application mode.

