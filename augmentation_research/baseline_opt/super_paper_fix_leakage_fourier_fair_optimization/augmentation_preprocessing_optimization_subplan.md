# Research Sub-Plan: Leakage-Safe Augmentation and Preprocessing Optimization

This document defines the next optimization campaign after establishing the leakage-safe shrimp disease segmentation baseline.

The goal is no longer to prove only that specimen leakage matters. The next goal is:

> Given a fair specimen-grouped evaluation protocol, systematically search for augmentation and preprocessing strategies that improve shrimp disease segmentation beyond the current strong YOLO baseline.

This sub-plan should be read together with:

- [research-plan.md](research-plan.md)
- [model-sweep-strat-group-split/model_sweep_strat_group_findings.md](model-sweep-strat-group-split/model_sweep_strat_group_findings.md)

## 1. Current Experimental Anchor

The accepted optimization baseline is:

- Dataset: hand-labeled shrimp disease segmentation dataset.
- Split: stratified grouped-specimen split.
- Model: `yolo11n-seg`.
- Training augmentation: clean-light YOLO augmentation.
- Ultralytics hidden Albumentations hook: enabled.
- Primary seed: `42`.
- Confirmation: repeat final candidates across multiple grouped-split seeds.

Current baseline reference metrics from the grouped model sweep:

| Metric | Value |
|---|---:|
| Full test mask mAP50 | 0.481532 |
| Labeled-only test mask mAP50 | 0.512002 |
| Labeled-only test mask mAP50-95 | 0.167785 |
| Healthy test mask false positive rate | 0.243902 |
| Healthy-aware labeled test score | 0.459581 |

This is the baseline that new augmentation/preprocessing methods must beat.

## 2. Core Hypothesis

YOLO's default augmentation stack is already strong. Therefore, naive fixed preprocessing such as CLAHE, LBP, or deterministic Fourier enhancement may not reliably improve performance.

The more promising hypothesis is:

> Shrimp disease segmentation can improve under fair grouped-specimen evaluation when augmentation is tuned toward disease visibility, healthy false-positive control, and small-data generalization rather than simply adding stronger generic transformations.

## 3. Method Families To Test

### 3.1 YOLO Augmentation Policy Search

This should be the first priority because it uses transformations already integrated safely with YOLO detection/segmentation labels.

Candidate hyperparameters:

| Parameter | Why It Matters |
|---|---|
| `hsv_h` | Tests robustness to color tone differences. |
| `hsv_s` | Disease regions may be color-sensitive; saturation changes may help or hurt. |
| `hsv_v` | Handles lighting differences and exposure. |
| `degrees` | Shrimp orientation varies, but excessive rotation may create unrealistic views. |
| `translate` | Handles imperfect framing and body position shifts. |
| `scale` | Helps different shrimp sizes and crop distances. |
| `shear` | Mild viewpoint variation; high values may distort anatomy. |
| `perspective` | Usually keep low; shrimp images are not strongly perspective-driven. |
| `fliplr` | Likely valid because left-right orientation is not biologically class-defining. |
| `flipud` | Test carefully; valid only if upside-down shrimp orientation appears naturally. |
| `mosaic` | Strong small-data augmentation but may hurt final mask localization. |
| `close_mosaic` | Important to recover clean localization late in training. |
| `mixup` | May regularize but can blur disease cues. Use low values only. |
| `copy_paste` | Segmentation-specific and likely promising for rare disease patterns. |
| `copy_paste_mode` | Compare `flip` and `mixup`. |

Initial policy candidates:

| Experiment | Intention |
|---|---|
| `baseline_clean_light_yolo_aug` | Current reference baseline. |
| `mosaic_low_close` | Reduce mosaic strength and close it near the end. |
| `mosaic_default_close` | Keep mosaic but improve late localization. |
| `mosaic_high_close` | Test whether more mixed context helps. |
| `copy_paste_low_flip` | Low-risk segmentation-specific augmentation. |
| `copy_paste_mid_flip` | Stronger same-image copy-paste. |
| `copy_paste_low_mixup` | Cross-image copy-paste. |
| `copy_paste_mid_mixup` | Stronger cross-image copy-paste. |
| `copy_paste_close_mosaic` | Combine the most promising segmentation-specific augmentations. |
| `hsv_mild` | Mild color robustness. |
| `hsv_strong` | Stronger photometric variation. |
| `scale_translate_mild` | Framing and size robustness. |
| `flipud_low` | Test whether vertical orientation improves generalization. |
| `mixup_low` | Regularization without overwhelming lesion signals. |
| `best_combined_candidate` | Combine best observed settings after screening. |

Expected contribution if successful:

> A leakage-safe YOLO augmentation policy improves shrimp disease segmentation while reducing or preserving healthy false positives.

### 3.2 Segmentation-Safe RandAugment-Style Photometric Policy

Ultralytics `auto_augment='randaugment'` is classification-specific, so it should not be treated as directly valid for segmentation.

Instead, implement a segmentation-safe RandAugment-style policy using image-only photometric transforms that do not alter masks:

- brightness shift
- contrast shift
- gamma correction
- hue/saturation/value shift
- RGB channel shift
- grayscale/desaturation
- sharpen
- blur
- Gaussian noise
- ISO noise
- JPEG compression

Candidate policies:

| Experiment | Intention |
|---|---|
| `photometric_mild` | Low-risk color/lighting variation. |
| `photometric_medium` | More diverse color and contrast variation. |
| `noise_blur_compression` | Robustness to acquisition artifacts. |
| `sharpen_contrast` | Disease texture emphasis without fixed CLAHE. |
| `randaug_photo_n2_mild` | RandAugment-style random two-transform policy. |
| `randaug_photo_n3_medium` | Stronger random photometric policy. |

Expected contribution if successful:

> Mask-safe photometric policy search improves real-world robustness without corrupting segmentation labels.

### 3.3 Copy-Paste Focused Sweep

Copy-paste is especially relevant because the task is segmentation and the dataset is small.

Start with YOLO built-in copy-paste before implementing custom disease-patch copy-paste.

Focused sweep:

| Parameter | Values |
|---|---|
| `copy_paste` | `0.05`, `0.10`, `0.20`, `0.30` |
| `copy_paste_mode` | `flip`, `mixup` |
| `close_mosaic` | `0`, `10`, `20` |
| `mosaic` | baseline value, reduced value |

Escalation only if built-in copy-paste helps:

- Disease-aware copy-paste: paste BG/WSSV mask regions between training images.
- Class-balanced copy-paste: oversample the weaker class or co-infection cases.
- Healthy-preserving copy-paste: avoid pasting lesions into healthy images unless explicitly designing synthetic disease cases.

Risk:

- Unnatural disease placement can harm biological plausibility.
- Disease-aware copy-paste must be visually audited before inclusion in the paper.

### 3.4 Frequency and Wavelet Augmentation

Fourier should remain in the study, but not as the only improvement path.

Current evidence:

- Deterministic Fourier all-splits did not beat the hook-enabled clean baseline.
- Train-only Fourier showed small mAP-oriented gains in some settings but worsened or failed to improve healthy-aware score.

Better future variants:

| Method | Description |
|---|---|
| Stochastic Fourier train-only | Randomize alpha/sigma per image or per run. |
| Low-alpha Fourier | Avoid over-amplifying false disease-like texture. |
| Frequency dropout | Randomly suppress bands to improve robustness. |
| Fourier style mixing | Modify low-frequency appearance to simulate domain/camera variation. |
| Wavelet detail enhancement | Enhance multi-scale local texture without global frequency distortion. |
| Stochastic wavelet augmentation | Randomize wavelet strength during training only. |

Priority:

- Run only after YOLO augmentation policy search.
- Keep as a secondary branch unless it beats the healthy-aware baseline.

### 3.5 Background Removal / Shrimp Isolation

Background removal may reduce healthy false positives if the model is distracted by non-shrimp texture.

Variants:

| Variant | Risk |
|---|---|
| Train and test on background-removed images | High risk if removal artifacts differ across splits. |
| Train with mixed original and background-removed images | Lower risk, acts as augmentation. |
| Background removal only as train-time augmentation | Safest first test. |

Recommendation:

- Do not prioritize before YOLO augmentation policy and copy-paste.
- If tested, report healthy FP rate carefully.

## 4. Evaluation Metrics

A method should not be selected by mask mAP50 alone.

Primary metrics:

| Metric | Priority | Meaning |
|---|---:|---|
| Labeled-only test mask mAP50 | High | Segmentation quality on diseased images only. |
| Full test mask mAP50 | High | Segmentation quality including healthy/background images. |
| Healthy test mask false positive rate | Very high | Fraction of healthy images where model predicts disease. |
| Healthy-aware test score | Very high | Composite score balancing disease segmentation and healthy specificity. |
| Mask mAP50-95 | Medium | Boundary/quality strictness beyond loose overlap. |
| Disease missed image rate | High | Fraction of diseased images with no useful disease prediction. |
| Count MAE | Medium | Whether the model over/under-predicts mask instances. |

Selection rule:

1. Rank candidates by validation healthy-aware score.
2. Reject candidates that raise healthy false positives substantially unless mAP gain is very large.
3. Use test metrics only for final reporting.
4. Confirm top candidates across multiple seeds.

Minimum improvement target:

- `+0.02` absolute labeled-only test mask mAP50, or
- `+0.02` absolute healthy-aware test score, or
- lower healthy FP rate with no meaningful mAP loss.

Strong improvement target:

- `+0.04` to `+0.06` absolute labeled-only test mask mAP50 under grouped-specimen split.

## 5. Recommended Experiment Order

### Phase A: Lock Baseline And Reporting Contract

Status: mostly done.

Required:

- Confirm baseline row and metrics in the paper table.
- Confirm grouped-split seed 42 manifest.
- Confirm exact Ultralytics version.
- Confirm hidden Albumentations hook is enabled.

### Phase B: YOLO Augmentation Policy Screening

Run 12 to 20 candidates on:

- model: `yolo11n-seg`
- split: grouped-specimen seed 42
- validation selection: healthy-aware score
- test evaluation: final only

Deliverables:

- Colab notebook for policy search.
- CSV summary.
- Paper-ready table with baseline and candidates.
- Best checkpoint for top candidates.

### Phase C: Copy-Paste Focused Sweep

Run only if Phase B suggests copy-paste is promising.

Deliverables:

- Copy-paste sweep CSV.
- Visual examples of augmented training images.
- Comparison against baseline and best Phase B policy.

### Phase D: Photometric RandAugment-Style Screening

Run segmentation-safe photometric policies.

Deliverables:

- CSV summary.
- Robustness table if noisy/compressed test sets are generated.
- Visual examples of transform strength.

### Phase E: Frequency/Wavelet Revisit

Run only focused variants:

- best train-only Fourier region
- stochastic Fourier train-only
- one wavelet detail-enhancement policy
- optional frequency dropout

Deliverables:

- CSV summary.
- Decision whether frequency methods remain in the final paper or appendix.

### Phase F: Multi-Seed Confirmation

Take top 2 or 3 methods:

- baseline
- best YOLO policy
- best custom policy

Run across at least 3 grouped-specimen seeds.

Report:

- mean
- standard deviation
- best/worst
- confidence interval if useful

## 6. Notebook Requirements

New notebooks should be Colab-ready and RUN ALL compatible.

Required notebook features:

- Download Roboflow hand-labeled dataset.
- Recreate grouped-specimen stratified split.
- Use deterministic split manifests where possible.
- Force the same Ultralytics version as prior runs.
- Print exact training config before each run.
- Save per-experiment YAML/config snapshot.
- Save `best.pt`, `last.pt`, `results.csv`, train args, validation metrics, test metrics.
- Export a zip file containing reports and checkpoints.
- Include boolean flags to enable/disable experiment groups.
- Include a smoke-test mode.

Suggested notebook names:

- `[AIP491_01]yolo_seg_yolo_aug_policy_search_group_stratified_colab.ipynb`
- `[AIP491_01]yolo_seg_copy_paste_sweep_group_stratified_colab.ipynb`
- `[AIP491_01]yolo_seg_photometric_randaug_group_stratified_colab.ipynb`
- `[AIP491_01]yolo_seg_frequency_wavelet_revisit_group_stratified_colab.ipynb`

## 7. Files To Download After Each Training Session

Always download:

- main summary CSV
- paper-table CSV
- partial CSV if training is interrupted
- split manifests
- config snapshots
- `best.pt` for each completed experiment
- `results.csv` for each completed experiment
- generated plots/confusion/PR curves if available
- final export zip

Recommended folder structure:

```text
augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/
  augmentation_policy_search/
    yolo_aug_policy/
    copy_paste_sweep/
    photometric_randaug/
    frequency_wavelet_revisit/
    multi_seed_confirmation/
```

## 8. Paper Framing

If optimization succeeds:

> After correcting specimen leakage, we optimize segmentation augmentation under the fair grouped protocol. A tuned YOLO augmentation/copy-paste policy improves disease segmentation while controlling false positives on healthy shrimp.

If optimization only gives small gains:

> Under fair specimen-grouped evaluation, many commonly proposed preprocessing enhancements do not reliably improve over YOLO's built-in augmentation. This demonstrates the importance of evaluating optimization methods under leakage-safe protocols, because gains observed under easier splits may not transfer to unseen specimens.

Both outcomes are publishable, but a confirmed improvement is stronger.

## 9. Current Recommendation

The next notebook should be:

> YOLO augmentation policy search under grouped-specimen split.

Rationale:

- It is closest to the current strong baseline.
- It is segmentation-safe.
- It is easier to justify than handcrafted preprocessing.
- It directly tests whether the existing YOLO augmentation hook can be tuned rather than replaced.
- It may reveal a meaningful improvement path through copy-paste, close-mosaic, and photometric strength control.

Do not run another broad Fourier grid until the YOLO augmentation policy search is complete.

