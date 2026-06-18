# Deep Research Update: Practical Augmentation and Preprocessing Optimization Plan

Date: 2026-06-15

This document updates the earlier 22-method research package using the current experimental evidence from the hand-labeled shrimp disease segmentation project. The goal is not to produce a broad literature list. The goal is to decide what is still worth testing to beat the current leakage-safe baseline.

## 1. Current Optimization Anchor

Accepted baseline:

| Item | Current Contract |
|---|---|
| Dataset | Hand-labeled shrimp disease instance segmentation |
| Model | `yolo11n-seg.pt` |
| Split | Stratified grouped-specimen split |
| Main seed | `42` |
| YOLO hidden Albumentations hook | Enabled |
| Explicit train augmentation | Clean-light YOLO augmentation |
| Full test mask mAP50 | `0.481532` |
| Labeled-only test mask mAP50 | `0.512002` |
| Labeled-only test mask mAP50-95 | `0.167785` |
| Healthy test FP rate | `0.243902` |
| Healthy-aware labeled test score | `0.459581` |

Baseline explicit train augmentation:

```python
{
    "auto_augment": None,
    "erasing": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.35,
    "hsv_v": 0.20,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.20,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": 0.0,
    "bgr": 0.0,
}
```

Any new method should be evaluated against this contract unless the experiment explicitly studies augmentation-policy interaction.

## 2. Correction To Earlier Research Expectations

The earlier research package expected `+0.03` to `+0.07` labeled-only mAP50 improvement from Phase B/C methods. That expectation is now too optimistic.

Observed project evidence:

| Method Family | Current Evidence |
|---|---|
| Photometric copies | Usually worse test transfer than baseline. |
| HSV mild/stronger color jitter | Worse than baseline in recovered logs. |
| Fourier | Not complementary to the hook-enabled baseline. |
| Wavelet | Reported worse than baseline. |
| Copy-paste flip | No visible benefit in current runs. |
| Copy-paste mixup | Tiny labeled mAP gain, much worse healthy FP. |
| SimAM+CA architecture | Sensitive to augmentation settings; weak under claimed baseline-like rerun. |

Updated hypothesis:

> The baseline is not weak. The optimization problem is now a narrow search for augmentations that improve unseen-specimen generalization without increasing healthy false positives.

This means the next experiments should be smaller, more controlled, and more failure-mode-driven.

## 3. Verified Source Notes

The older documents contain useful method ideas, but some claims should be treated as generated planning notes rather than citation-ready facts. The following sources are verified enough to guide implementation:

| Topic | Source | Practical Takeaway |
|---|---|---|
| Ultralytics augmentation parameters | Ultralytics YOLO Data Augmentation docs: https://docs.ultralytics.com/guides/yolo-data-augmentation/ | Use official parameter behavior for HSV, geometric, mosaic, mixup, cutmix, copy-paste, auto-augment, custom Albumentations. |
| Mosaic and close-mosaic | Ultralytics docs | Mosaic can improve context/small objects but can make training harder; `close_mosaic` disables it near the end. |
| Copy-paste for segmentation | Ghiasi et al., Simple Copy-Paste: https://arxiv.org/abs/2012.07177 | Copy-paste is relevant for instance segmentation and rare object classes, but must be tuned for this dataset. |
| RandAugment | Cubuk et al.: https://arxiv.org/abs/1909.13719 | Good general augmentation principle, but Ultralytics documents `auto_augment` as classification-specific; do not assume segmentation benefit. |
| GridMask | Chen et al.: https://arxiv.org/abs/2001.04086 | Structured information dropout can help detection/segmentation, but disease masks may be small, so conservative use is required. |
| Albumentations | Buslaev et al.: https://arxiv.org/abs/1809.06839 | Useful for custom image/mask-safe transforms; in Ultralytics custom Albumentations replace default hook transforms, so handle carefully. |
| AugMix | Hendrycks et al.: https://arxiv.org/abs/1912.02781 | Robustness-oriented idea, but mainly classification; use only as inspiration for mild photometric robustness, not direct segmentation claim. |
| ClassMix/DACS | ClassMix: https://arxiv.org/abs/2007.07936, DACS: https://arxiv.org/abs/2007.08702 | Segmentation-aware mixing should respect mask structure; custom implementation cost is non-trivial. |
| Hard examples / imbalance | Focal Loss: https://arxiv.org/abs/1708.02002 | False positives are a hard-negative problem; focus training on hard healthy cases may be more useful than generic image enhancement. |
| TTA / uncertainty | TAAL: https://arxiv.org/abs/2301.06624 | Test-time augmentation can expose uncertainty and sometimes improve segmentation, but should be treated as inference-time optimization. |
| Diffusion augmentation | SatSynth: https://arxiv.org/abs/2403.16605 | Powerful but too expensive and out-of-scope unless simple policies plateau and more compute is available. |

## 4. Method Feasibility Assessment

### Tier 1: Highest ROI, Test Next

#### 4.1 Low-Mosaic With Close-Mosaic

Why:

- Baseline has mosaic disabled.
- Ultralytics docs state mosaic can help context and small objects but can make training harder.
- A low probability may improve generalization without severely damaging localization.

Risks:

- Too much mosaic may create unrealistic shrimp/disease context.
- Disease masks may become small or fragmented after mosaic.

Recommended candidates:

```python
[
    {
        "key": "mosaic_0p10_close20",
        "mosaic": 0.10,
        "close_mosaic": 20,
    },
    {
        "key": "mosaic_0p25_close20",
        "mosaic": 0.25,
        "close_mosaic": 20,
    },
    {
        "key": "mosaic_0p25_close30",
        "mosaic": 0.25,
        "close_mosaic": 30,
    },
]
```

Decision:

- Keep if labeled-test mAP50 improves by at least `+0.010` and healthy FP stays at or below `0.264`.
- Drop if mAP gains only on validation but not test.

#### 4.2 Copy-Paste With Low Mosaic

Why:

- Copy-paste is directly relevant to instance segmentation.
- Current copy-paste mixup had a small mAP signal but increased healthy FP.
- It may need low mosaic or a different probability to become useful.

Risks:

- Mixup can paste disease into unnatural context and increase healthy false positives.
- Flip mode may produce no practical effect when mosaic is off.

Recommended candidates:

```python
[
    {
        "key": "cp_0p05_mixup_mosaic_0p10",
        "copy_paste": 0.05,
        "copy_paste_mode": "mixup",
        "mosaic": 0.10,
        "close_mosaic": 20,
    },
    {
        "key": "cp_0p10_mixup_mosaic_0p10",
        "copy_paste": 0.10,
        "copy_paste_mode": "mixup",
        "mosaic": 0.10,
        "close_mosaic": 20,
    },
    {
        "key": "cp_0p05_flip_mosaic_0p25",
        "copy_paste": 0.05,
        "copy_paste_mode": "flip",
        "mosaic": 0.25,
        "close_mosaic": 20,
    },
]
```

Decision:

- Reject if healthy FP rate exceeds baseline by more than `0.05`.
- Treat mAP gain below `+0.005` as noise.

#### 4.3 Conservative HSV + Geometry Middle Policy

Why:

- Teammate results suggest explicit augmentation policy can materially change performance.
- Baseline is conservative; teammate policy was much stronger.
- Test a middle region rather than jumping to aggressive settings.

Recommended candidates:

```python
[
    {
        "key": "hsv_geo_middle_no_rotate",
        "hsv_h": 0.015,
        "hsv_s": 0.45,
        "hsv_v": 0.25,
        "translate": 0.08,
        "scale": 0.30,
        "degrees": 0.0,
    },
    {
        "key": "hsv_geo_middle_rotate3",
        "hsv_h": 0.015,
        "hsv_s": 0.45,
        "hsv_v": 0.25,
        "translate": 0.08,
        "scale": 0.30,
        "degrees": 3.0,
    },
]
```

Decision:

- Keep only if test improves, not just validation.
- Do not include `flipud` until visual inspection confirms upside-down shrimp are plausible in deployment.

### Tier 2: Targeted Error-Driven Optimization

#### 4.4 Healthy Hard-Negative Mining

Why:

- Healthy FP is one of the main weaknesses.
- Generic preprocessing often increases false positives by amplifying texture.
- Hard-negative training targets the actual error mode.

Implementation options:

1. Run current baseline on train/valid healthy images at `conf=0.25`.
2. Collect healthy images with predicted masks.
3. Create duplicated training copies of these healthy images with empty labels.
4. Retrain with baseline augmentation.

Candidate:

```python
{
    "key": "healthy_hard_negative_2x",
    "hard_negative_source": "baseline_false_positive_healthy_train_valid",
    "duplicate_factor": 2,
}
```

Evaluation:

- Primary: healthy FP rate.
- Secondary: labeled-only mAP50.
- Reject if healthy FP improves only by making the model miss disease.

This is likely the best next method if YOLO policy tuning fails.

#### 4.5 Threshold and Calibration Sweep

Why:

- mAP evaluates across thresholds, but practical use needs one operating threshold.
- Some methods may be acceptable at a higher confidence threshold even if `conf=0.25` FP is high.

Run for baseline and finalists:

```python
CONF_GRID = [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80]
```

Metrics:

- healthy FP rate
- healthy FP masks/image
- missed diseased image rate
- mask count MAE
- predicted masks per GT

This should be done before declaring a method unusable if its mAP is strong but FP is high.

### Tier 3: Custom Albumentations, Only If Needed

Ultralytics supports custom Albumentations transforms through Python API, but the official docs state that provided custom transforms replace the default Albumentations transforms while standard YOLO augmentations remain active.

Therefore, this is not a simple add-on. It changes the hook layer.

Recommended only as controlled study:

| Policy | Transforms | Purpose |
|---|---|---|
| `custom_albu_minimal` | very mild `RandomBrightnessContrast`, optional `CLAHE` p<=0.15 | Test if controlled hook replacement can match default hook. |
| `custom_albu_noise_light` | mild noise/compression p<=0.10 | Robustness, not primary mAP. |
| `custom_albu_grid_light` | GridDropout low p and low area | Regularization without hiding disease too often. |

Do not run this before the YOLO-native policy search, because replacing the default hook may remove the current strongest regularizer.

### Tier 4: Lower Priority / Defer

| Method | Reason To Defer |
|---|---|
| Fourier grid search | Current best Fourier did not beat baseline. |
| Wavelet enhancement | Already reported worse than baseline. |
| Macenko stain normalization | Natural shrimp RGB images are not histology stains; possible mismatch. |
| Diffusion augmentation | Too expensive and difficult to validate for this paper stage. |
| Full custom MaskMix | Implementation complexity; should wait until copy-paste has clear signal. |
| Strong RandAugment / AugMix | Official Ultralytics docs describe `auto_augment` as classification-specific. Use cautiously. |

## 5. Updated Optimization Checklist

### Phase 0: Instrumentation

- [ ] Confirm baseline notebook exports all metrics and split manifests.
- [ ] Confirm `DISABLE_ULTRALYTICS_ALBUMENTATIONS = False`.
- [ ] Confirm Ultralytics version in every run.
- [ ] Save exact train args for every experiment.
- [ ] Save visual augmented batch examples for any non-baseline policy.
- [ ] Export partial CSV after every completed run.

### Phase 1: YOLO-Native Policy Search

Run these first under grouped seed `42`:

| Order | Experiment |
|---:|---|
| 1 | `baseline_clean_light_hook_on` sanity row |
| 2 | `mosaic_0p10_close20` |
| 3 | `mosaic_0p25_close20` |
| 4 | `mosaic_0p25_close30` |
| 5 | `cp_0p05_mixup_mosaic_0p10` |
| 6 | `cp_0p10_mixup_mosaic_0p10` |
| 7 | `cp_0p05_flip_mosaic_0p25` |
| 8 | `hsv_geo_middle_no_rotate` |
| 9 | `hsv_geo_middle_rotate3` |

Selection:

```text
candidate passes if:
    labeled_test_mask_map50 >= 0.522
    AND healthy_test_mask_fp_rate <= 0.264
    AND healthy_aware_test_score > 0.459581
```

If no candidate passes, move to hard-negative mining.

### Phase 2: Healthy Hard-Negative Mining

- [ ] Run baseline on healthy train/valid images.
- [ ] Save healthy images with false masks at `conf=0.25`.
- [ ] Create `healthy_hard_negative_2x` duplicated-empty-label training set.
- [ ] Retrain with baseline augmentation.
- [ ] Evaluate threshold sweep.

Pass criteria:

```text
healthy_test_mask_fp_rate < 0.20
AND labeled_test_mask_map50 >= 0.500
```

This accepts a small mAP drop only if healthy FP improves strongly.

### Phase 3: Combined Best Policy

Only combine methods that individually help.

Candidate examples:

```python
best_mosaic + hard_negative_2x
best_copy_paste + hard_negative_2x
hsv_geo_middle + hard_negative_2x
```

Reject combinations that reduce mAP50-95 or increase count MAE significantly.

### Phase 4: Confidence Calibration / Threshold Sweep

Run for:

- baseline
- best Phase 1 policy
- best hard-negative policy
- best combined policy

Export:

- `threshold_sweep_healthy_fp.csv`
- `threshold_sweep_labeled_count_miss.csv`
- operating-point recommendation table

### Phase 5: Multi-Seed Confirmation

Run only if a method passes Phase 1/2/3.

Minimum:

```text
baseline vs best_candidate across seeds 42, 123, 3407
```

Better:

```text
baseline vs best_candidate across 5 grouped seeds
```

Report:

- mean
- standard deviation
- range
- per-seed split size
- train-test group overlap, expected `0`

## 6. Papers / Sources To Read In Priority Order

### Must Read

1. Ultralytics YOLO Data Augmentation documentation
   - https://docs.ultralytics.com/guides/yolo-data-augmentation/
   - Reason: Defines actual behavior of `mosaic`, `copy_paste`, `copy_paste_mode`, `auto_augment`, custom `augmentations`.

2. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation
   - https://arxiv.org/abs/2012.07177
   - Reason: Directly relevant to instance segmentation and rare classes.

3. RandAugment: Practical automated data augmentation with a reduced search space
   - https://arxiv.org/abs/1909.13719
   - Reason: Explains the principle behind reduced-search augmentation strength tuning.

4. Albumentations: fast and flexible image augmentations
   - https://arxiv.org/abs/1809.06839
   - Reason: Needed if replacing/default hook with custom transforms.

5. GridMask Data Augmentation
   - https://arxiv.org/abs/2001.04086
   - Reason: Candidate for regularization, but must be conservative.

### Read If Implementing Error-Driven Methods

6. Focal Loss for Dense Object Detection
   - https://arxiv.org/abs/1708.02002
   - Reason: Background/foreground imbalance and hard examples.

7. ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning
   - https://arxiv.org/abs/2007.07936
   - Reason: Segmentation-aware mixing concept.

8. DACS: Domain Adaptation via Cross-domain Mixed Sampling
   - https://arxiv.org/abs/2007.08702
   - Reason: Mixed sampling with segmentation labels and domain shift.

### Lower Priority

9. AugMix
   - https://arxiv.org/abs/1912.02781
   - Reason: Robustness inspiration, mostly classification.

10. TAAL
   - https://arxiv.org/abs/2301.06624
   - Reason: Test-time augmentation and uncertainty idea for segmentation.

11. SatSynth
   - https://arxiv.org/abs/2403.16605
   - Reason: Diffusion-generated image-mask pairs; high complexity, not immediate.

## 7. Recommended Next Notebook

Create a new notebook:

```text
shrimp-leakage-aware-segmentation/notebooks/yolo_aug_policy_search/yolo_aug_policy_search_v2_targeted.ipynb
```

Purpose:

- Replace broad Phase B grid with a targeted baseline-centered search.
- Keep hook enabled.
- Use only the 8-9 candidates listed in Phase 1.
- Export identical metrics to existing notebooks.

Do not include Fourier/wavelet/photometric copies in this notebook. They have already been tested enough to deprioritize.

## 8. Decision Summary

The next optimization effort should not expand randomly across every literature method. It should focus on:

1. low mosaic,
2. copy-paste plus low mosaic,
3. middle-strength HSV/geometric tuning,
4. healthy hard-negative mining,
5. threshold/calibration sweeps.

The most likely path to a real improvement is not generic image enhancement. It is a controlled augmentation policy that improves diseased-specimen generalization while explicitly controlling healthy false positives.
