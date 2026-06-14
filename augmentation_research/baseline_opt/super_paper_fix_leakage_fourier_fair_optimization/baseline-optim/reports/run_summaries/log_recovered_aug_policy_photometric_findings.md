# Log-Recovered Findings: YOLO Augmentation Policy and Photometric Screening

Source logs:

- `C:\Users\Admin\.codex\attachments\9d7c6327-02a2-486a-b92c-106a9e59cb2d\pasted-text.txt`
- `C:\Users\Admin\.codex\attachments\8c71196a-f7e3-41a4-816f-947479cc2a4b\pasted-text.txt`

These findings were recovered from pasted Colab logs after GPU/session loss. The original checkpoints, summary CSVs, healthy false-positive CSVs, and count-error CSVs were not available.

Therefore:

- Standard YOLO validation/test mAP values are recoverable from logs.
- Healthy false-positive rate, count MAE, disease miss rate, and healthy-aware score are not recoverable from these pasted logs.
- Any result without full test/labeled-test evaluation should be treated as incomplete.

## 1. Shared Experimental Contract

All visible runs used:

- model: `yolo11n-seg.pt`
- Ultralytics: `8.4.62`
- runtime: Colab T4
- image size: `640`
- epochs: `100`
- patience: `30`
- seed: `42`
- split: stratified grouped-specimen split
- hidden Ultralytics Albumentations hook: enabled
- base training augmentation: clean-light YOLO augmentation

Clean-light YOLO train args:

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
    "bgr": 0.0
}
```

Accepted baseline reference from prior grouped model sweep:

| Metric | Baseline |
|---|---:|
| Full test mask mAP50 | 0.481532 |
| Labeled-only test mask mAP50 | 0.512002 |
| Labeled-only test mask mAP50-95 | 0.167785 |
| Healthy test mask false-positive rate | 0.243902 |
| Healthy-aware labeled test score | 0.459581 |

## 2. Fully Recovered Runs

These runs include visible best-checkpoint validation plus full-test and labeled-only test evaluation in the pasted logs.

### 2.1 `hsv_mild`

Configuration:

```python
{
    "hsv_h": 0.015,
    "hsv_s": 0.45,
    "hsv_v": 0.25,
    "translate": 0.05,
    "scale": 0.20,
    "mosaic": 0.0,
    "copy_paste": 0.0,
    "mixup": 0.0
}
```

Training:

- early stopping best epoch: `58`
- total epochs run: `88`
- training time: `0.803 h`

Recovered metrics:

| Evaluation set | Images | Instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| best checkpoint validation | 115 | 102 | 0.602 | 0.265 | 0.477 | 0.190 |
| full validation | 115 | 102 | 0.601 | 0.266 | 0.478 | 0.190 |
| labeled-only validation | 74 | 102 | 0.624 | 0.274 | 0.498 | 0.196 |
| full test | 129 | 119 | 0.507 | 0.201 | 0.424 | 0.144 |
| labeled-only test | 88 | 119 | 0.537 | 0.213 | 0.450 | 0.152 |

Interpretation:

- `hsv_mild` improved validation mask mAP50 relative to many prior weak augmentation runs, but it underperformed the accepted baseline on test.
- Labeled-only test mask mAP50 dropped from baseline `0.512002` to `0.450`.
- Full test mask mAP50 dropped from baseline `0.481532` to `0.424`.
- This is a negative ablation: stronger HSV did not generalize better under grouped-specimen evaluation.

### 2.2 `sharpen_contrast`

Configuration:

Train-only photometric copies:

```python
{
    "mode": "fixed_sequence",
    "copies": 1,
    "ops": [
        {"op": "sharpness", "min": 1.10, "max": 1.70, "p": 0.90},
        {"op": "contrast", "min": 1.05, "max": 1.30, "p": 0.80},
        {"op": "autocontrast", "p": 0.35},
        {"op": "brightness", "min": 0.92, "max": 1.08, "p": 0.50}
    ]
}
```

Dataset expansion:

- created train-only photometric copies: `905`
- skipped images: `0`
- train images after copy expansion: `1810`
- train labeled images: `1168`
- train healthy images: `642`
- train instances: `1620`

Training:

- early stopping best epoch: `29`
- total epochs run: `59`
- training time: `1.072 h`

Recovered metrics:

| Evaluation set | Images | Instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| best checkpoint validation | 115 | 102 | 0.558 | 0.248 | 0.530 | 0.183 |
| full validation | 115 | 102 | 0.558 | 0.246 | 0.514 | 0.180 |
| labeled-only validation | 74 | 102 | 0.583 | 0.259 | 0.535 | 0.189 |
| full test | 129 | 119 | 0.499 | 0.196 | 0.431 | 0.138 |
| labeled-only test | 88 | 119 | 0.537 | 0.210 | 0.462 | 0.148 |

Interpretation:

- `sharpen_contrast` had strong validation numbers, especially labeled-only validation mask mAP50 `0.535`.
- It did not beat the accepted baseline on test.
- Labeled-only test mask mAP50 dropped from baseline `0.512002` to `0.462`.
- Full test mask mAP50 dropped from baseline `0.481532` to `0.431`.
- The validation-to-test drop suggests this policy may overfit to validation appearance or improve easy/visible contrast without improving unseen-specimen generalization.

## 3. Partially Recovered Runs

These runs do not contain full test/labeled-test evaluation in the pasted logs.

### 3.1 `scale_translate_mild`

Configuration:

```python
{
    "translate": 0.08,
    "scale": 0.30,
    "hsv_h": 0.01,
    "hsv_s": 0.35,
    "hsv_v": 0.20,
    "mosaic": 0.0,
    "copy_paste": 0.0,
    "mixup": 0.0
}
```

Training:

- early stopping best epoch: `39`
- total epochs run: `69`
- training time: `0.644 h`

Recovered validation metrics:

| Evaluation set | Images | Instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| best checkpoint validation | 115 | 102 | 0.540 | 0.226 | 0.482 | 0.185 |
| full validation | 115 | 102 | 0.540 | 0.224 | 0.483 | 0.183 |
| labeled-only validation | 74 | 102 | 0.543 | 0.226 | 0.485 | 0.184 |

Interpretation:

- Validation performance was not obviously better than the baseline direction.
- No full-test or labeled-test metrics were visible in the pasted log, so this run cannot be accepted or rejected conclusively.
- Since validation mask mAP50 is below `sharpen_contrast` and not clearly exceptional, this is probably not a high-priority rerun unless the YOLO augmentation sweep needs completion.

### 3.2 `randaug_photo_n2_mild`

Configuration:

Train-only photometric RandAugment-style copies:

```python
{
    "mode": "randaugment",
    "copies": 1,
    "num_ops": 2,
    "magnitude": "mild",
    "op_pool": [
        "brightness",
        "contrast",
        "color",
        "sharpness",
        "gamma",
        "autocontrast",
        "equalize"
    ]
}
```

Dataset expansion:

- created train-only photometric copies: `905`
- skipped images: `0`
- train images after copy expansion: `1810`
- train labeled images: `1168`
- train healthy images: `642`
- train instances: `1620`

Visible training status:

- training started normally
- pasted log stops around epoch `29`
- no early stopping line
- no final validation/test evaluation
- no checkpoint-selected metrics are recoverable

Best visible validation row before interruption:

| Epoch | Images | Instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 27 | 115 | 102 | 0.552 | 0.236 | 0.471 | 0.177 |

Interpretation:

- `randaug_photo_n2_mild` showed some validation signal before interruption.
- Because no best-checkpoint final evaluation or test evaluation is visible, this result is incomplete and should not be compared against the baseline.
- If photometric methods are revisited, this is more worth rerunning than `scale_translate_mild`, because its visible validation mask mAP50 reached `0.471` before completion.

## 4. Listed But Not Recovered

The logs list the following planned experiments, but no final metrics were recoverable from the pasted text:

| Experiment | Status |
|---|---|
| `randaug_photo_n3_medium` | listed, not started or not visible in pasted log |
| `copy_paste_0p10_flip_mosaic_low_close20` | listed, not started or not visible in pasted log |

## 5. Main Finding

The recovered completed runs do not beat the accepted clean-light YOLO baseline on the practical healthy-aware metric.

| Method | Full test mask mAP50 | Labeled-only test mask mAP50 | Labeled-only test mask mAP50-95 | Test status |
|---|---:|---:|---:|---|
| accepted baseline | 0.481532 | 0.512002 | 0.167785 | complete prior CSV |
| `hsv_mild` | 0.424 | 0.450 | 0.152 | recovered from log |
| `sharpen_contrast` | 0.431 | 0.462 | 0.148 | recovered from log |
| `scale_translate_mild` | missing | missing | missing | validation-only recovered |
| `randaug_photo_n2_mild` | missing | missing | missing | interrupted |

Current interpretation:

> Under grouped-specimen evaluation, stronger HSV and train-only sharpen/contrast photometric copies improved or looked competitive on validation, but did not transfer to the unseen-specimen test set. This supports the emerging pattern that extra augmentation can overfit validation behavior or increase apparent robustness without improving fair test generalization.

## 6. CSV-Summary-Backed Results Added Later

The user later provided summary-table outputs for completed copy-paste and early photometric runs. These are stronger evidence than raw pasted training logs because they include the notebook's custom metrics: healthy false positives, count error, disease miss rate, and healthy-aware score.

### 6.1 YOLO Copy-Paste Summary

| Method | Copy-paste | Mode | Full test mask mAP50 | Labeled test mask mAP50 | Labeled test mask mAP50-95 | Healthy test FP rate | Healthy-aware test score |
|---|---:|---|---:|---:|---:|---:|---:|
| accepted baseline | 0.00 | none | 0.481532 | 0.512002 | 0.167785 | 0.243902 | 0.459581 |
| `copy_paste_0p05_flip` | 0.05 | flip | 0.481532 | 0.512002 | 0.167785 | 0.243902 | 0.459581 |
| `copy_paste_0p10_flip` | 0.10 | flip | 0.481532 | 0.512002 | 0.167785 | 0.243902 | 0.459581 |
| `copy_paste_0p10_mixup` | 0.10 | mixup | 0.464436 | 0.514959 | 0.143876 | 0.487805 | 0.424322 |

Interpretation:

- `copy_paste_0p10_mixup` gives the only visible raw labeled-test mAP50 gain: `0.514959` versus baseline `0.512002`.
- The gain is very small: `+0.002957` absolute labeled-only test mask mAP50.
- It also worsens important practical metrics:
  - healthy test FP rate rises from `0.243902` to `0.487805`.
  - healthy-aware test score drops from `0.459581` to `0.424322`.
  - labeled test mAP50-95 drops from `0.167785` to `0.143876`.
- `copy_paste_0p05_flip` and `copy_paste_0p10_flip` are numerically identical to the accepted baseline across the reported metrics.

Possible explanation:

> In the pinned Ultralytics configuration, flip-mode copy-paste may have had no practical effect when `mosaic=0`, or the augmentation path may not have been activated in a way that changes the training distribution. Mixup-mode copy-paste changed behavior, but it increased healthy false positives and reduced stricter mask quality.

Conclusion:

> Copy-paste is still worth focused investigation, but the current evidence does not justify selecting `copy_paste_0p10_mixup` as the final model because its tiny mAP50 gain is outweighed by worse healthy false positives and lower healthy-aware score.

### 6.2 Photometric Summary

| Method | Full test mask mAP50 | Labeled test mask mAP50 | Labeled test mask mAP50-95 | Healthy test FP rate | Healthy-aware test score |
|---|---:|---:|---:|---:|---:|
| accepted baseline | 0.481532 | 0.512002 | 0.167785 | 0.243902 | 0.459581 |
| `photometric_mild` | 0.446587 | 0.469547 | 0.176691 | 0.243902 | 0.412203 |
| `photometric_medium` | 0.443425 | 0.476580 | 0.167594 | 0.243902 | 0.420466 |
| `noise_blur_compression` | 0.449309 | 0.470237 | 0.135589 | 0.365854 | 0.400508 |

Interpretation:

- None of the early train-only photometric policies beat the baseline.
- `photometric_mild` and `photometric_medium` preserve the same healthy FP rate as baseline, but reduce labeled-test mAP50.
- `noise_blur_compression` increases healthy false positives and reduces both labeled mAP50 and mAP50-95.
- These results agree with the later log-recovered `sharpen_contrast` finding: photometric augmentation can look plausible on validation but does not currently improve grouped-specimen test performance.

### 6.3 Updated Practical Ranking

Based on available complete metrics:

| Rank | Method | Reason |
|---:|---|---|
| 1 | accepted clean-light YOLO baseline | Best healthy-aware score and strong labeled-test mAP. |
| 2 | `copy_paste_0p10_mixup` | Slightly highest labeled-test mAP50, but much worse healthy FP. |
| 3 | `copy_paste_0p05_flip` / `copy_paste_0p10_flip` | Equivalent to baseline, no evidence of actual benefit. |
| 4 | photometric policies | Generally weaker test transfer. |

Research implication:

> The first optimization sweep did not yet find a clear replacement for the clean-light YOLO baseline. However, copy-paste mixup shows a small sensitivity-oriented signal, so the next focused experiment should tune copy-paste while explicitly controlling healthy false positives.

## 7. Recommendation

Do not spend more GPU immediately rerunning all lost augmentation experiments.

Priority if GPU becomes available:

1. Run a focused copy-paste sweep, because `copy_paste_0p10_mixup` is the only method with any labeled-test mAP50 gain.
2. Include healthy FP as a hard constraint in the copy-paste sweep.
3. Test whether copy-paste only works when paired with low mosaic or close-mosaic.
4. Rerun only the most plausible incomplete photometric candidate: `randaug_photo_n2_mild`.
5. Skip or deprioritize `hsv_mild`, `sharpen_contrast`, and the completed photometric policies as final candidates because their test results are below baseline.
6. If continuing augmentation optimization, move toward smaller, targeted policies and always save/export after each experiment.

Operational lesson:

> Long multi-experiment Colab runs are fragile. Future notebooks should either run one experiment per cell/session or create an emergency zip export after every completed experiment before starting the next one.

## 8. Frequency/Wavelet Partial Result: Fixed Fourier `sigma=50`, `alpha=0.30`

The user later provided `seg_frequency_wavelet_revisit_partial.csv`. This partial file contains four completed Fourier rows and no Wavelet rows yet.

All Fourier rows used:

- Fourier sigma: `50`
- Fourier alpha: `0.30`
- model: `yolo11n-seg.pt`
- split: stratified grouped-specimen split
- clean-light YOLO train args
- seed: `42`

### 8.1 Fourier Result Table

| Method | Mode | Hook | Full test mask mAP50 | Labeled test mask mAP50 | Labeled test mask mAP50-95 | Healthy FP rate | Healthy-aware test score |
|---|---|---|---:|---:|---:|---:|---:|
| accepted baseline | none | on | 0.481532 | 0.512002 | 0.167785 | 0.243902 | 0.459581 |
| `fourier_s50_a0p30_hook_on_train_only` | train-only copies | on | 0.435524 | 0.467705 | 0.162519 | 0.341463 | 0.402782 |
| `fourier_s50_a0p30_hook_on_full_splits` | full-splits preprocessing | on | 0.392726 | 0.418055 | 0.132921 | 0.365854 | 0.338666 |
| `fourier_s50_a0p30_hook_off_train_only` | train-only copies | off | 0.440524 | 0.465859 | 0.149726 | 0.317073 | 0.383015 |
| `fourier_s50_a0p30_hook_off_full_splits` | full-splits preprocessing | off | 0.452348 | 0.497591 | 0.166278 | 0.243902 | 0.426231 |

### 8.2 Interpretation

The strongest Fourier row is:

> `fourier_s50_a0p30_hook_off_full_splits`

Why:

- It has the best Fourier labeled-test mask mAP50: `0.497591`.
- It has the best Fourier healthy-aware score: `0.426231`.
- It preserves the baseline healthy FP rate: `0.243902`.
- Its mAP50-95 is close to baseline: `0.166278` vs baseline `0.167785`.

However, it still does not beat the accepted baseline:

| Metric | Baseline | Best Fourier |
|---|---:|---:|
| Full test mask mAP50 | 0.481532 | 0.452348 |
| Labeled test mask mAP50 | 0.512002 | 0.497591 |
| Labeled test mask mAP50-95 | 0.167785 | 0.166278 |
| Healthy FP rate | 0.243902 | 0.243902 |
| Healthy-aware test score | 0.459581 | 0.426231 |

### 8.3 Scientific Meaning

This Fourier result is useful even though it does not beat baseline.

It suggests:

- Fourier is not complementary to the current hook-on YOLO baseline.
- Hook-on train-only Fourier gives strong validation performance but worsens test transfer and healthy false positives.
- Hook-on full-splits Fourier performs worst, matching earlier evidence that full-split Fourier with YOLO hook enabled is not a good deployment preprocessing path.
- Hook-off full-splits Fourier is the most coherent Fourier setting, suggesting Fourier may partially substitute for YOLO's hidden Albumentations hook, but not surpass it.

Paper-safe claim:

> Fixed Fourier enhancement was most competitive when used as full-split preprocessing with the Ultralytics Albumentations hook disabled, but it still underperformed the default hook-on YOLO baseline. This indicates that frequency-domain enhancement captures some useful contrast/detail signal, but does not currently provide a net improvement over the default YOLO augmentation pipeline under grouped-specimen evaluation.

### 8.4 Recommendation From Fourier Partial

Do not expand into a large Fourier parameter sweep unless the paper needs a dedicated frequency appendix.

Recommended next actions:

1. Let the Wavelet rows finish, because Wavelet may preserve local texture better than global Fourier.
2. Keep `fourier_s50_a0p30_hook_off_full_splits` as the best Fourier representative.
3. Do not choose Fourier as the final optimized model unless Wavelet/copy-paste also fail and the paper needs a frequency-domain comparison section.
4. If retesting Fourier, retest only `hook_off_full_splits` and possibly `hook_off_train_only`, not all four modes.
