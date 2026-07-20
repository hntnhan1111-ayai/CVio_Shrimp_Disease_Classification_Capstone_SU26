# Notebook 23: Fourier Order Comparison

## Scope

This report compares:

- Original order: fixed FourierTF Fourier preprocessing on the copied dataset, followed by YOLO HSV augmentation during training.
- Reordered path: YOLO HSV augmentation first, followed by FourierTF Fourier inside the training transform.

FourierTF is the image high-pass policy `highpass_s50_a0p10` (`sigma=50`, `alpha=0.10`). Both experiments use YOLO11n-seg, seed 42, grouped-specimen splitting, batch 16, workers 0, 100 requested epochs, patience 40, clean-light augmentation disabled, and the hidden Albumentations hook disabled.

Evidence files:

- `23_only_fourier_n1_hsv_w1_interaction_kaggle_seed42_summary.csv`
- `23_only_fourier_n1_hsv_w1_interaction_kaggle_seed42_paper_row.csv`
- `only_fourier_n1_hsv_w1_interaction_kaggle_seed42_aug_before_fourier_paper_row.csv`
- `023_n1_hsv_w1_interaction_kaggle_seed42.ipynb`
- `023_n1_hsv_w1_interaction_kaggle_seed42_aug_before_fourier.ipynb`

## Implementation verification

The original notebook calls `apply_fourier_highpass_to_dataset()` over `train`, `valid`, and `test` before `yolo.train()`. Therefore, for FourierTF rows the effective training path is:

```text
original image -> FourierTF Fourier on disk -> YOLO HSV -> model
```

The reordered notebook sets `FOURIER_AFTER_TRAIN_AUGMENTATION = True`. For FourierTF rows it Fourier-preprocesses only `valid` and `test`, then patches Ultralytics `RandomHSV.__call__` so the train image is transformed as:

```text
original image -> YOLO HSV -> FourierTF Fourier -> model
```

The reordered notebook explicitly records `fourier_train_order` and `fourier_preprocessed_splits`. The original notebook does not record these fields, but its source code establishes the original order directly.

The reordered implementation is specifically a post-HSV hook, not a generic final-transform hook. This is appropriate for notebook 23 because HSV is the only enabled augmentation family and all other explicit YOLO augmentation values are zero.

### Important naming caveat for W1 only

`W1 only` means that the experiment has no nonzero HSV augmentation parameters. Ultralytics still constructs the `RandomHSV` transform, but its configured gains are all zero for this row. Therefore:

- Original W1-only: cached W1 image -> a no-op HSV transform -> model.
- Reordered W1-only: a no-op HSV transform -> online W1 hook -> model.

The `HSV -> W1` label in the reordered row describes the hook location, not an actual color augmentation. These two W1-only paths also differ in execution: the original notebook applies W1 once to copied training images before training, while the reordered notebook computes W1 online inside the training transform. The W1-only result is therefore a diagnostic of the changed execution path, not a clean causal estimate of HSV-versus-Fourier order. The clean order comparison is provided by the nonzero mild, medium, and strong HSV pairs.

## Paired test results

Values below are labeled-only test metrics unless stated otherwise. Delta is reordered minus original.

| HSV setting | FourierTF order | mask mAP50 | mask mAP50-95 | healthy FP rate | disease miss rate | healthy-aware test score |
|---|---|---:|---:|---:|---:|---:|
| FourierTF only | FourierTF -> Zero_HSV | 0.345105 | 0.088484 | 0.634146 | 0.193182 | 0.210763 |
| FourierTF only | Zero_HSV -> FourierTF | 0.278724 | 0.080753 | 0.341463 | 0.295455 | 0.172040 |
| FourierTF only | delta | -0.066381 | -0.007731 | -0.292683 | +0.102273 | -0.038723 |
| Mild | FourierTF -> HSV | 0.245578 | 0.068791 | 0.121951 | 0.420455 | 0.138875 |
| Mild | HSV -> FourierTF | 0.280301 | 0.074607 | 0.292683 | 0.431818 | 0.157283 |
| Mild | delta | +0.034723 | +0.005817 | +0.170732 | +0.011364 | +0.018407 |
| Medium | FourierTF -> HSV | 0.228811 | 0.070411 | 0.268293 | 0.477273 | 0.098762 |
| Medium | HSV -> FourierTF | 0.276781 | 0.077475 | 0.317073 | 0.340909 | 0.164108 |
| Medium | delta | +0.047970 | +0.007065 | +0.048780 | -0.136364 | +0.065346 |
| Strong | FourierTF -> HSV | 0.334920 | 0.122206 | 0.463415 | 0.238636 | 0.217272 |
| Strong | HSV -> FourierTF | 0.299051 | 0.086543 | 0.439024 | 0.181818 | 0.201550 |
| Strong | delta | -0.035870 | -0.035662 | -0.024390 | -0.056818 | -0.015723 |

The augmentation-only controls are effectively identical between the two CSVs:

| HSV setting | mAP50 original | mAP50 reordered | healthy FP original | healthy FP reordered |
|---|---:|---:|---:|---:|
| None | 0.235374 | 0.235374 | 0.390244 | 0.390244 |
| Mild | 0.280658 | 0.280658 | 0.341463 | 0.341463 |
| Medium | 0.305874 | 0.305874 | 0.317073 | 0.317073 |
| Strong | 0.295270 | 0.295270 | 0.195122 | 0.195122 |

This is important: the order effect is isolated to the FourierTF rows rather than being caused by a changed split, label set, augmentation-only configuration, or baseline model result. The split fingerprint and all substantive training metadata are unchanged; only the generated report path differs.

## Impact by operating objective

### Raw segmentation accuracy

Reordering FourierTF after HSV helps mild and medium HSV:

- Mild: mAP50 improves by 0.034723.
- Medium: mAP50 improves by 0.047970.

It hurts FourierTF alone and strong HSV:

- FourierTF alone: mAP50 decreases by 0.066381.
- Strong HSV: mAP50 decreases by 0.035870.

The best reordered FourierTF combination is medium HSV plus FourierTF at mAP50 `0.276781`, but it is still below the best original-order FourierTF result, which is FourierTF alone at `0.345105`. It is also below the medium HSV-only control at `0.305874`.

### Healthy false positives

The reordered path is substantially safer for FourierTF alone:

- Healthy FP rate decreases from `0.634146` to `0.341463`.
- FP masks per healthy image decrease from `0.926829` to `0.341463`.

For augmented runs, the effect is mixed:

- Mild FourierTF: FP rate increases by `0.170732`.
- Medium FourierTF: FP rate increases by `0.048780`.
- Strong FourierTF: FP rate decreases by `0.024390`.

Thus, applying FourierTF after HSV does not generally solve Fourier-induced healthy false positives. It only clearly helps the FourierTF-only case.

### Disease recall and count behavior

The reordered path lowers disease mask miss rate for medium and strong HSV:

- Medium: `0.477273 -> 0.340909`.
- Strong: `0.238636 -> 0.181818`.

It increases miss rate slightly for mild HSV and substantially for FourierTF alone. Mask-count MAE improves in all four FourierTF comparisons, but this does not translate into uniformly higher mAP because localization/overlap quality also changes.

### Healthy-aware score

The reordered path improves the test healthy-aware score for mild and medium HSV, especially medium HSV:

- Mild: `0.138875 -> 0.157283`.
- Medium: `0.098762 -> 0.164108`.

It decreases the score for FourierTF alone and strong HSV:

- FourierTF alone: `0.210763 -> 0.172040`.
- Strong: `0.217272 -> 0.201550`.

The best reordered FourierTF row by healthy-aware test score is medium HSV plus FourierTF at `0.164108`; the best original-order FourierTF row is strong HSV plus FourierTF at `0.217272`. The reordered best is therefore not a new overall Fourier result.

## Training-time impact

The reordered FourierTF rows apply CPU Fourier during training for every transformed training image. Their recorded training times are much higher:

| FourierTF setting | Original time (min) | Reordered time (min) | Change (min) |
|---|---:|---:|---:|
| FourierTF only | 29.78 | 87.35 | +57.57 |
| Mild | 40.85 | 109.79 | +68.94 |
| Medium | 48.87 | 86.93 | +38.06 |
| Strong | 35.44 | 97.16 | +61.72 |

These timing differences are implementation costs, not model-quality evidence. The original path preprocesses each training image once and reuses the result. The reordered path recomputes FourierTF online during training, which is also why its wall-clock behavior should not be compared as if both paths used the same Fourier implementation cost.

## Interpretation

The results support a narrower conclusion than “augmentation should always precede Fourier.” The order changes the interaction in a setting-dependent way:

1. The original cached-Fourier path produces the strongest raw mAP for W1 alone, but that W1-only comparison is confounded by cached versus online Fourier execution. W1 before actual strong HSV also gives higher mAP than the reordered strong-HSV path, but with high healthy false positives.
2. HSV before FourierTF is more compatible with medium HSV: it recovers diseased-mask recall and improves the healthy-aware score relative to FourierTF after medium HSV in the original order.
3. The reordered path does not make FourierTF independently beneficial. Medium HSV-only still beats medium HSV plus FourierTF on mAP50 (`0.305874` versus `0.276781`) and has a slightly lower healthy FP rate (`0.317073` versus `0.317073`, equal in this run).
4. The strongest recorded result remains the original-order W1-only row by labeled test mAP50 (`0.345105`), but because its counterpart uses a different online-Fourier execution path, this should be treated as a result to reproduce, not as proof that W1-before-no-op-HSV is superior. It also has a high healthy FP rate (`0.634146`) and is not the best practical operating point.

The most defensible paper statement is therefore:

> Fourier order materially changes the augmentation interaction. Applying FourierTF after HSV improved medium-augmentation disease recall and healthy-aware score relative to the corresponding FourierTF-before-HSV run, but it did not surpass the best FourierTF-before-HSV raw mAP result and did not establish a generally superior order.

## Limitations and next decision

- These are single seed-42 runs. The order conclusions are directional, not multi-seed claims.
- The reordered implementation injects FourierTF through `RandomHSV.__call__`, so it tests HSV -> FourierTF specifically, not an arbitrary “after all augmentations” pipeline.
- The W1-only pair is confounded by cached versus online Fourier execution and should not be used as the primary order conclusion.
- The original paper-row CSV lacks explicit order metadata; the order is verified from notebook source code.
- The validation-selected checkpoint can differ across rows, so epoch counts and best epochs are descriptive rather than causal evidence.

For the next experiment, do not expand the order study across more HSV strengths. The useful candidate is a confirmation of medium HSV with HSV -> FourierTF against medium HSV-only and the original-order medium HSV + FourierTF, using at least seeds 42, 123, and 3407. If the goal is maximum raw mAP, retain FourierTF-only as a high-FP result but do not treat it as the practical winner. If the goal is healthy-aware segmentation, medium HSV -> FourierTF is the only reordered candidate that merits confirmation from this notebook.
