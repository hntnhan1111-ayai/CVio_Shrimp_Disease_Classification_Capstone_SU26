# Fourier Interaction With Augmentation: N=1 and N=2 Summary

## Scope

This summary consolidates the executed seed-42 N=1 and N=2 experiments in `ONLY_fourier`.

The primary comparison is labeled-test mask mAP50. Fourier interaction is measured against the matched augmentation-only row:

```text
Fourier delta = Fourier-order mAP50 - augmentation-only mAP50
```

The two Fourier orders are:

- `augmentation -> Fourier`: offline augmentation is applied to the training images, then Fourier preprocessing is applied to train/valid/test.
- `Fourier -> augmentation`: Fourier preprocessing is applied to train/valid/test, then offline augmentation is applied to the training images.

All later N=1/N=2 offline experiments use the same grouped-specimen split fingerprint:

```text
1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb
```

The controlled settings are seed 42, YOLO11n-seg, Ultralytics 8.4.62, batch 16, workers 0, hidden Albumentations hook disabled, and W1 high-pass Fourier with `sigma=50`, `alpha=0.10`.

## Executive Findings

1. Fourier is not uniformly beneficial. Its effect depends strongly on the augmentation family, strength, and order.
2. The best raw augmentation-only result is N=2 Translate `0.08` + FlipLR `0.25`, with labeled-test mAP50 `0.412850`. Fourier reduces it in both orders.
3. The clearest positive N=1 Fourier interactions are Degrees and selected FlipLR/FlipUD settings.
4. The clearest positive N=2 Fourier interactions are Scale + FlipUD with Fourier applied before augmentation, and Translate + Erasing with Fourier applied before augmentation.
5. Fourier often improves disease recall or healthy FP behavior while reducing raw mAP50. The healthy-aware score must therefore be checked alongside mAP50.
6. There is no universal winning order. `augmentation -> Fourier` is usually less damaging for Scale and FlipLR, while `Fourier -> augmentation` is often better for Erasing, FlipUD, and Degrees.

## N=1 Results

The tables below report labeled-test mask mAP50. The selected augmentation-only value is the raw-performance winner within each tested family, not the Fourier winner.

### N=1 Translate

| Translate | Augmentation only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Better Fourier order |
|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0.321386 | 0.265800 | -0.055586 | 0.319359 | -0.002027 | Fourier -> Aug |
| 0.08 | 0.378245 | 0.350022 | -0.028223 | 0.300384 | -0.077861 | Aug -> Fourier |

Fourier does not improve Translate. The lower Translate value is relatively stable when Fourier is applied after augmentation, but the stronger Translate value is harmed by both orders.

### N=1 Scale

| Scale | Augmentation only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Better Fourier order |
|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 0.369081 | 0.318537 | -0.050544 | 0.230283 | -0.138798 | Aug -> Fourier |
| 0.20 | 0.333299 | 0.294439 | -0.038860 | 0.269252 | -0.064047 | Aug -> Fourier |

Scale is consistently harmed by Fourier. Applying augmentation first is less damaging.

### N=1 FlipLR

| FlipLR | Augmentation only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Better Fourier order |
|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 0.327451 | 0.333991 | +0.006540 | 0.365107 | +0.037656 | Fourier -> Aug |
| 0.50 | 0.291080 | 0.292293 | +0.001213 | 0.284207 | -0.006873 | Aug -> Fourier |

FlipLR is one of the strongest N=1 Fourier interactions. The low probability benefits from Fourier, especially when Fourier is applied before augmentation. This result does not generalize to the stronger probability.

### N=1 Degrees

| Degrees | Augmentation only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Better Fourier order |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.310849 | 0.297214 | -0.013635 | 0.359537 | +0.048688 | Fourier -> Aug |
| 5 | 0.307844 | 0.344493 | +0.036649 | 0.336845 | +0.029001 | Aug -> Fourier |

Degrees has a clear positive interaction at both tested strengths, but the preferred order changes with strength. Degrees `3` favors Fourier before augmentation; Degrees `5` favors augmentation before Fourier.

### N=1 Erasing

| Erasing | Augmentation only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Better Fourier order |
|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 0.338450 | 0.262397 | -0.076053 | 0.233618 | -0.104832 | Aug -> Fourier |
| 0.15 | 0.198143 | 0.272676 | +0.074533 | 0.290710 | +0.092567 | Fourier -> Aug |

Erasing is highly strength-sensitive. Fourier is harmful at `0.10` but beneficial at `0.15`. The absolute performance at `0.15` remains low despite the positive delta.

### N=1 FlipUD

| FlipUD | Augmentation only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Better Fourier order |
|---:|---:|---:|---:|---:|---:|---|
| 0.15 | 0.302918 | 0.306720 | +0.003802 | 0.296649 | -0.006269 | Aug -> Fourier |
| 0.30 | 0.334533 | 0.345064 | +0.010531 | 0.282653 | -0.051880 | Aug -> Fourier |

FlipUD favors augmentation before Fourier. The `0.30` setting gives a modest raw mAP50 gain, but its healthy FP rate increases from `0.292683` to `0.414634`.

### N=1 HSV

HSV was initially tested through the online YOLO augmentation path. These rows are useful interaction evidence but are not perfectly comparable to the later offline geometric/erasing experiments.

| HSV strength | Augmentation only | Fourier -> HSV | Delta | HSV -> Fourier | Delta | Better Fourier order |
|---|---:|---:|---:|---:|---:|---|
| Mild | 0.280658 | 0.245578 | -0.035080 | 0.280301 | -0.000357 | HSV -> Fourier |
| Medium | 0.305874 | 0.228811 | -0.077063 | 0.276781 | -0.029093 | HSV -> Fourier |
| Strong | 0.295270 | 0.334920 | +0.039650 | 0.299051 | +0.003781 | Fourier -> HSV |

HSV shows the same regime dependence: Fourier is negative for mild/medium settings and slightly positive for strong settings. The offline HSV study also found medium HSV followed by offline Fourier to be the strongest offline HSV interaction, but it did not include an HSV-only control in that notebook.

## N=1 Raw Augmentation Ranking

Using the best augmentation-only value from each family:

| Rank | Family and selected value | Augmentation-only mAP50 |
|---:|---|---:|
| 1 | Translate `0.08` | 0.378245 |
| 2 | Scale `0.10` | 0.369081 |
| 3 | Erasing `0.10` | 0.338450 |
| 4 | FlipUD `0.30` | 0.334533 |
| 5 | FlipLR `0.25` | 0.327451 |
| 6 | Degrees `3` | 0.310849 |
| 7 | HSV medium | 0.305874 |

This ranking was used to prioritize N=2 pairs.

## N=2 Results

### First Four N=2 Pairs

| Pair | Augmentation only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Best raw result |
|---|---:|---:|---:|---:|---:|---|
| Translate + Scale | 0.318218 | 0.307586 | -0.010632 | 0.300178 | -0.018041 | Augmentation only |
| Translate + Erasing | 0.224153 | 0.265065 | +0.040912 | 0.302729 | +0.078576 | Fourier -> Aug |
| Translate + FlipUD | 0.316863 | 0.265739 | -0.051124 | 0.302382 | -0.014481 | Augmentation only |
| Scale + Erasing | **0.342583** | 0.246335 | -0.096249 | 0.280140 | -0.062444 | Augmentation only |

Scale + Erasing is the best raw augmentation-only result among the first four pairs, but both Fourier orders reduce it. Translate + Erasing has the strongest positive Fourier delta, although its absolute augmentation-only baseline is weak.

### Second Four N=2 Pairs

| Pair | Augmentation only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Best raw result |
|---|---:|---:|---:|---:|---:|---|
| Translate + FlipLR | **0.412850** | 0.339754 | -0.073096 | 0.319382 | -0.093468 | Augmentation only |
| Scale + FlipUD | 0.271389 | 0.237733 | -0.033656 | **0.327079** | **+0.055690** | Fourier -> Aug |
| Scale + FlipLR | 0.322144 | **0.324253** | +0.002109 | 0.250022 | -0.072122 | Aug -> Fourier |
| Translate + Degrees | 0.291783 | 0.308670 | +0.016887 | **0.310324** | **+0.018541** | Fourier -> Aug |

Translate + FlipLR is the best N=2 augmentation-only result overall. It reaches mAP50 `0.412850`, healthy FP `0.170732`, and healthy-aware score `0.300322`. Fourier reduces raw performance in both orders, so this pair currently favors no Fourier.

Scale + FlipUD is the strongest positive N=2 Fourier interaction by delta. Fourier before augmentation raises mAP50 from `0.271389` to `0.327079`, but the absolute result remains below the best N=1 augmentation-only rows.

## Order Effects Across Families

### Augmentation Before Fourier Is Usually Better For

- Scale at both tested strengths.
- FlipUD at both tested strengths.
- FlipLR at `0.50`.
- Translate + Scale.
- Scale + FlipLR.
- Translate at the stronger `0.08` setting.

This order is generally safer when the augmentation changes geometry or scale in a way that Fourier would otherwise amplify after the transformation.

### Fourier Before Augmentation Is Usually Better For

- Erasing at `0.15`.
- Degrees at `3`.
- Translate + Erasing.
- Scale + FlipUD.
- Translate + Degrees.
- FlipLR at `0.25`.

This order can be beneficial when the augmentation partially suppresses or disrupts the frequency-enhanced signal, but the effect is not universal.

### No Universal Order

The order effect changes with augmentation strength. Examples:

- Degrees `3`: Fourier -> augmentation is better.
- Degrees `5`: augmentation -> Fourier is better.
- Erasing `0.10`: both orders hurt.
- Erasing `0.15`: both orders help, with Fourier -> augmentation better.
- FlipLR `0.25`: both orders help, with Fourier -> augmentation better.
- FlipLR `0.50`: only augmentation -> Fourier is slightly positive.

Therefore, order must be treated as an interaction factor rather than a globally fixed preprocessing rule.

## Practical Selection

### Best Raw Performance

The current best method in this experiment family is:

```text
Translate 0.08 + FlipLR 0.25, augmentation only
mAP50 = 0.412850
healthy FP rate = 0.170732
healthy-aware score = 0.300322
```

Fourier should not be added to this pair based on the current seed-42 evidence because it reduces both raw mAP50 and healthy-aware score.

### Best Positive Fourier Interaction

The strongest positive N=2 Fourier interaction is:

```text
Scale 0.10 + FlipUD 0.30
Fourier -> augmentation
mAP50: 0.271389 -> 0.327079
delta = +0.055690
```

This is a positive interaction result, not the best overall method. It should only be promoted if the research question is specifically Fourier synergy rather than maximum segmentation performance.

### Most Stable Fourier Order

Translate + Scale is the most stable pair across the two Fourier orders. Its raw mAP50 decreases modestly with augmentation -> Fourier and more with Fourier -> augmentation. The augmentation -> Fourier result retains a healthy-aware score close to the augmentation-only control.

## Limitations

- These are single-seed results with seed 42. They are screening evidence, not multi-seed confirmation.
- N=1 HSV used an earlier online YOLO path, while the later geometric/erasing studies used offline transformations. HSV comparisons should therefore be interpreted cautiously.
- The N=2 composition order between the two augmentation families was fixed as listed in each pair. The experiment varies Fourier position relative to the composed augmentation, not the internal order of the two augmentation families.
- Raw mAP50 and healthy-aware score can disagree because Fourier changes healthy false positives, disease miss rate, and count error.
- A positive Fourier delta does not imply a strong absolute model. Translate + Erasing and Scale + FlipUD demonstrate this distinction.

## Recommended Next Step

The strongest candidate for multi-seed confirmation is Translate `0.08` + FlipLR `0.25` without Fourier. For a Fourier-focused confirmation, use Scale `0.10` + FlipUD `0.30` with Fourier -> augmentation, but retain healthy FP and healthy-aware score as acceptance constraints.

