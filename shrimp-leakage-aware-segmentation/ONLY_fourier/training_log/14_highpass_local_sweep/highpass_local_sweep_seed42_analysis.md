# Mode G High-Pass Local Sweep Seed 42 Analysis

Source CSVs:

- `C:\Users\Admin\Downloads\only_fourier_mode_G_highpass_local_sweep_seed42_summary.csv`
- `C:\Users\Admin\Downloads\only_fourier_mode_G_highpass_local_sweep_seed42_paper_row.csv`

Both files contain 9 rows with matching metrics:

- present modes: T1, T2, T3, T4, T5, T6, T7, T8, T9
- missing mode: T0
- split fingerprint: `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`

T0 was the intended G4 anchor repeat, but it is absent from both CSVs. Compare this sweep against the prior G4 anchor:

- G4 prior high-pass `sigma=50 alpha=0.10`
- labeled test mask mAP50: `0.531771`
- labeled test mask mAP50-95: `0.170904`
- healthy FP: `0.243902`
- healthy-aware: `0.470165`

## mAP50 Ranking

| Mode | Sigma | Alpha | mAP50 | Delta vs G4 | Delta vs 0.54 | mAP50-95 | Healthy FP | Miss | Count MAE | Healthy-aware |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 35 | 0.08 | 0.486738 | -0.045033 | -0.053262 | 0.169787 | 0.414634 | 0.056818 | 0.371212 | 0.418192 |
| T6 | 40 | 0.12 | 0.484870 | -0.046901 | -0.055130 | 0.162447 | 0.341463 | 0.125000 | 0.397727 | 0.412087 |
| T3 | 35 | 0.12 | 0.476721 | -0.055050 | -0.063279 | 0.165574 | 0.414634 | 0.102273 | 0.342803 | 0.402776 |
| T7 | 45 | 0.10 | 0.468098 | -0.063673 | -0.071902 | 0.142472 | 0.219512 | 0.193182 | 0.448864 | 0.394726 |
| T8 | 45 | 0.12 | 0.461169 | -0.070602 | -0.078831 | 0.158254 | 0.317073 | 0.079545 | 0.306818 | 0.402189 |
| T2 | 35 | 0.10 | 0.436085 | -0.095686 | -0.103915 | 0.134966 | 0.341463 | 0.068182 | 0.488636 | 0.367280 |
| T5 | 40 | 0.10 | 0.435122 | -0.096649 | -0.104878 | 0.143280 | 0.512195 | 0.102273 | 0.543561 | 0.341384 |
| T4 | 40 | 0.08 | 0.429584 | -0.102187 | -0.110416 | 0.145919 | 0.341463 | 0.215909 | 0.587121 | 0.333695 |
| T9 | 55 | 0.10 | 0.400008 | -0.131763 | -0.139992 | 0.136314 | 0.292683 | 0.181818 | 0.337121 | 0.326610 |

## Interpretation

This local high-pass sweep did not improve mAP50. No candidate beats the prior G4 anchor, and no candidate approaches the teammate threshold of `0.54`.

The best row, T1 (`sigma=35 alpha=0.08`), has only `0.486738` mAP50. It has low miss rate but high healthy false positives, so it is not a useful replacement.

The likely reason is that moving sigma down from 50 over-emphasizes finer texture/edge detail. That can increase sensitivity for some diseased samples, but it appears to damage the broader shape/context features YOLO needs under the grouped split. Moving sigma up to 55 also hurts, so the useful region may be narrow around the original `sigma=50 alpha=0.10`.

## Decision

Drop this local high-pass sweep as a route to `0.56-0.57` mAP50.

Keep prior G4 as the current best deterministic Fourier setting:

- high-pass `sigma=50 alpha=0.10`
- clean-light aug on
- default hook on

## Recommended Next Actions

1. Do not continue dense sigma/alpha tuning around 35-55 unless T0 is rerun for sanity.
2. If the goal is mAP50 above 0.54, shift from deterministic preprocessing-only tuning to interaction tuning:
   - Fourier G4 + YOLO augmentation policy tuning
   - Fourier G4 + threshold/calibration sweep
   - Fourier G4 + train-only or mixed original/transformed training while evaluating on original/test-transform variants
3. For paper integrity, once a new winner appears, confirm against the clean baseline and prior G4 over seeds 123 and 3407.
