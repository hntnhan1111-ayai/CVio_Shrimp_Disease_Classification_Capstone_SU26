# Factorial A-H Ablation: Fourier High-Pass s50 a0.10

Source CSV: `only_fourier_factorial_A_H_highpass_s50_a0p10_summary.csv`

Fixed contract:

- split: stratified grouped-specimen, seed 42
- split fingerprint: `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`
- model: `yolo11n-seg.pt`
- Fourier ON: high-pass `sigma=50`, `alpha=0.10`, applied to train/valid/test
- clean-light augmentation ON: explicit YOLO clean-light policy
- default hook ON: Ultralytics default Albumentations hook restored

## Main Table

| Mode | Fourier | Clean-light aug | Default hook | mAP50 | mAP50-95 | Healthy FP | FP/img | Miss rate | Count MAE | Healthy-aware |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | off | off | off | 0.235374 | 0.060555 | 0.390244 | 0.560976 | 0.227273 | 0.429924 | 0.140762 |
| B | on | off | off | 0.345105 | 0.088484 | 0.634146 | 0.926829 | 0.193182 | 0.839015 | 0.210763 |
| C | off | on | off | 0.473408 | 0.158613 | 0.390244 | 0.439024 | 0.090909 | 0.518939 | 0.394800 |
| D | off | on | on | 0.512002 | 0.167785 | 0.243902 | 0.268293 | 0.079545 | 0.321970 | 0.459581 |
| E | on | on | off | 0.408162 | 0.124771 | 0.365854 | 0.390244 | 0.125000 | 0.357955 | 0.334929 |
| F | on | off | on | 0.299415 | 0.082874 | 0.414634 | 0.439024 | 0.306818 | 0.732955 | 0.175281 |
| G | on | on | on | 0.531771 | 0.170904 | 0.243902 | 0.292683 | 0.102273 | 0.437500 | 0.470165 |
| H | off | off | on | 0.306111 | 0.077802 | 0.365854 | 0.414634 | 0.272727 | 0.626894 | 0.197272 |

## Key Effects

Fourier effect:

| Comparison | mAP50 delta | Healthy FP delta | Miss delta | Healthy-aware delta | Interpretation |
|---|---:|---:|---:|---:|---|
| B - A: Fourier with no aug, hook off | +0.109731 | +0.243902 | -0.034091 | +0.070000 | Improves disease detection, but causes severe healthy FP. |
| F - H: Fourier with no aug, hook on | -0.006696 | +0.048780 | +0.034091 | -0.021991 | Not useful with hook alone. |
| E - C: Fourier with clean aug, hook off | -0.065246 | -0.024390 | +0.034091 | -0.059871 | Hurts clean-light hook-off baseline. |
| G - D: Fourier with clean aug, hook on | +0.019769 | +0.000000 | +0.022727 | +0.010584 | Best interaction: small gain over strongest baseline, no healthy FP-rate increase. |

Clean-light augmentation effect:

| Comparison | mAP50 delta | Healthy FP delta | Miss delta | Healthy-aware delta |
|---|---:|---:|---:|---:|
| C - A: clean aug without Fourier, hook off | +0.238034 | +0.000000 | -0.136364 | +0.254038 |
| E - B: clean aug with Fourier, hook off | +0.063057 | -0.268293 | -0.068182 | +0.124166 |

Default hook effect:

| Comparison | mAP50 delta | Healthy FP delta | Miss delta | Healthy-aware delta |
|---|---:|---:|---:|---:|
| H - A: hook only, no Fourier/clean aug | +0.070737 | -0.024390 | +0.045455 | +0.056509 |
| F - B: hook with Fourier only | -0.045690 | -0.219512 | +0.113636 | -0.035482 |
| D - C: hook with clean aug only | +0.038594 | -0.146341 | -0.011364 | +0.064781 |
| G - E: hook with Fourier + clean aug | +0.123609 | -0.121951 | -0.022727 | +0.135236 |

## Interpretation

Fourier high-pass `sigma=50`, `alpha=0.10` is not independently strong. It helps the bare baseline, but mostly by increasing sensitivity and causing many healthy false positives. It also hurts when paired with clean-light augmentation but the default hook is disabled.

The important result is the three-way interaction. Fourier becomes useful only when clean-light augmentation and the default hook are both enabled:

- D, no Fourier + clean aug + hook: healthy-aware 0.459581
- G, Fourier + clean aug + hook: healthy-aware 0.470165

G is the best row by mAP50 and healthy-aware score:

- mAP50: 0.531771
- mAP50-95: 0.170904
- healthy FP rate: 0.243902
- healthy-aware score: 0.470165

The gain over D is modest but meaningful for seed 42:

- mAP50: +0.019769
- mAP50-95: +0.003119
- healthy FP rate: unchanged
- healthy-aware score: +0.010584

The downside is a small increase in missed diseased images and count error:

- disease miss rate: 0.079545 -> 0.102273
- count MAE: 0.321970 -> 0.437500

## Decision

Keep G as the current best frequency-boosting candidate:

`Fourier high-pass sigma50 alpha0.10 + clean-light YOLO augmentation + default hook on`

Do not claim Fourier alone is the final method. The supported claim is narrower and stronger:

> Frequency boosting is complementary to the full clean augmentation pipeline, but only when the default augmentation hook is enabled. Without that full augmentation context, Fourier either increases healthy false positives or reduces diseased recall.

## Next Actions

1. Repeat mode D vs G across seeds 123 and 3407 first. These are the only two rows needed for confirmation.
2. If G remains better, run a smaller Fourier parameter check only under clean-light + hook-on:
   - high-pass `sigma=50`, `alpha=0.05`
   - high-pass `sigma=50`, `alpha=0.10`
   - high-frequency damping `sigma=50`, `alpha=0.10`
3. Add threshold/calibration sweeps for D and G, because G has better mAP but slightly worse disease miss/count MAE.
4. Build a visual diagnostics notebook comparing D and G predictions on:
   - healthy false positives
   - diseased misses
   - corrected detections
   - new errors introduced by Fourier
