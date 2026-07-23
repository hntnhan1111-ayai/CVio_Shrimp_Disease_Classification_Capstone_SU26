# Mode G Fourier Tuning Seed 42 Analysis

Source workbook: `C:\Users\Admin\Downloads\only_fourier_mode_G_fourier_tuning_seed42_summary.xlsx`

Compact paper-row CSV checked later:

- `C:\Users\Admin\Downloads\only_fourier_mode_G_fourier_tuning_seed42_paper_row.csv`
- shape: 7 rows x 56 columns
- contains: G2, G3, G4, G5, G6, G7, G8
- missing: G0 and G1

G0 was intentionally skipped by the user. G1 was present in the full `.xlsx` summary, so the paper-row CSV is not a complete compact export of the full tuning run.

Fixed contract:

- split: stratified grouped-specimen
- seed: 42
- split fingerprint: `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`
- model: `yolo11n-seg.pt`
- clean-light YOLO augmentation: on
- default Ultralytics Albumentations hook: on
- epochs: 100
- patience: 40
- Fourier preprocessing: GPU, zero failed images

Note: G0 / no-Fourier Mode D reference was skipped in this run. Comparisons below use the prior Mode D seed-42 anchor.

## Main Test Ranking

| Mode | Transform | Sigma | Alpha | mAP50 | mAP50-95 | Healthy FP | FP/img | Miss | Count MAE | Healthy-aware |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G4 | high-pass boost | 50 | 0.10 | 0.531771 | 0.170904 | 0.243902 | 0.292683 | 0.102273 | 0.437500 | 0.470165 |
| G3 | high-pass boost | 50 | 0.07 | 0.502473 | 0.163647 | 0.317073 | 0.414634 | 0.125000 | 0.456439 | 0.429194 |
| G6 | high-frequency damping | 50 | 0.05 | 0.486759 | 0.161947 | 0.268293 | 0.292683 | 0.125000 | 0.365530 | 0.422903 |
| G8 | high-frequency damping | 70 | 0.10 | 0.471856 | 0.174861 | 0.463415 | 0.731707 | 0.034091 | 0.346591 | 0.403071 |
| G7 | high-frequency damping | 50 | 0.10 | 0.464590 | 0.152450 | 0.292683 | 0.292683 | 0.159091 | 0.441288 | 0.389393 |
| G1 | high-pass boost | 50 | 0.03 | 0.484988 | 0.165772 | 0.414634 | 0.414634 | 0.215909 | 0.481061 | 0.387085 |
| G2 | high-pass boost | 50 | 0.05 | 0.422659 | 0.148011 | 0.341463 | 0.341463 | 0.136364 | 0.412879 | 0.347414 |
| G5 | high-pass boost | 50 | 0.15 | 0.378245 | 0.134318 | 0.390244 | 0.463415 | 0.125000 | 0.511364 | 0.294902 |

## Comparison Against Prior Mode D Anchor

Prior Mode D anchor:

- mAP50: 0.512002
- mAP50-95: 0.167785
- Healthy FP: 0.243902
- FP/img: 0.268293
- Miss: 0.079545
- Count MAE: 0.321970
- Healthy-aware: 0.459581

Only G4 beats Mode D on healthy-aware score:

- mAP50: +0.019769
- mAP50-95: +0.003119
- Healthy FP: unchanged
- FP/img: +0.024390
- Miss: +0.022728
- Count MAE: +0.115530
- Healthy-aware: +0.010584

## Interpretation

The sweep does not find a better Fourier setting than the original Mode G configuration. High-pass `sigma=50 alpha=0.10` remains the best current candidate.

Lower high-pass strengths did not solve the healthy false-positive issue. `alpha=0.07` is second by healthy-aware score, but it loses mAP against Mode D and raises healthy FP to 0.317073. `alpha=0.03` and `alpha=0.05` are worse.

Stronger high-pass `alpha=0.15` collapses performance and should be dropped.

High-frequency damping is not a better headline candidate. G8 has the best mAP50-95 and lowest disease miss rate, but it produces too many healthy false positives. G6 has better count MAE than G4 but lower mAP and lower healthy-aware score.

## Decision

Keep:

- G4: high-pass boost `sigma=50 alpha=0.10`

Drop for headline:

- G1, G2, G3, G5
- G6, G7, G8

Possible diagnostic-only follow-up:

- inspect G8 predictions because it has low miss rate and high mAP50-95 but fails healthy false-positive control
- inspect G6 predictions because it has lower count MAE than G4 but weaker detection score

Next experiment should not keep broadening this seed-42 hyperparameter sweep. Confirm D vs G4 over seeds 123 and 3407, then add a confidence-threshold sweep for D and G4.
