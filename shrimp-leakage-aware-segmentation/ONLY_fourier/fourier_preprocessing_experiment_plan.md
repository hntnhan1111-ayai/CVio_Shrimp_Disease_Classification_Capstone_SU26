# ONLY Fourier Preprocessing Experiment Plan

Date: 2026-06-17

## Objective

Map which deterministic Fourier-domain preprocessing filters can improve shrimp disease segmentation under the strict control setting:

- YOLO model: `yolo11n-seg.pt`
- Split: stratified grouped-specimen split, seed `42`
- Hidden Ultralytics Albumentations hook: off
- Explicit YOLO augmentation: off
- Fourier transform applied before train/valid/test evaluation dataset construction
- Metrics: labeled-only mask mAP50, mAP50-95, healthy false-positive rate, disease miss rate, count MAE, healthy-aware score

## Current Anchors

| Run | Labeled mAP50 | Healthy FP | Disease Miss | Healthy-Aware |
|---|---:|---:|---:|---:|
| No aug, hook off, no Fourier | `0.235374` | `0.390244` | `0.227273` | `0.140762` |
| Clean-light YOLO aug, hook off, no Fourier | `0.473408` | `0.390244` | `0.090909` | `0.394800` |
| High-pass `sigma=50`, `alpha=0.30` | `0.242205` | `0.146341` | `0.522727` | `0.116776` |
| High-pass `sigma=50`, `alpha=0.10` | `0.345105` | `0.634146` | `0.193182` | `0.210763` |

## Interpretation So Far

High-pass strength controls a sensitivity/conservatism tradeoff:

- `alpha=0.10` improves disease mAP but overfires on healthy shrimp.
- `alpha=0.30` suppresses healthy false positives but misses too much disease.

The next step is not only alpha tuning. We need filters that shape which frequency bands are affected.

## Candidate Notebook Checklist

### 03 Band-Pass Boost, Narrow Mid-Band

- Notebook: `only_fourier_03_bandpass_l12_h60_a0p20_no_aug_hook_off.ipynb`
- Method: boost frequencies between Gaussian low-pass widths `12` and `60`
- Params: `low_sigma=12`, `high_sigma=60`, `alpha=0.20`
- Hypothesis: recover lesion texture while avoiding the finest shell/glare noise.
- Pass condition:
  - labeled test mAP50 `> 0.345`
  - healthy FP `< 0.50`
  - healthy-aware score `> 0.210`

### 04 Band-Pass Boost, Wider Mid-Band

- Notebook: `only_fourier_04_bandpass_l20_h80_a0p20_no_aug_hook_off.ipynb`
- Method: boost a wider mid-frequency band
- Params: `low_sigma=20`, `high_sigma=80`, `alpha=0.20`
- Hypothesis: a wider band may preserve larger lesion structure while still avoiding full high-pass over-sensitivity.
- Pass condition:
  - labeled test mAP50 `> 0.345`
  - healthy FP `< 0.50`
  - healthy-aware score `> 0.210`

### 05 High-Frequency Damping

- Notebook: `only_fourier_05_highfreq_damping_s50_a0p10_no_aug_hook_off.ipynb`
- Method: subtract a small high-frequency residual
- Params: `sigma=50`, `alpha=0.10`
- Hypothesis: suppress healthy shell/noise false positives without the severe disease miss rate seen from strong high-pass behavior.
- Pass condition:
  - healthy FP `< 0.30`
  - labeled test mAP50 `>= 0.235`
  - disease miss rate `<= 0.30`

### 06 Low-Frequency Illumination Flattening

- Notebook: `only_fourier_06_lowfreq_flatten_s100_b0p30_no_aug_hook_off.ipynb`
- Method: subtract low-frequency illumination variation while preserving mean brightness
- Params: `sigma=100`, `beta=0.30`
- Hypothesis: reduce lighting/background bias while preserving local disease cues.
- Pass condition:
  - labeled test mAP50 `> 0.235`
  - healthy FP `<= 0.390`
  - healthy-aware score `> 0.140`

### 07 Homomorphic Filtering

- Notebook: `only_fourier_07_homomorphic_s50_gl0p70_gh1p20_no_aug_hook_off.ipynb`
- Method: log-domain low-frequency attenuation plus mild high-frequency gain
- Params: `sigma=50`, `gamma_low=0.70`, `gamma_high=1.20`
- Hypothesis: handle multiplicative illumination while mildly improving lesion texture.
- Pass condition:
  - labeled test mAP50 `> 0.300`
  - healthy FP `<= 0.45`
  - healthy-aware score `> 0.210`

## Run Order

1. Band-pass narrow mid-band.
2. Band-pass wider mid-band.
3. High-frequency damping.
4. Low-frequency flattening.
5. Homomorphic filtering.

## Recording Checklist

For each run:

- [ ] Confirm split fingerprint is `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`.
- [ ] Confirm hook is disabled.
- [ ] Confirm YOLO augmentation args are all zero/none.
- [ ] Confirm Fourier preprocessing logs show train/valid/test image counts.
- [ ] Confirm GPU transform count equals all image counts or document CPU fallback.
- [ ] Record best epoch and epochs run.
- [ ] Record labeled test mAP50 and mAP50-95.
- [ ] Record healthy FP rate and FP masks/image.
- [ ] Record disease miss rate.
- [ ] Record count MAE.
- [ ] Record healthy-aware validation and test scores.
- [ ] Mark result: `KEEP`, `TUNE`, or `DROP`.

## Decision Rules

- `KEEP`: improves healthy-aware score and does not create a large new FP or miss-rate failure.
- `TUNE`: improves one axis strongly but fails another; adjust strength/band before discarding.
- `DROP`: worse than no-Fourier on healthy-aware score with no useful tradeoff.

## Expected Next Tuning

If band-pass helps, tune:

- `alpha`: `0.10`, `0.15`, `0.25`
- low/high sigma pairs: `(10, 50)`, `(15, 70)`, `(25, 90)`

If damping helps, tune:

- `alpha`: `0.05`, `0.15`, `0.20`
- `sigma`: `30`, `70`

