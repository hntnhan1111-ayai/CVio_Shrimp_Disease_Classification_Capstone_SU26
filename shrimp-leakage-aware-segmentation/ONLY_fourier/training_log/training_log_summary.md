# ONLY_fourier Training Log Summary

Source folder: `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/`

All Fourier runs below used:

- split: stratified grouped-specimen, seed 42
- split fingerprint: `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`
- model: `yolo11n-seg.pt`
- hidden Ultralytics Albumentations hook: off
- explicit YOLO augmentation: off for Fourier runs
- training: 100 epochs requested, patience 40
- selection: best checkpoint by validation metrics, test retained for audit/reporting

The strict model+data baseline is `001_clean-aug-off-hook-off-baseline.ipynb`: no explicit YOLO augmentation, hidden hook off, no Fourier. This is the correct anchor for "how far Fourier can go without augmentation." The older `00_clean-aug-hook-off.ipynb` row is a clean-light augmentation control, not the bare baseline.

## Validity Notes

- Fixed all-splits Fourier runs transformed train/valid/test with GPU Fourier, CPU fallback 0, OOM splits 0.
- `08` random train augmentation added 905 Fourier train copies to the original 905 train images: actual train image files 1,810. GPU copies 905, CPU fallback 0, OOM splits 0. Valid/test stayed original.
- `09` mixed original plus fixed high-pass added 905 Fourier train copies to the original 905 train images: actual train image files 1,810. GPU copies 905, CPU fallback 0, OOM splits 0. Valid/test stayed original.
- `10` train-only high-pass transformed only the 905 train images with GPU Fourier. Valid/test stayed original.
- `11` alpha sweep transformed train/valid/test separately for each alpha; the summary CSV contains the per-alpha rows.

## Main Results

| Run | Policy | Best epoch | Epochs run | Train images used | Eval domain | mAP50 | mAP50-95 | Healthy FP | FP/img | Miss rate | Count MAE | Healthy-aware |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Strict no-aug hook-off baseline | no Fourier, no YOLO aug, hook off | 20 | 60 | 905 | original | 0.235374 | 0.060555 | 0.390244 | 0.560976 | 0.227273 | 0.429924 | 0.140762 |
| Clean aug hook-off control | no Fourier, clean-light YOLO aug | - | 100 | 905 | original | 0.473408 | 0.158613 | 0.390244 | 0.439024 | 0.090909 | 0.518939 | 0.394800 |
| High-pass a0.1 s50 | fixed all-splits high-pass | 10 | 50 | 905 | Fourier | 0.345105 | 0.088484 | 0.634146 | 0.926829 | 0.193182 | 0.839015 | 0.210763 |
| High-pass a0.3 s50 | fixed all-splits high-pass | 45 | 85 | 905 | Fourier | 0.242205 | 0.080400 | 0.146341 | 0.146341 | 0.522727 | 0.647727 | 0.116776 |
| Band-pass l12 h60 a0.2 | fixed all-splits band-pass | 48 | 88 | 905 | Fourier | 0.295219 | 0.083854 | 0.268293 | 0.268293 | 0.431818 | 0.630682 | 0.172083 |
| Band-pass l20 h80 a0.2 | fixed all-splits band-pass | 52 | 92 | 905 | Fourier | 0.286980 | 0.088123 | 0.195122 | 0.195122 | 0.522727 | 0.621212 | 0.157998 |
| High-frequency damping a0.1 s50 | fixed all-splits damping | 11 | 51 | 905 | Fourier | 0.321119 | 0.106764 | 0.390244 | 0.414634 | 0.522727 | 0.592803 | 0.174045 |
| Low-frequency flatten b0.3 s100 | fixed all-splits lowfreq flatten | 14 | 54 | 905 | Fourier | 0.278693 | 0.085208 | 0.146341 | 0.219512 | 0.465909 | 0.551136 | 0.166616 |
| Homomorphic gl0.70 gh1.20 s50 | fixed all-splits homomorphic | 22 | 62 | 905 | Fourier | 0.261197 | 0.091581 | 0.414634 | 0.731707 | 0.238636 | 0.666667 | 0.150605 |
| Random high-pass train aug a0.05-0.20 | original + random Fourier train copies | 14 | 54 | 1,810 | original | 0.283641 | 0.081329 | 0.146341 | 0.170732 | 0.602273 | 0.685606 | 0.144385 |
| Mixed original + high-pass a0.1 train aug | original + fixed Fourier train copies | 28 | 68 | 1,810 | original | 0.242476 | 0.073790 | 0.146341 | 0.170732 | 0.454545 | 0.541667 | 0.132577 |
| Train-only high-pass a0.1 | train transformed, valid/test original | 10 | 50 | 905 | original | 0.336343 | 0.091158 | 0.609756 | 0.902439 | 0.193182 | 0.873106 | 0.202734 |
| High-pass a0.05 s50 | fixed all-splits high-pass | 20 | 60 | 905 | Fourier | 0.308774 | 0.079226 | 0.341463 | 0.463415 | 0.250000 | 0.562500 | 0.209003 |
| High-pass a0.15 s50 | fixed all-splits high-pass | 16 | 71 | 905 | Fourier | 0.248861 | 0.072006 | 0.146341 | 0.170732 | 0.511364 | 0.630682 | 0.125988 |
| High-pass a0.20 s50 | fixed all-splits high-pass | 24 | 65 | 905 | Fourier | 0.291338 | 0.082667 | 0.292683 | 0.341463 | 0.329545 | 0.539773 | 0.185649 |
| Hook-on clean baseline anchor | no Fourier, clean-light YOLO aug, hook on | 56 | 86 | 905 | original | 0.511424 | 0.163868 | 0.243902 | 0.268293 | 0.079545 | 0.331439 | 0.458530 |

## Ranking Among No-Aug Hook-Off Fourier Runs

By labeled test mAP50:

1. High-pass a0.1 all-splits: 0.345105
2. Train-only high-pass a0.1 eval-original: 0.336343
3. High-frequency damping a0.1: 0.321119
4. High-pass a0.05 all-splits: 0.308774
5. Band-pass l12 h60 a0.2: 0.295219
6. High-pass a0.20 all-splits: 0.291338
7. Band-pass l20 h80 a0.2: 0.286980
8. Random high-pass train augmentation: 0.283641
9. Low-frequency flatten b0.3: 0.278693
10. Homomorphic gl0.70 gh1.20: 0.261197
11. High-pass a0.15 all-splits: 0.248861
12. Mixed original + high-pass a0.1 train augmentation: 0.242476
13. High-pass a0.3 all-splits: 0.242205

Strict no-aug hook-off baseline reference: 0.235374.

By healthy-aware test score:

1. High-pass a0.1 all-splits: 0.210763
2. High-pass a0.05 all-splits: 0.209003
3. Train-only high-pass a0.1 eval-original: 0.202734
4. High-pass a0.20 all-splits: 0.185649
5. High-frequency damping a0.1: 0.174045
6. Band-pass l12 h60 a0.2: 0.172083
7. Low-frequency flatten b0.3: 0.166616
8. Band-pass l20 h80 a0.2: 0.157998
9. Homomorphic gl0.70 gh1.20: 0.150605
10. Random high-pass train augmentation: 0.144385
11. Mixed original + high-pass a0.1 train augmentation: 0.132577
12. High-pass a0.15 all-splits: 0.125988
13. High-pass a0.3 all-splits: 0.116776

Strict no-aug hook-off baseline reference: 0.140762.

By healthy FP suppression:

1. High-pass a0.3 all-splits: 0.146341
2. Low-frequency flatten b0.3: 0.146341
3. Random high-pass train augmentation: 0.146341
4. Mixed original + high-pass a0.1 train augmentation: 0.146341
5. High-pass a0.15 all-splits: 0.146341
6. Band-pass l20 h80 a0.2: 0.195122
7. Band-pass l12 h60 a0.2: 0.268293
8. High-pass a0.20 all-splits: 0.292683
9. High-pass a0.05 all-splits: 0.341463
10. High-frequency damping a0.1: 0.390244
11. Homomorphic gl0.70 gh1.20: 0.414634
12. Train-only high-pass a0.1 eval-original: 0.609756
13. High-pass a0.1 all-splits: 0.634146

## Interpretation

The new train-signal experiments did not rescue Fourier. They mostly repeat the earlier tradeoff:

- Adding Fourier train copies strongly suppresses healthy false positives, but it also causes severe disease misses.
- Train-only high-pass keeps disease sensitivity closer to high-pass a0.1, but healthy FP remains very high.
- The weak-alpha sweep confirms there is no obvious sweet spot between alpha 0.05 and 0.30. Alpha 0.05 and 0.10 preserve more disease sensitivity but produce too many healthy false positives. Alpha 0.15 and 0.30 suppress healthy false positives but become too conservative. Alpha 0.20 sits between these behaviors but still does not approach the clean-light controls.

Against the strict no-augmentation hook-off baseline, Fourier can improve mAP50: high-pass a0.1 rises from 0.235374 to 0.345105, and train-only high-pass a0.1 reaches 0.336343. This means Fourier is not useless as a no-augmentation signal. However, the improvement is unstable because the best mAP50 settings also produce high healthy false positives.

The strongest Fourier-only healthy-aware score is still high-pass a0.1 all-splits at 0.210763, essentially tied with high-pass a0.05 at 0.209003. These improve over the strict no-aug baseline healthy-aware score of 0.140762, but they remain far below the clean-light hook-off control at 0.394800 and the hook-on clean baseline anchor at 0.458530.

## Decision

Do not continue broad global Fourier high-pass variants as final performance candidates. The current evidence is enough to say global frequency preprocessing/augmentation can beat the bare no-augmentation baseline, but it does not approach clean augmentation on this dataset under grouped evaluation.

Keep the Fourier runs as an analysis section, not as the final method:

- They show that Fourier can boost a weak no-augmentation baseline.
- They demonstrate a measurable sensitivity-specificity tradeoff.
- They show healthy FP can be reduced by stronger or mixed frequency transforms, but only by losing diseased recall.
- They motivate a more selective frequency-domain method if this paper continues: lesion/body-localized transforms, learned frequency gates, or frequency perturbations combined with a healthy-FP-aware objective.

## Recommended Next Step

Move away from image-global deterministic Fourier. The next technically meaningful notebook should be a visual/diagnostic one:

- original vs transformed image
- YOLO label overlay
- predicted masks from baseline vs Fourier model
- separated by healthy false positives, disease misses, and correct detections

This will show whether Fourier is enhancing disease texture, shell artifacts, boundary contrast, or background/lighting artifacts.
