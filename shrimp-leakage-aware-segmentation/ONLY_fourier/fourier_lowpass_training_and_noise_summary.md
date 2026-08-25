# Fourier Low-Pass Training and Noise-Robustness Summary

## Scope

This report evaluates the low-pass Fourier sweep from notebook 047 against:

- the strict no-Fourier/no-augmentation control;
- the previously selected W1 high-pass Fourier model; and
- the previously evaluated augmentation-only methods where the same noisy-test protocol is available.

All low-pass runs use the original grouped-specimen split (`split_fingerprint = 1ffd4a...9fcb`), YOLO segmentation, no custom augmentation, disabled YOLO augmentation hook, and the same seed (`42`). For noisy evaluation, noise is applied first and Fourier preprocessing is then applied to the noisy image for the Fourier models.

Evidence files:

- [Low-pass training paper rows](C:/Users/Admin/Downloads/047_only_fourier_lowpass_sweep_noise_paper_row.csv)
- [Low-pass noisy-test evaluation](C:/Users/Admin/Downloads/047_lowpass_noise_evaluation_summary.csv)
- [Low-pass executed notebook](C:/Users/Admin/Downloads/047-fourier-lowpass-sweep-noise-kaggle-seed42.ipynb)
- [Previous Panel A noisy-test summary](C:/Users/Admin/Downloads/panel_a_noise_evaluation_summary.txt)
- [Previous Panel A paper rows](C:/Users/Admin/Downloads/45_only_fourier_panel_a_noise_robustness_kaggle_seed42_paper_row.txt)

## 1. Clean Training/Evaluation Metrics

The table below uses the clean-test rows recorded by the training sweep. The no-Fourier row is the control generated in the same sweep. W1 values are taken from the previous Panel A clean evaluation because W1 was not rerun in notebook 047.

| Method | Fourier setting | Labeled test mAP50 | Labeled test mAP50-95 | Healthy FP rate | Disease miss rate | Healthy-aware score |
|---|---|---:|---:|---:|---:|---:|
| Nothing baseline | No Fourier, no augmentation | 0.2354 | not extracted | 0.3902 | 0.2273 | 0.1408 |
| Low-pass | sigma=25 | 0.2602 | not extracted | 0.3171 | 0.2386 | 0.1497 |
| Low-pass | sigma=50 | 0.1832 | not extracted | 0.3171 | 0.3750 | 0.0642 |
| Low-pass | sigma=100 | **0.3150** | not extracted | 0.3171 | 0.4318 | **0.1860** |
| W1 high-pass | W1 | **approximately 0.3381** | not extracted | 0.6098 | 0.1705 | **0.2120** |

### Clean-data interpretation

- Low-pass sigma=100 is the strongest low-pass setting and improves over the no-Fourier control by approximately `+0.0797` labeled mAP50 and `+0.0452` healthy-aware score.
- W1 remains better on clean disease segmentation mAP50 and disease recall.
- Low-pass settings reduce healthy false positives relative to W1, but sigma=100 has a much higher disease miss rate.
- Sigma=50 is not competitive and should be rejected.

## 2. Noisy-Test Performance

The following values are labeled-test mask mAP50. The healthy-aware score is included because raw mAP alone can hide harmful false positives on healthy images.

### 2.1 Gaussian noise

| Method | mAP50 | Healthy FP rate | Disease miss rate | Healthy-aware score |
|---|---:|---:|---:|---:|
| W1 high-pass | 0.1324 | 0.5366 | 0.2500 | 0.0072 |
| No-Fourier control | 0.1050 | 0.4878 | 0.2841 | -0.0205 |
| Low-pass sigma=25 | 0.2244 | 0.2927 | 0.2273 | 0.1168 |
| Low-pass sigma=50 | 0.2027 | 0.2927 | 0.4205 | 0.0816 |
| Low-pass sigma=100 | **0.2894** | **0.2683** | 0.4773 | **0.1559** |

Low-pass sigma=100 substantially outperforms W1 under Gaussian noise: `+0.1570` mAP50 and `+0.1487` healthy-aware score.

### 2.2 Salt-and-pepper noise

| Method | mAP50 | Healthy FP rate | Disease miss rate | Healthy-aware score |
|---|---:|---:|---:|---:|
| W1 high-pass | 0.1139 | 0.7805 | 0.2273 | -0.0438 |
| No-Fourier control | 0.0497 | 0.4878 | 0.3750 | -0.0899 |
| Low-pass sigma=25 | 0.2182 | 0.2927 | 0.2841 | 0.1064 |
| Low-pass sigma=50 | 0.1744 | 0.2927 | 0.4773 | 0.0432 |
| Low-pass sigma=100 | **0.2470** | **0.1951** | 0.4432 | **0.1292** |

Low-pass sigma=100 is again clearly strongest: `+0.1331` mAP50 and `+0.1730` healthy-aware score over W1.

### 2.3 Other noise types

| Noise | Best observed method | mAP50 | Healthy-aware score | Comment |
|---|---|---:|---:|---|
| Color cast | W1 high-pass | 0.3092 | 0.1790 | Low-pass sigma=100 falls to 0.1982 / 0.0352 |
| Low contrast | N2 scale+erasing by score; W1 by mAP50 | 0.2074 / 0.2075 | 0.0719 / 0.0563 | Low-pass sigma=100: 0.2024 / 0.0388 |
| Motion blur | W1 by mAP50; N1 flipud by score | 0.2800 / 0.2633 | 0.1370 / 0.1724 | Low-pass sigma=25 is competitive at 0.2467 / 0.1448 |

The prior augmentation methods were augmentation-only at inference. Their noisy-test rows are included in the [Panel A summary](C:/Users/Admin/Downloads/panel_a_noise_evaluation_summary.txt). They generally control healthy false positives better than W1, but do not match low-pass sigma=100 on Gaussian or salt-and-pepper corruption.

## 3. Rationale and Decision

### Why low-pass helps

Gaussian and salt-and-pepper noise inject high-frequency changes. Low-pass filtering suppresses those components before inference, which explains the large reduction in healthy false positives and the mAP50 recovery under those two corruptions.

### Why low-pass also hurts

Disease boundaries and small lesions can also contain high-frequency information. Removing too much high-frequency content increases disease misses. This is most visible with sigma=100: it has the best noisy mAP50, but its disease miss rate is substantially higher than W1.

### Recommended status

1. **Retain W1 as the general clean-data model.** It has the strongest clean mAP50 and better disease recall.
2. **Retain low-pass sigma=100 as a noise-specialist candidate.** It is clearly superior under Gaussian and salt-and-pepper noise.
3. **Keep low-pass sigma=25 as a secondary candidate for motion blur.** It is less destructive than sigma=100 and competitive on that corruption.
4. **Reject sigma=50.** It is inferior to sigma=25 and sigma=100 in both clean and noisy evaluations.
5. **Do not claim that low-pass universally improves the model.** The evidence supports a conditional robustness benefit, not a general replacement for W1.

## Caveats

- W1 was not retrained in notebook 047; its comparison values come from the earlier Panel A evaluation.
- The low-pass sweep is a single seed (`42`), so the robustness ranking should be treated as a candidate result until confirmed across additional grouped-specimen seeds.
- The noisy-test rows evaluate each model with its intended preprocessing. For low-pass models the order is `noise -> Fourier -> inference`; for augmentation-only models it is `noise -> inference`.
- Healthy-aware score should be interpreted together with disease miss rate. A higher score produced mainly by suppressing predictions can conceal loss of disease recall.
