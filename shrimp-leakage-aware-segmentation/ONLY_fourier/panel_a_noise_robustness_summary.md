# Panel A Noise Robustness: Fourier vs Matched Alternatives

Source files:

- `45_only_fourier_panel_a_noise_robustness_kaggle_seed42_paper_row.txt`
- `panel_a_noise_evaluation_summary.txt`
- Executed notebook: `045-run.ipynb`

All methods used the same grouped-specimen split, seed 42, test images, and corruption definitions. The comparison is a single-seed screening result, not multi-seed confirmation.

## Methods

| Method | Description | Fourier at inference |
|---|---|---:|
| W1 | Fourier high-pass, `sigma=50`, `alpha=0.10` | Yes |
| Erasing | Offline erasing, `0.10` | No |
| FlipUD | Offline vertical flip, `0.30` | No |
| Scale + Erasing | Offline scale `0.10` + erasing `0.10` | No |

For W1, each corrupted test image was processed as `noise -> Fourier -> YOLO`. For the other methods, inference was `noise -> YOLO`; their training-time offline augmentation was not applied to the test images.

## Clean Reference

The notebook's stored delta columns used the training paper-row metric as the clean denominator. The actual `clean_reference` evaluation rows should be used instead.

| Method | Paper-row clean labeled mask mAP50 | Actual clean-reference mAP50 | Difference |
|---|---:|---:|---:|
| W1 | 0.345105 | 0.338093 | -0.007012 |
| Erasing | 0.300018 | 0.314321 | +0.014303 |
| FlipUD | 0.268234 | 0.272135 | +0.003901 |
| Scale + Erasing | 0.342583 | 0.329938 | -0.012646 |

The noisy metrics below are valid. Any delta or retention claim should be recomputed relative to the actual clean-reference value.

## Noisy Test Results

Values are labeled-only test mask mAP50. The bold value is the best method for that corruption condition.

| Noise condition | W1 Fourier | Erasing | FlipUD | Scale + Erasing |
|---|---:|---:|---:|---:|
| Gaussian, sigma 20 | 0.132355 | **0.175936** | 0.125218 | 0.170465 |
| Salt-and-pepper, p 0.03 | 0.113905 | **0.120217** | 0.116965 | 0.117798 |
| Blue color cast, +40 | **0.309241** | 0.300049 | 0.165302 | 0.279277 |
| Low contrast, factor 0.50 | 0.207488 | **0.235327** | 0.134388 | 0.207375 |
| Motion blur, kernel 11 | **0.280040** | 0.250525 | 0.263318 | 0.250728 |
| **Mean across noisy conditions** | **0.208606** | **0.216411** | 0.161038 | 0.205129 |

## Interpretation

### Fourier W1

W1 is strongest under blue color cast and motion blur. This is consistent with a frequency-domain transformation preserving or emphasizing useful structural detail when the corruption is primarily photometric or blur-related. It is not strongest against Gaussian or salt-and-pepper noise; high-pass processing can preserve or emphasize those high-frequency disturbances.

### Erasing

Erasing has the highest mean noisy mAP50 and the highest mean retention after correcting the denominator. It is strongest for Gaussian noise, salt-and-pepper noise, and low contrast. However, this does not mean erasing reproduces Fourier behavior: its false-positive and disease-miss tradeoffs are different.

### FlipUD

FlipUD is the weakest overall noise-robustness candidate by mean mAP50. It performs reasonably under motion blur but is substantially weaker under color cast and low contrast.

### Scale + Erasing

Scale + Erasing is competitive under Gaussian noise and has a strong low-contrast healthy false-positive result, but it does not lead any noisy mAP50 condition and has lower mean noisy mAP50 than Erasing.

## Healthy-Aware Behavior

The noise response is not captured by mAP50 alone. Selected healthy false-positive rates were:

| Noise condition | W1 Fourier | Erasing | FlipUD | Scale + Erasing |
|---|---:|---:|---:|---:|
| Gaussian | 0.537 | 0.195 | 0.268 | 0.537 |
| Salt-and-pepper | 0.780 | 0.098 | 0.341 | 0.561 |
| Blue color cast | 0.488 | 0.195 | 0.146 | 0.220 |
| Low contrast | 0.195 | 0.122 | 0.098 | **0.024** |
| Motion blur | 0.756 | 0.268 | 0.317 | 0.366 |

W1's main weakness is false-positive amplification under salt-and-pepper and motion blur. Therefore, the claim should be method-specific: Fourier is promising for some structured corruptions, but it is not a general-purpose noise suppressor.

## Inference-Time Evidence

The log does **not** contain isolated inference time for Fourier versus the other methods.

The notebook records `noise_eval_time_sec`, but this timer surrounds the complete `panel_a_metric_row` operation. That operation includes:

- full-test validation;
- labeled-only validation;
- prediction-error counting;
- healthy-image false-positive evaluation;
- metric extraction and temporary evaluation-dataset creation.

It therefore cannot be interpreted as pure YOLO inference time, nor as the cost of Fourier preprocessing alone. The paper-row `train_time_min` is training time, not inference time.

## Recommended Follow-Up Timing Test

Use the already trained checkpoints and the same clean/noisy test images. For each method, measure separately:

1. image loading and conversion;
2. Fourier preprocessing time, for W1 only;
3. YOLO prediction time with warm-up excluded from the measurement;
4. total per-image latency;
5. peak GPU memory.

Use identical `imgsz`, batch size, device, number of images, confidence settings, and warm-up policy. Report mean and p95 milliseconds per image. This can be done without retraining or changing the noise experiment.

## Bottom Line

The existing log is sufficient for the current robustness comparison after recomputing deltas from `clean_reference`. No retraining is required. A separate timed inference pass is required before making a latency claim about Fourier versus non-Fourier methods.
