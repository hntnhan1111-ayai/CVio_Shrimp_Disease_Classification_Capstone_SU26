# Consolidated Fourier Effects Report

## Purpose

This report consolidates the current evidence for three questions:

1. How Fourier interacts with other training approaches and augmentation methods, including Fourier-before/after augmentation order.
2. How Fourier and matched alternatives behave under noisy test conditions.
3. How Fourier changes inference latency.

The report uses raw CSV/JSON artifacts and executed notebooks from both `ONLY_fourier/training_log/` and `C:\Users\Admin\Downloads`. Existing Markdown summaries were used as indexes, then checked against the underlying artifacts where available.

## Common Evaluation Contract

The later controlled experiments use:

- YOLO11n-seg;
- seed 42;
- stratified grouped-specimen split;
- split fingerprint `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`;
- 905 train, 115 validation, and 129 test images;
- hidden Albumentations hook disabled for the ONLY_fourier controls;
- batch 16 and workers 0 for the training comparisons;
- labeled-only test mask mAP50 as the primary ranking metric;
- healthy FP rate, disease miss rate, mask-count MAE, and healthy-aware score as safety metrics.

These are predominantly single-seed screening experiments. They support directional conclusions, not generalization claims across seeds.

## Executive Conclusions

- Fourier is strongly regime-dependent. It helps the bare no-augmentation control and the clean-light hook-on baseline, but it slightly hurts the strongest matched baseline-architecture strong-augmentation recipe.
- Fourier is beneficial inside the reproduced SimAM+CA attention branch, where it raises labeled test mAP50 from `0.493754` to `0.556110`.
- The best current matched strong-augmentation result is **without Fourier**: `0.595095` labeled test mAP50 versus `0.577126` with Fourier.
- Across N=1 and N=2 offline augmentation studies, Fourier is not a general improvement. It is positive for selected Degrees, FlipLR, FlipUD, Erasing, Translate+Erasing, Scale+FlipUD, and Translate+Degrees settings, but often lowers the absolute result.
- There is no universal best order. Augmentation-before-Fourier is safer for Scale and most FlipUD/FlipLR cases; Fourier-before-augmentation is stronger for selected Erasing, Degrees, and composed pairs.
- On noisy test images, W1 Fourier is best for blue color cast and motion blur. Erasing has the highest mean noisy mAP50 and retention across the tested corruptions.
- W1 is much slower at inference. With single-image Fourier timing, effective latency is `87.38 ms/image` versus `12.69-12.82 ms/image` for the three alternatives.

## 1. Fourier With Other Approaches

### 1.1 Matched Fourier On/Off Regimes

These comparisons come from the final metrics registry and the Factorial A-H / Path 15 records. Fourier is high-pass `sigma=50`, `alpha=0.10` unless stated otherwise.

| Regime | Fourier off mAP50 | Fourier on mAP50 | Delta | Healthy FP off -> on | Healthy-aware off -> on | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Bare YOLO, no explicit augmentation, hook off | 0.235374 | 0.345105 | +0.109731 | 0.390244 -> 0.634146 | 0.140762 -> 0.210763 | Large diseased mAP gain, but healthy false positives increase sharply. |
| No clean-light augmentation, default hook on | 0.306111 | 0.299415 | -0.006696 | 0.365854 -> 0.414634 | 0.197272 -> 0.175281 | Fourier does not help this hook-only regime. |
| Clean-light augmentation, hook off | 0.473408 | 0.408162 | -0.065246 | 0.390244 -> 0.365854 | 0.394800 -> 0.334929 | Fourier lowers mAP despite a small FP improvement. |
| Clean-light augmentation, hook on | 0.512002 | 0.531771 | +0.019769 | 0.243902 -> 0.243902 | 0.459581 -> 0.470165 | Small positive and relatively balanced Fourier effect. |
| Strong augmentation, hook off, baseline YOLO11n-seg | 0.595095 | 0.577126 | -0.017969 | 0.341463 -> 0.317073 | 0.532634 -> 0.501480 | Fourier slightly lowers the best raw recipe while reducing FP. |
| Strong augmentation, hook off, SimAM+CA | 0.493754 | 0.556110 | +0.062356 | 0.365854 -> 0.268293 | 0.422699 -> 0.494527 | Fourier clearly helps this attention branch. |

### 1.2 What This Means

Fourier does not simply add a fixed amount of useful information. Its effect depends on what the model already receives:

- With weak input diversity, Fourier can supply useful contrast/detail cues.
- With clean-light augmentation and the hidden hook enabled, Fourier gives a modest additional gain.
- With strong augmentation and the baseline architecture, augmentation already produces a stronger representation and Fourier becomes slightly harmful on raw mAP50.
- With SimAM+CA, Fourier and attention appear complementary in this seed-42 reproduction, but the no-Fourier attention control is weaker than the matched baseline-architecture strong-augmentation control.

The Path 15 comparison is especially important: the current best result is not caused by Fourier. The best matched baseline is strong augmentation without Fourier at `0.595095`.

### 1.3 N=1 Fourier Interaction by Augmentation Family

The following table summarizes the detailed N=1 experiments. `Delta` is Fourier-on mAP50 minus the matched augmentation-only mAP50. The best order is selected by the larger Fourier-on mAP50 for that setting.

| Family | Setting | Aug-only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Better order |
|---|---:|---:|---:|---:|---:|---:|---|
| Translate | 0.05 | 0.321386 | 0.265800 | -0.055586 | 0.319359 | -0.002027 | Fourier -> Aug |
| Translate | 0.08 | 0.378245 | 0.350022 | -0.028223 | 0.300384 | -0.077861 | Aug -> Fourier |
| Scale | 0.10 | 0.369081 | 0.318537 | -0.050544 | 0.230283 | -0.138798 | Aug -> Fourier |
| Scale | 0.20 | 0.333299 | 0.294439 | -0.038860 | 0.269252 | -0.064047 | Aug -> Fourier |
| FlipLR | 0.25 | 0.327451 | 0.333991 | +0.006540 | 0.365107 | +0.037656 | Fourier -> Aug |
| FlipLR | 0.50 | 0.291080 | 0.292293 | +0.001213 | 0.284207 | -0.006873 | Aug -> Fourier |
| Degrees | 3 | 0.310849 | 0.297214 | -0.013635 | 0.359537 | +0.048688 | Fourier -> Aug |
| Degrees | 5 | 0.307844 | 0.344493 | +0.036649 | 0.336845 | +0.029001 | Aug -> Fourier |
| Erasing | 0.10 | 0.338450 | 0.262397 | -0.076053 | 0.233618 | -0.104832 | Aug -> Fourier |
| Erasing | 0.15 | 0.198143 | 0.272676 | +0.074533 | 0.290710 | +0.092567 | Fourier -> Aug |
| FlipUD | 0.15 | 0.302918 | 0.306720 | +0.003802 | 0.296649 | -0.006269 | Aug -> Fourier |
| FlipUD | 0.30 | 0.334533 | 0.345064 | +0.010531 | 0.282653 | -0.051880 | Aug -> Fourier |

N=1 HSV was tested through an earlier online YOLO path and is therefore less directly comparable to the later offline experiments:

| HSV setting | HSV-only | Fourier -> HSV | Delta | HSV -> Fourier | Delta |
|---|---:|---:|---:|---:|---:|
| Mild | 0.280658 | 0.245578 | -0.035080 | 0.280301 | -0.000357 |
| Medium | 0.305874 | 0.228811 | -0.077063 | 0.276781 | -0.029093 |
| Strong | 0.295270 | 0.334920 | +0.039650 | 0.299051 | +0.003781 |

N=1 conclusion: the strongest positive interactions are Degrees `5`, Erasing `0.15`, Degrees `3` in the Fourier-before-augmentation order, and FlipLR `0.25`. Scale and Translate are consistently harmed by Fourier.

### 1.4 N=2 Fourier Interaction

| Pair | Aug-only | Aug -> Fourier | Delta | Fourier -> Aug | Delta | Best Fourier order |
|---|---:|---:|---:|---:|---:|---|
| Translate + Scale | 0.318218 | 0.307586 | -0.010632 | 0.300178 | -0.018041 | Aug -> Fourier |
| Translate + Erasing | 0.224153 | 0.265065 | +0.040912 | 0.302729 | +0.078576 | Fourier -> Aug |
| Translate + FlipUD | 0.316863 | 0.265739 | -0.051124 | 0.302382 | -0.014481 | Fourier -> Aug |
| Scale + Erasing | 0.342583 | 0.246335 | -0.096249 | 0.280140 | -0.062444 | Fourier -> Aug |
| Translate + FlipLR | 0.412850 | 0.339754 | -0.073096 | 0.319382 | -0.093468 | Neither; augmentation-only wins |
| Scale + FlipUD | 0.271389 | 0.237733 | -0.033656 | 0.327079 | +0.055690 | Fourier -> Aug |
| Scale + FlipLR | 0.322144 | 0.324253 | +0.002109 | 0.250022 | -0.072122 | Aug -> Fourier |
| Translate + Degrees | 0.291783 | 0.308670 | +0.016887 | 0.310324 | +0.018541 | Fourier -> Aug |

The best augmentation-only result is Translate `0.08` + FlipLR `0.25` at `0.412850`; Fourier reduces it in both orders. The strongest positive N=2 Fourier delta is Scale `0.10` + FlipUD `0.30` with Fourier-before-augmentation, rising from `0.271389` to `0.327079`. This is a positive interaction, but not the best absolute model.

### 1.5 Order Findings

The order is an interaction factor, not a universal rule.

**Augmentation -> Fourier is usually safer for:**

- Scale;
- FlipUD;
- FlipLR at the stronger probability;
- Translate + Scale;
- Scale + FlipLR;
- Translate at the stronger tested value.

**Fourier -> Augmentation is often better for:**

- Erasing `0.15`;
- Degrees `3`;
- FlipLR `0.25`;
- Translate + Erasing;
- Scale + FlipUD;
- Translate + Degrees.

The original Notebook 23 HSV order experiment needs a caveat. Its W1-only pair compares cached offline Fourier with online Fourier inside a hook, even though HSV gains are zero in the W1-only row. That pair is an execution-path comparison, not a clean causal order comparison. The nonzero mild/medium/strong HSV rows are the more defensible order evidence.

## 2. Fourier and Alternatives on Noisy Test Sets

The source is the Panel A execution of `045-run.ipynb`, summarized in `panel_a_noise_robustness_summary.md` and the downloaded Panel A evaluation text. Four checkpoints were evaluated on the same corrupted test images.

| Method | Training approach | Fourier at inference |
|---|---|---:|
| W1 | High-pass `sigma=50`, `alpha=0.10` | Yes |
| Erasing | Offline erasing `0.10` | No |
| FlipUD | Offline vertical flip `0.30` | No |
| Scale + Erasing | Offline scale `0.10` + erasing `0.10` | No |

### 2.1 Noisy Labeled-Test Mask mAP50

| Noise condition | W1 Fourier | Erasing | FlipUD | Scale + Erasing | Best |
|---|---:|---:|---:|---:|---|
| Gaussian, sigma 20 | 0.132355 | **0.175936** | 0.125218 | 0.170465 | Erasing |
| Salt-and-pepper, p 0.03 | 0.113905 | **0.120217** | 0.116965 | 0.117798 | Erasing |
| Blue color cast, +40 | **0.309241** | 0.300049 | 0.165302 | 0.279277 | W1 |
| Low contrast, factor 0.50 | 0.207488 | **0.235327** | 0.134388 | 0.207375 | Erasing |
| Motion blur, kernel 11 | **0.280040** | 0.250525 | 0.263318 | 0.250728 | W1 |
| **Mean noisy mAP50** | 0.208606 | **0.216411** | 0.161038 | 0.205129 | Erasing |

Mean retention, correctly recomputed against each method's actual `clean_reference` row, was:

| Method | Mean retention across five noisy conditions |
|---|---:|
| Erasing | **0.6885** |
| Scale + Erasing | 0.6217 |
| W1 Fourier | 0.6170 |
| FlipUD | 0.5918 |

### 2.2 Interpretation

- W1 is strongest for blue color cast and motion blur.
- Erasing is strongest overall across the five corruptions and is best for Gaussian, salt-and-pepper, and low contrast.
- FlipUD is weakest overall.
- Scale + Erasing is intermediate and does not lead any noisy mAP50 condition.
- W1 is not a universal noise suppressor. High-pass processing can preserve or emphasize high-frequency corruption, which is consistent with its weak Gaussian and salt-and-pepper results.

W1 also has a healthy false-positive problem under some corruptions: its healthy FP rate is `0.780488` for salt-and-pepper and `0.756098` for motion blur. Erasing has much lower FP rates under those same conditions (`0.097561` and `0.268293`). Therefore raw mAP50 alone is insufficient for deployment conclusions.

### 2.3 Clean-Reference Audit

The notebook's stored `delta_mAP50_vs_clean` and retention fields used the training paper-row metric as the denominator, not the separately evaluated `clean_reference` row. The raw noisy mAP50 values are valid, but those stored delta/retention fields must be recomputed.

| Method | Training paper-row clean mAP50 | Actual clean-reference mAP50 |
|---|---:|---:|
| W1 | 0.345105 | 0.338093 |
| Erasing | 0.300018 | 0.314321 |
| FlipUD | 0.268234 | 0.272135 |
| Scale + Erasing | 0.342583 | 0.329938 |

No retraining or noise rerun is required for this correction. The corrected values above and the raw noisy rows are sufficient for the current report.

## 3. Inference Time

The source is `46_inference_timing_all_methods_seed42.json` and the executed `46-log.ipynb`. This is the second timing run with `FOURIER_BATCH_SIZE = 1`, which is the appropriate setting for single-image deployment latency.

All four methods used 129 test images, 119 timed predictions, 10 warm-up images, CUDA, `imgsz=640`, and the same split fingerprint.

| Method | Fourier preprocessing | YOLO inference | Effective latency |
|---|---:|---:|---:|
| W1 Fourier | 48.62 ms/image | 38.77 ms/image | **87.38 ms/image** |
| Erasing | 0 ms/image | 12.69 ms/image | **12.69 ms/image** |
| FlipUD | 0 ms/image | 12.69 ms/image | **12.69 ms/image** |
| Scale + Erasing | 0 ms/image | 12.82 ms/image | **12.82 ms/image** |

W1 is approximately `6.9x` slower overall than the alternatives. Fourier preprocessing alone contributes approximately `48.6 ms/image`.

The W1 YOLO inference component is also higher than the other three models (`38.77 ms` versus approximately `12.7 ms`). Because all models were timed sequentially in one run, this component should be confirmed with repeated randomized model order before claiming that the checkpoint itself is intrinsically slower. The Fourier preprocessing overhead is nevertheless directly measured and clear.

### Inference-Time Difference: Potential Culprits

The four checkpoints are all YOLO11n-seg models, so a large difference in the neural-network forward pass is not expected from the architecture alone. The current notebook times the complete call:

```python
model.predict(source=image, imgsz=640, device=0, ...)
```

That timing includes input conversion, resizing/letterboxing, tensor transfer, model forward execution, mask/box postprocessing, and NMS. It is therefore an end-to-end prediction timing, not a pure YOLO-forward timing.

The most plausible contributors to the W1-versus-other difference are:

1. **Fourier array memory layout.** Raw images are returned directly by OpenCV, while W1 images are produced through a tensor permutation and conversion to NumPy. The resulting arrays may be non-contiguous, causing an additional copy or conversion inside Ultralytics.
2. **More prediction candidates.** High-pass processing can amplify edges and noise. W1's higher false-positive behavior suggests that it may generate more candidate boxes/masks, increasing NMS and postprocessing time.
3. **Fixed benchmark order.** W1 was timed first in the single run. CUDA initialization, kernel selection, GPU clock state, and memory allocation can make the first model slower even after warm-up.

Consequently, the measured `87.38 ms/image` is a valid estimate of the current W1 deployment pipeline, but the `38.77 ms` W1 YOLO component should not yet be interpreted as proof that its YOLO network is intrinsically slower than the other three. A diagnostic timing pass should make Fourier outputs contiguous with `np.ascontiguousarray`, randomize model order, repeat each model several times, and record Ultralytics' internal `result[0].speed` fields separately for preprocessing, pure inference, and postprocessing.

Timing exclusions and inclusions:

- Excluded: disk loading and output rendering.
- Included: Fourier image transform, Ultralytics image preprocessing, model forward pass, and prediction postprocessing.
- Warm-up predictions were excluded from the reported prediction statistics.

## Source Verification and Caveats

Primary sources used:

- `training_log/only_fourier_factorial_A_H_highpass_s50_a0p10_summary.csv`
- `training_log/only_fourier_consolidated_training_metrics.csv`
- `training_log/training_log_summary.md`
- `[hand-explanation]final_metrics_and_training_configs_registry.md`
- `fourier_augmentation_interaction_order_summary.md`
- downloaded N=1/N=2 paper-row CSVs in `C:\Users\Admin\Downloads`
- `panel_a_noise_robustness_summary.md`
- `45_only_fourier_panel_a_noise_robustness_kaggle_seed42_paper_row.txt`
- `panel_a_noise_evaluation_summary.txt`
- `46_inference_timing_all_methods_seed42.json`
- `46-log.ipynb`

Important limitations:

1. Most training comparisons are single seed 42.
2. N=1 HSV used an earlier online YOLO augmentation implementation, while later geometric and erasing experiments used offline transformations.
3. Notebook 23's W1-only order pair is confounded by cached versus online Fourier execution.
4. The N=2 experiments vary Fourier position relative to the composed augmentation, not the internal order of the two augmentation operations.
5. The older `training_log/only_fourier_running_metrics_report.md` is incomplete for the final Path 15 comparison because it omits the later Mode A result. The final metrics registry is preferred for that matched comparison.
6. FDDem and other feature-Fourier variants are separate architectural Fourier mechanisms, not directly interchangeable with image high-pass W1. They should be reported as separate ablations rather than merged into the image-Fourier order tables.

## Final Assessment

The current evidence supports this paper-level framing:

> Fourier high-pass preprocessing is useful in weak/no-augmentation and selected clean-light or attention-enhanced regimes, but its benefit is not universal. Its interaction with augmentation depends on family, strength, and operation order. The strongest current matched segmentation recipe uses strong augmentation without Fourier, while Fourier introduces a substantial single-image inference cost and provides selective rather than general noise robustness.

For maximum clean-test mAP50, the current leading direction is strong augmentation without Fourier. For a Fourier-focused analysis, retain W1 as a regime-dependent ablation and report its healthy false-positive and latency costs. Do not claim that Fourier is the overall best method based on the current seed-42 evidence.

## 4. Fourier Configuration Selection: Why W1 Was Chosen

This section traces the Fourier-selection notebooks and logs used to choose W1. Here, W1 means the image-level high-pass boost:

```text
high-pass boost, sigma=50, alpha=0.10
```

W1 is also called `G4` when used inside the clean-light augmentation plus default-hook-on Mode G regime. The Fourier transform is the same; the surrounding training regime is different.

### 4.1 Strict No-Augmentation Image-Fourier Search

The first selection stage used the strict control: no explicit YOLO augmentation, hidden hook off, and Fourier applied to train/valid/test. The strict no-Fourier baseline was labeled test mAP50 `0.235374` and healthy-aware score `0.140762`.

| Configuration | Transform | Labeled mAP50 | mAP50-95 | Healthy FP | Disease miss | Healthy-aware |
|---|---|---:|---:|---:|---:|---:|
| Baseline | No Fourier | 0.235374 | 0.060555 | 0.390244 | 0.227273 | 0.140762 |
| W1 | High-pass, sigma 50, alpha 0.10 | **0.345105** | 0.088484 | 0.634146 | 0.193182 | **0.210763** |
| High-pass alpha 0.05 | High-pass, sigma 50, alpha 0.05 | 0.308774 | 0.079226 | 0.341463 | 0.250000 | 0.209003 |
| High-pass alpha 0.15 | High-pass, sigma 50, alpha 0.15 | 0.248861 | 0.072006 | 0.146341 | 0.511364 | 0.125988 |
| High-pass alpha 0.20 | High-pass, sigma 50, alpha 0.20 | 0.291338 | 0.082667 | 0.292683 | 0.329545 | 0.185649 |
| High-pass alpha 0.30 | High-pass, sigma 50, alpha 0.30 | 0.242205 | 0.080400 | 0.146341 | 0.522727 | 0.116776 |
| Band-pass narrow | Low 12, high 60, alpha 0.20 | 0.295219 | 0.083854 | 0.268293 | 0.431818 | 0.172083 |
| Band-pass wide | Low 20, high 80, alpha 0.20 | 0.286980 | 0.088123 | 0.195122 | 0.522727 | 0.157998 |
| High-frequency damping | Sigma 50, alpha 0.10 | 0.321119 | **0.106764** | 0.390244 | 0.522727 | 0.174045 |
| Low-frequency flattening | Sigma 100, beta 0.30 | 0.278693 | 0.085208 | 0.146341 | 0.465909 | 0.166616 |
| Homomorphic | Sigma 50, gamma-low 0.70, gamma-high 1.20 | 0.261197 | 0.091581 | 0.414634 | 0.238636 | 0.150605 |
| Random high-pass train copies | Train-only random alpha 0.05-0.20 | 0.283641 | 0.081329 | 0.146341 | 0.602273 | 0.144385 |
| Mixed original + high-pass copies | Train-only fixed alpha 0.10 | 0.242476 | 0.073790 | 0.146341 | 0.454545 | 0.132577 |

Selection result: W1 was the best strict no-augmentation configuration by labeled mAP50 and healthy-aware score. It improved mAP50 by `+0.109731` over the strict baseline, although the gain came with a substantial healthy-FP increase. Alpha values above `0.10` became too destructive to disease recall; weaker or alternative filters did not match W1's mAP50.

### 4.2 Mode G Configuration Search With Clean-Light Augmentation

The next selection stage evaluated Fourier configurations in the clean-light YOLO augmentation plus default-hook-on regime. This is the regime in which W1 is named G4. The matched no-Fourier Mode D anchor was labeled mAP50 `0.512002` and healthy-aware score `0.459581`.

| Mode | Transform | Sigma | Alpha / parameter | Labeled mAP50 | mAP50-95 | Healthy FP | Miss | Healthy-aware |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G4 / W1 | High-pass boost | 50 | 0.10 | **0.531771** | 0.170904 | **0.243902** | 0.102273 | **0.470165** |
| G3 | High-pass boost | 50 | 0.07 | 0.502473 | 0.163647 | 0.317073 | 0.125000 | 0.429194 |
| G6 | High-frequency damping | 50 | 0.05 | 0.486759 | 0.161947 | 0.268293 | 0.125000 | 0.422903 |
| G8 | High-frequency damping | 70 | 0.10 | 0.471856 | **0.174861** | 0.463415 | **0.034091** | 0.403071 |
| G7 | High-frequency damping | 50 | 0.10 | 0.464590 | 0.152450 | 0.292683 | 0.159091 | 0.389393 |
| G1 | High-pass boost | 50 | 0.03 | 0.484988 | 0.165772 | 0.414634 | 0.215909 | 0.387085 |
| G2 | High-pass boost | 50 | 0.05 | 0.422659 | 0.148011 | 0.341463 | 0.136364 | 0.347414 |
| G5 | High-pass boost | 50 | 0.15 | 0.378245 | 0.134318 | 0.390244 | 0.125000 | 0.294902 |
| Mode D anchor | No Fourier | - | - | 0.512002 | 0.167785 | 0.243902 | 0.079545 | 0.459581 |

W1/G4 was the only tested Fourier configuration that beat the no-Fourier Mode D anchor on both labeled mAP50 and healthy-aware score:

- mAP50: `0.512002 -> 0.531771` (`+0.019769`);
- mAP50-95: `0.167785 -> 0.170904` (`+0.003119`);
- healthy FP: unchanged at `0.243902`;
- healthy-aware score: `0.459581 -> 0.470165` (`+0.010584`).

G8 is worth noting diagnostically: it has the best mAP50-95 and lowest disease miss rate in this table, but its healthy FP rate is `0.463415`, making it unsuitable as the headline configuration. G6 has lower count MAE than G4 but loses on mAP50 and healthy-aware score.

### 4.3 Local Sigma/Alpha Sweep Around W1

The local sweep tested nearby sigma/alpha combinations while keeping the Mode G surrounding regime fixed. The intended T0 G4 repeat was absent from the downloaded sweep CSV, so comparison uses the prior G4 anchor.

| Mode | Sigma | Alpha | Labeled mAP50 | Delta vs G4 | Healthy FP | Miss | Healthy-aware |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior G4 / W1 | 50 | 0.10 | **0.531771** | 0.000000 | 0.243902 | 0.102273 | **0.470165** |
| T1 | 35 | 0.08 | 0.486738 | -0.045033 | 0.414634 | **0.056818** | 0.418192 |
| T6 | 40 | 0.12 | 0.484870 | -0.046901 | 0.341463 | 0.125000 | 0.412087 |
| T3 | 35 | 0.12 | 0.476721 | -0.055050 | 0.414634 | 0.102273 | 0.402776 |
| T7 | 45 | 0.10 | 0.468098 | -0.063673 | 0.219512 | 0.193182 | 0.394726 |
| T8 | 45 | 0.12 | 0.461169 | -0.070602 | 0.317073 | 0.079545 | 0.402189 |
| T2 | 35 | 0.10 | 0.436085 | -0.095686 | 0.341463 | 0.068182 | 0.367280 |
| T5 | 40 | 0.10 | 0.435122 | -0.096649 | 0.512195 | 0.102273 | 0.341384 |
| T4 | 40 | 0.08 | 0.429584 | -0.102187 | 0.341463 | 0.215909 | 0.333695 |
| T9 | 55 | 0.10 | 0.400008 | -0.131763 | 0.292683 | 0.181818 | 0.326610 |

No local candidate beat W1. The best T1 row had lower disease miss rate, but its mAP50 and healthy-aware score were both worse and its healthy FP rate increased. Moving sigma below or above 50 did not improve the headline result, suggesting that the useful region is narrow around the original W1 setting in this regime.

### 4.4 Separate Feature-Fourier / FDDem Selection

FDDem is not the same operation as W1. W1 is an image-level offline high-pass transform; FDDem injects learnable or fixed Fourier feature refinement into the network. These results are included to show why FDDem was not merged into the W1 selection table.

| FDDem family / mode | Fourier mechanism | Labeled mAP50 | mAP50-95 | Healthy FP | Miss | Healthy-aware |
|---|---|---:|---:|---:|---:|---:|
| G2 conservative | P3/P4/P5 one-band, gamma 0.01 | 0.309129 | 0.086619 | 0.170732 | 0.375000 | 0.208818 |
| G3 | P4/P5 two bands, identity gamma | 0.300441 | 0.086397 | 0.268293 | 0.409091 | 0.183934 |
| G4 | FEM P3 prototype | 0.233096 | 0.065546 | 0.317073 | 0.522727 | 0.092487 |
| V3 | FDDem cutoff 0.14 | 0.319630 | 0.084989 | 0.243902 | 0.420455 | 0.199596 |
| Tuned T1 | FDDem cutoff 0.12 | 0.328544 | **0.108696** | 0.365854 | 0.215909 | **0.230217** |
| U0 baseline | No Fourier feature module | 0.235374 | 0.060555 | 0.390244 | 0.227273 | 0.140762 |
| W1 image high-pass | Sigma 50, alpha 0.10 | **0.345105** | 0.088484 | 0.634146 | 0.193182 | 0.210763 |

FDDem T1 is competitive on healthy-aware score and mAP50-95, but its raw mAP50 remains below W1. FDDem should therefore be described as a separate architectural branch, not as evidence that a different W1 image filter was better.

### 4.5 Selection Decision

W1 was selected for the main image-Fourier path because it satisfied all three selection checkpoints:

1. **Strict control:** best image-Fourier mAP50 and healthy-aware score among the initial no-augmentation filters.
2. **Augmented working regime:** best Mode G Fourier configuration, beating the matched clean-light no-Fourier anchor.
3. **Local robustness of choice:** no nearby sigma/alpha candidate beat it in the local sweep.

This does not mean W1 is globally optimal for every task objective. Its costs are explicit:

- high healthy FP in the strict no-augmentation regime;
- selective rather than universal noise robustness;
- large single-image preprocessing latency;
- reduced mAP50 when added to the strongest baseline-architecture strong augmentation.

The defensible reason for retaining W1 is therefore **best validated image-level Fourier configuration under the selected working regime**, not “Fourier always improves the model.”
