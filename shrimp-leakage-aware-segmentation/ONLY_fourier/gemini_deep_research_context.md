# Gemini Deep Research Context: Shrimp Disease Segmentation Optimization

Use this file as the full context for a deep research task. The goal is to collect technically plausible, literature-backed ideas for improving a leakage-safe YOLOv11n-seg shrimp disease segmentation pipeline. Fourier/frequency-domain methods are important, but the search should not be limited to Fourier.

## Research Request

We are working on shrimp disease instance segmentation using YOLOv11n-seg. The active goal is to find methods that can improve held-out grouped-specimen segmentation performance beyond the current best known runs.

Please conduct a broad but implementation-oriented deep research pass and return:

1. Literature-backed ideas that could improve this task.
2. Concrete experiment plans that can be implemented as Kaggle notebooks.
3. Ranked hypotheses with expected benefit, risk, and rationale.
4. Suggested hyperparameter ranges and ablation design.
5. Notes about what failure modes each idea targets.
6. Warnings about methods that are likely to inflate metrics or violate the grouped-specimen evaluation discipline.

## Hard Exclusion

Do not propose U-Net, U-Net++, nnU-Net, or U-Net-like encoder-decoder segmentation models as the main direction. We want to stay in the YOLO/instance-segmentation/detection-style family or use preprocessing, augmentation, training-policy, loss/objective, inference-time, attention-module, or frequency-domain ideas around that family.

## Main Objective

Primary target:

- Improve labeled test mask mAP50.
- We want to beat `0.54`; ideally reach `0.56-0.57+`.

Secondary diagnostics:

- Healthy false-positive rate.
- Disease miss rate.
- Count MAE.
- Healthy-aware score.

Healthy FP is not the main target for this next research pass, but it remains diagnostically useful. A method that improves mAP50 while catastrophically increasing healthy false positives should be treated as risky, not automatically accepted.

## Important Recent Result

The most recent inspected notebook log confirms:

- Fourier G4 high-pass + strong YOLO augmentation + hook off + baseline YOLO11n-seg achieved `0.577126` labeled test mAP50.

Use this as a strong signal that:

- We should not be too narrow about Fourier-only methods.
- Heavy augmentation or augmentation-policy interaction may be important.
- Fourier may help most when combined with stronger augmentation or training-policy changes.
- Attention is not automatically beneficial for top mAP50: SimAM+CA with Fourier reached `0.556110`, lower than the baseline architecture Fourier run, but with better healthy FP, miss rate, and count MAE.

## Project Context

Dataset:

- Hand-labeled shrimp disease instance segmentation dataset.
- Approximate size: 1149 images, 416 shrimp specimens, 1031 masks.
- Disease labels include BG, WSSV, WSSV_BG, and Healthy/background-style specimens.
- There are healthy images with no disease masks.
- Some co-infection images contain both BG and WSSV labels.

Evaluation protocol:

- Stratified grouped-specimen split.
- Same shrimp/specimen must never appear across train/valid/test.
- Primary seed currently: `42`.
- Split fingerprint used in many runs: `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb`.
- Model: `yolo11n-seg.pt`.
- Training usually requests 100 epochs with patience 40.
- Best checkpoint selected by validation metrics.
- Test metrics retained for audit/reporting.

Important experiment knobs:

- Fourier on/off.
- Fourier transform type and hyperparameters.
- Clean-light YOLO augmentation on/off.
- Ultralytics hidden Albumentations/default augmentation hook on/off.
- Heavy YOLO augmentation policies.
- Possible architecture attention modules, if still in YOLO-style family.

## Accepted Baseline / Anchor

The current hook-enabled clean-light baseline contract from the active optimization plan:

| Item | Value |
|---|---:|
| Model | `yolo11n-seg.pt` |
| Split | stratified grouped-specimen |
| Seed | `42` |
| Explicit augmentation | clean-light YOLO augmentation |
| Hidden Ultralytics Albumentations hook | enabled |
| Full test mask mAP50 | `0.481532` |
| Labeled-only test mask mAP50 | `0.512002` |
| Labeled-only test mask mAP50-95 | `0.167785` |
| Healthy test FP rate | `0.243902` |
| Healthy-aware labeled test score | `0.459581` |

Clean-light explicit YOLO augmentation policy:

```python
{
    "auto_augment": None,
    "erasing": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.35,
    "hsv_v": 0.20,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.20,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": 0.0,
    "bgr": 0.0,
}
```

## Current Interpretation Before New Research

Earlier Fourier-only experiments showed that global deterministic Fourier preprocessing can beat the strict no-augmentation baseline, but it was weak compared with clean-light augmentation. Later factorial runs suggest Fourier becomes much more useful when combined with clean-light augmentation and the YOLO default hook.

Current strongest verified table row:

- Mode G: Fourier high-pass s50 a0.10 + clean-light augmentation + YOLO default hook on.
- Labeled test mAP50: `0.531771`.
- Healthy FP: `0.243902`.
- Healthy-aware score: `0.470165`.

New verified best:

- Fourier G4 + strong augmentation + hook off + baseline YOLO11n-seg: `0.577126` mAP50.
- Fourier G4 + strong augmentation + hook off + SimAM+CA: `0.556110` mAP50.

This suggests the next research should focus on interactions:

- Fourier + heavy augmentation.
- Fourier + stronger but biologically plausible photometric/geometric policies.
- Fourier + mosaic/copy-paste/mixup/cutmix-style policies.
- Fourier + attention modules or lightweight architectural changes, excluding U-Net.
- Frequency-aware augmentation that changes train distribution without corrupting disease masks.
- Inference-time policies such as TTA or confidence/IoU threshold calibration.

## Full Current Metrics Tables

The following tables collect the current local logs before the newest unlogged `0.577` result.

### Mode Key

| Mode family | Mode | Meaning |
| --- | --- | --- |
| Factorial A-H | A | Fourier off, clean-light aug off, YOLO default hook off |
| Factorial A-H | B | Fourier high-pass s50 a0.10 on, clean-light aug off, YOLO default hook off |
| Factorial A-H | C | Fourier off, clean-light aug on, YOLO default hook off |
| Factorial A-H | D | Fourier off, clean-light aug on, YOLO default hook on |
| Factorial A-H | E | Fourier high-pass s50 a0.10 on, clean-light aug on, YOLO default hook off |
| Factorial A-H | F | Fourier high-pass s50 a0.10 on, clean-light aug off, YOLO default hook on |
| Factorial A-H | G | Fourier high-pass s50 a0.10 on, clean-light aug on, YOLO default hook on |
| Factorial A-H | H | Fourier off, clean-light aug off, YOLO default hook on |
| Local sweep T1-T9 | T1-T9 | Mode G kept fixed; high-pass alpha/sigma varied locally |
| Mode G Fourier tuning | G* | Mode G kept fixed; Fourier transform/type or hyperparameters varied. Only rows saved in the notebook output are included here. |
| Path 15 interaction | B | Fourier G4 high-pass s50 a0.10, strong aug, hook off, baseline YOLO11n-seg |
| Path 15 interaction | C | No Fourier, strong aug, hook off, SimAM+CA |
| Path 15 interaction | D | Fourier G4 high-pass s50 a0.10, strong aug, hook off, SimAM+CA |

### Overall Leaderboard by Labeled Test mAP50

| Rank | Family | Run | Policy / Name | mAP50 | Healthy FP | Miss | Healthy-aware | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | path_15_fourier_G4_strong_aug_attention | B | Fourier G4 + strong aug + hook off + baseline YOLO11n-seg | 0.577126 | 0.317073 | 0.147727 | 0.501480 | notebook_training_log |
| 2 | path_15_fourier_G4_strong_aug_attention | D | Fourier G4 + strong aug + hook off + SimAM+CA | 0.556110 | 0.268293 | 0.113636 | 0.494527 | notebook_training_log |
| 3 | factorial_A_H_highpass_s50_a0p10 | G | - | 0.531771 | 0.243902 | 0.102273 | 0.470165 | csv_summary |
| 4 | mode_G_fourier_tuning_seed42 | G4 | - | 0.531771 | 0.243902 | 0.102273 | 0.470165 | notebook_rendered_table |
| 5 | factorial_A_H_highpass_s50_a0p10 | D | - | 0.512002 | 0.243902 | 0.079545 | 0.459581 | csv_summary |
| 6 | training_log_summary_main_results | Hook-on clean baseline anchor | no Fourier, clean-light YOLO aug, hook on | 0.511424 | 0.243902 | 0.079545 | 0.458530 | markdown_main_results |
| 7 | path_15_fourier_G4_strong_aug_attention | C | no Fourier + strong aug + hook off + SimAM+CA | 0.493754 | 0.365854 | 0.125000 | 0.422699 | notebook_training_log |
| 8 | mode_G_highpass_local_sweep_seed42 | T1 | - | 0.486738 | 0.414634 | 0.056818 | 0.418192 | csv_summary |
| 9 | mode_G_highpass_local_sweep_seed42 | T6 | - | 0.484870 | 0.341463 | 0.125000 | 0.412087 | csv_summary |
| 10 | mode_G_highpass_local_sweep_seed42 | T3 | - | 0.476721 | 0.414634 | 0.102273 | 0.402776 | csv_summary |
| 11 | mode_G_highpass_local_sweep_seed42 | T7 | - | 0.468098 | 0.219512 | 0.193182 | 0.394726 | csv_summary |
| 12 | mode_G_highpass_local_sweep_seed42 | T8 | - | 0.461169 | 0.317073 | 0.079545 | 0.402189 | csv_summary |
| 13 | mode_G_highpass_local_sweep_seed42 | T2 | - | 0.436085 | 0.341463 | 0.068182 | 0.367280 | csv_summary |
| 14 | mode_G_highpass_local_sweep_seed42 | T5 | - | 0.435122 | 0.512195 | 0.102273 | 0.341384 | csv_summary |
| 15 | mode_G_highpass_local_sweep_seed42 | T4 | - | 0.429584 | 0.341463 | 0.215909 | 0.333695 | csv_summary |
| 16 | factorial_A_H_highpass_s50_a0p10 | E | - | 0.408162 | 0.365854 | 0.125000 | 0.334929 | csv_summary |
| 17 | mode_G_highpass_local_sweep_seed42 | T9 | - | 0.400008 | 0.292683 | 0.181818 | 0.326610 | csv_summary |
| 18 | mode_G_fourier_tuning_seed42 | G5 | - | 0.378245 | 0.390244 | 0.125000 | 0.294902 | notebook_rendered_table |
| 19 | factorial_A_H_highpass_s50_a0p10 | B | - | 0.345105 | 0.634146 | 0.193182 | 0.210763 | csv_summary |
| 20 | training_log_summary_main_results | High-pass a0.1 s50 | fixed all-splits high-pass | 0.345105 | 0.634146 | 0.193182 | 0.210763 | markdown_main_results |

### Path 15: Fourier G4 x Strong Aug x SimAM+CA

| Run | Policy / Name | Mode | Fourier | Alpha | Sigma | Hook | Strong Aug | Attention | mAP50 | mAP50-95 | Healthy FP | FP/img | Miss | Count MAE | Healthy-aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | Fourier G4 + strong aug + hook off + baseline YOLO11n-seg | B | highpass_boost | 0.100000 | 50 | no | yes | no | 0.577126 | 0.208817 | 0.317073 | 0.365854 | 0.147727 | 0.435606 | 0.501480 |
| D | Fourier G4 + strong aug + hook off + SimAM+CA | D | highpass_boost | 0.100000 | 50 | no | yes | yes | 0.556110 | 0.198244 | 0.268293 | 0.268293 | 0.113636 | 0.354167 | 0.494527 |
| C | no Fourier + strong aug + hook off + SimAM+CA | C | none | - | - | no | yes | yes | 0.493754 | 0.171782 | 0.365854 | 0.414634 | 0.125000 | 0.314394 | 0.422699 |

Important caveat: Mode A was defined but disabled. The missing Mode A control is no Fourier, strong augmentation, hook off, baseline YOLO11n-seg architecture. Without Mode A, the notebook proves the combined Mode B recipe is strong but does not fully isolate Fourier's gain from the strong augmentation policy.

### Canonical Historical Runs

| Run | Policy / Name | Mode | Fourier | Alpha | Sigma | Hook | Clean Aug | Eval | Best Ep | Ep Run | Train Img | mAP50 | mAP50-95 | Healthy FP | Miss | Count MAE | Healthy-aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hook-on clean baseline anchor | no Fourier, clean-light YOLO aug, hook on | - | none | - | - | yes | yes | original | 56 | 86 | 905 | 0.511424 | 0.163868 | 0.243902 | 0.079545 | 0.331439 | 0.458530 |
| Clean aug hook-off control | no Fourier, clean-light YOLO aug | - | none | - | - | no | yes | original | - | 100 | 905 | 0.473408 | 0.158613 | 0.390244 | 0.090909 | 0.518939 | 0.394800 |
| High-pass a0.1 s50 | fixed all-splits high-pass | - | highpass | 0.100000 | 50 | no | no | Fourier | 10 | 50 | 905 | 0.345105 | 0.088484 | 0.634146 | 0.193182 | 0.839015 | 0.210763 |
| Train-only high-pass a0.1 | train transformed, valid/test original | - | highpass | 0.100000 | - | no | no | original | 10 | 50 | 905 | 0.336343 | 0.091158 | 0.609756 | 0.193182 | 0.873106 | 0.202734 |
| High-frequency damping a0.1 s50 | fixed all-splits damping | - | high_frequency_damping | 0.100000 | 50 | no | no | Fourier | 11 | 51 | 905 | 0.321119 | 0.106764 | 0.390244 | 0.522727 | 0.592803 | 0.174045 |
| High-pass a0.05 s50 | fixed all-splits high-pass | - | highpass | 0.050000 | 50 | no | no | Fourier | 20 | 60 | 905 | 0.308774 | 0.079226 | 0.341463 | 0.250000 | 0.562500 | 0.209003 |
| Band-pass l12 h60 a0.2 | fixed all-splits band-pass | - | bandpass | 0.200000 | - | no | no | Fourier | 48 | 88 | 905 | 0.295219 | 0.083854 | 0.268293 | 0.431818 | 0.630682 | 0.172083 |
| High-pass a0.20 s50 | fixed all-splits high-pass | - | highpass | 0.200000 | 50 | no | no | Fourier | 24 | 65 | 905 | 0.291338 | 0.082667 | 0.292683 | 0.329545 | 0.539773 | 0.185649 |
| Band-pass l20 h80 a0.2 | fixed all-splits band-pass | - | bandpass | 0.200000 | - | no | no | Fourier | 52 | 92 | 905 | 0.286980 | 0.088123 | 0.195122 | 0.522727 | 0.621212 | 0.157998 |
| Random high-pass train aug a0.05-0.20 | original + random Fourier train copies | - | highpass | 0.050000 | - | no | no | original | 14 | 54 | 1810 | 0.283641 | 0.081329 | 0.146341 | 0.602273 | 0.685606 | 0.144385 |
| Low-frequency flatten b0.3 s100 | fixed all-splits lowfreq flatten | - | low_frequency_flatten | - | 100 | no | no | Fourier | 14 | 54 | 905 | 0.278693 | 0.085208 | 0.146341 | 0.465909 | 0.551136 | 0.166616 |
| Homomorphic gl0.70 gh1.20 s50 | fixed all-splits homomorphic | - | homomorphic | - | 50 | no | no | Fourier | 22 | 62 | 905 | 0.261197 | 0.091581 | 0.414634 | 0.238636 | 0.666667 | 0.150605 |
| High-pass a0.15 s50 | fixed all-splits high-pass | - | highpass | 0.150000 | 50 | no | no | Fourier | 16 | 71 | 905 | 0.248861 | 0.072006 | 0.146341 | 0.511364 | 0.630682 | 0.125988 |
| Mixed original + high-pass a0.1 train aug | original + fixed Fourier train copies | - | highpass | 0.100000 | - | no | no | original | 28 | 68 | 1810 | 0.242476 | 0.073790 | 0.146341 | 0.454545 | 0.541667 | 0.132577 |
| High-pass a0.3 s50 | fixed all-splits high-pass | - | highpass | 0.300000 | 50 | no | no | Fourier | 45 | 85 | 905 | 0.242205 | 0.080400 | 0.146341 | 0.522727 | 0.647727 | 0.116776 |
| Strict no-aug hook-off baseline | no Fourier, no YOLO aug, hook off | - | none | - | - | no | no | original | 20 | 60 | 905 | 0.235374 | 0.060555 | 0.390244 | 0.227273 | 0.429924 | 0.140762 |

### Factorial A-H: Fourier High-pass s50 a0.10

| Run | Policy / Name | Mode | Fourier | Alpha | Sigma | Hook | Clean Aug | Eval | Best Ep | Ep Run | Train Img | mAP50 | mAP50-95 | Healthy FP | Miss | Count MAE | Healthy-aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G | - | G | highpass | 0.100000 | 50 | yes | yes | - | 28 | 100 | 905 | 0.531771 | 0.170904 | 0.243902 | 0.102273 | 0.437500 | 0.470165 |
| D | - | D | none | - | - | yes | yes | - | 56 | 96 | 905 | 0.512002 | 0.167785 | 0.243902 | 0.079545 | 0.321970 | 0.459581 |
| C | - | C | none | - | - | yes | yes | - | 64 | 100 | 905 | 0.473408 | 0.158613 | 0.390244 | 0.090909 | 0.518939 | 0.394800 |
| E | - | E | highpass | 0.100000 | 50 | yes | yes | - | 26 | 72 | 905 | 0.408162 | 0.124771 | 0.365854 | 0.125000 | 0.357955 | 0.334929 |
| B | - | B | highpass | 0.100000 | 50 | yes | yes | - | 35 | 50 | 905 | 0.345105 | 0.088484 | 0.634146 | 0.193182 | 0.839015 | 0.210763 |
| H | - | H | none | - | - | yes | yes | - | 12 | 52 | 905 | 0.306111 | 0.077802 | 0.365854 | 0.272727 | 0.626894 | 0.197272 |
| F | - | F | highpass | 0.100000 | 50 | yes | yes | - | 16 | 49 | 905 | 0.299415 | 0.082874 | 0.414634 | 0.306818 | 0.732955 | 0.175281 |
| A | - | A | none | - | - | yes | yes | - | 13 | 60 | 905 | 0.235374 | 0.060555 | 0.390244 | 0.227273 | 0.429924 | 0.140762 |

### Mode G High-pass Local Sweep

| Run | Policy / Name | Mode | Fourier | Alpha | Sigma | Hook | Clean Aug | Eval | Best Ep | Ep Run | Train Img | mAP50 | mAP50-95 | Healthy FP | Miss | Count MAE | Healthy-aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | - | T1 | highpass | 0.080000 | 35 | yes | yes | - | 70 | 100 | 905 | 0.486738 | 0.169787 | 0.414634 | 0.056818 | 0.371212 | 0.418192 |
| T6 | - | T6 | highpass | 0.120000 | 40 | yes | yes | - | 17 | 100 | 905 | 0.484870 | 0.162447 | 0.341463 | 0.125000 | 0.397727 | 0.412087 |
| T3 | - | T3 | highpass | 0.120000 | 35 | yes | yes | - | 99 | 100 | 905 | 0.476721 | 0.165574 | 0.414634 | 0.102273 | 0.342803 | 0.402776 |
| T7 | - | T7 | highpass | 0.100000 | 45 | yes | yes | - | 26 | 61 | 905 | 0.468098 | 0.142472 | 0.219512 | 0.193182 | 0.448864 | 0.394726 |
| T8 | - | T8 | highpass | 0.120000 | 45 | yes | yes | - | 69 | 100 | 905 | 0.461169 | 0.158254 | 0.317073 | 0.079545 | 0.306818 | 0.402189 |
| T2 | - | T2 | highpass | 0.100000 | 35 | yes | yes | - | 46 | 85 | 905 | 0.436085 | 0.134966 | 0.341463 | 0.068182 | 0.488636 | 0.367280 |
| T5 | - | T5 | highpass | 0.100000 | 40 | yes | yes | - | 34 | 88 | 905 | 0.435122 | 0.143280 | 0.512195 | 0.102273 | 0.543561 | 0.341384 |
| T4 | - | T4 | highpass | 0.080000 | 40 | yes | yes | - | 70 | 76 | 905 | 0.429584 | 0.145919 | 0.341463 | 0.215909 | 0.587121 | 0.333695 |
| T9 | - | T9 | highpass | 0.100000 | 55 | yes | yes | - | 46 | 100 | 905 | 0.400008 | 0.136314 | 0.292683 | 0.181818 | 0.337121 | 0.326610 |

### Mode G Fourier Tuning

| Run | Policy / Name | Mode | Fourier | Alpha | Sigma | Hook | Clean Aug | Eval | Best Ep | Ep Run | Train Img | mAP50 | mAP50-95 | Healthy FP | Miss | Count MAE | Healthy-aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G4 | - | G4 | - | - | - | - | - | - | - | - | - | 0.531771 | 0.170904 | 0.243902 | 0.102273 | 0.437500 | 0.470165 |
| G8 | - | G8 | - | - | - | - | - | - | - | - | - | 0.471856 | 0.174861 | 0.463415 | 0.034091 | 0.346591 | 0.403071 |
| G5 | - | G5 | - | - | - | - | - | - | - | - | - | 0.378245 | 0.134318 | 0.390244 | 0.125000 | 0.511364 | 0.294902 |

## Methods Already Tested or Considered

Already tested:

- Strict no-augmentation hook-off baseline.
- Clean-light augmentation with hook off.
- Hook-on clean baseline anchor.
- Global Fourier high-pass.
- Global Fourier band-pass.
- High-frequency damping.
- Low-frequency flattening.
- Homomorphic filtering.
- Random Fourier train augmentation.
- Mixed original + Fourier train copies.
- Train-only high-pass with original validation/test.
- Factorial Fourier/augmentation/hook A-H.
- Local high-pass alpha/sigma tuning in Mode G.
- Some Mode G Fourier tuning rows.
- Rembg/U2Net background removal. It did not help and should not be prioritized.

Promising:

- Fourier high-pass s50 a0.10 in Mode G.
- Heavy augmentation interaction, based on latest user-reported `0.577`.

Weak or risky:

- Global Fourier-only preprocessing without augmentation.
- Strong deterministic Fourier that suppresses healthy false positives by missing diseased regions.
- Background removal with rembg/U2Net.
- Methods that break mask geometry or produce unrealistic shrimp/disease layouts.

## What We Need From Deep Research

Please return a ranked plan. For each proposed method, include:

| Field | Required detail |
|---|---|
| Method name | Concise name |
| Category | augmentation, preprocessing, frequency-domain, architecture, loss/objective, inference, calibration, hard-negative mining, etc. |
| Core idea | What changes and why |
| Why it may help this dataset | Tie to shrimp disease texture, small data, grouped-specimen generalization, false positives, disease miss rate |
| Literature support | Give papers/docs/URLs and summarize evidence without overclaiming |
| Implementation plan | Kaggle-notebook-ready steps |
| Suggested config | Concrete parameter ranges |
| Expected benefit | Estimated direction and magnitude if possible |
| Risks | How it might fail or inflate metrics |
| Ablation design | Minimal set of runs to validate |
| Priority | High / medium / low |

## Preferred Experiment Style

- Keep runs auditable and notebook-per-regime when possible.
- Use boolean mode switches if a notebook contains multiple experiments.
- Save full summary CSV, paper-row CSV, best checkpoint, train args, split manifest, and data.yaml.
- Always preserve grouped-specimen split.
- Compare against the hook-on clean baseline and the best Mode G/Fourier-heavy run.
- For promising methods, recommend multi-seed confirmation after seed 42.

## Ideas Worth Investigating

These are not instructions to blindly accept; they are directions to investigate:

- Heavy YOLO augmentation policies that are still mask-safe.
- Mosaic with close-mosaic schedule.
- Copy-paste/cutmix/mixup variants for instance segmentation.
- Frequency-domain augmentation combined with heavy augmentation.
- Fourier amplitude/phase mixing between images.
- FDA-style low-frequency domain adaptation.
- Frequency dropout, band-stop, high-frequency damping, random band perturbation.
- Lesion/body-localized frequency transforms rather than whole-image transforms.
- CLAHE or color constancy only if biologically plausible and mask-safe.
- Test-time augmentation and confidence/IoU threshold calibration.
- Hard-negative mining using healthy false positives.
- YOLO-style lightweight attention modules, excluding U-Net-style models.
- Training hyperparameters such as image size, optimizer, learning rate schedule, close-mosaic, freeze/unfreeze, EMA, batch size, and patience.

## What Not To Do

- Do not suggest U-Net-like model replacement.
- Do not suggest methods that require changing to semantic segmentation only.
- Do not suggest specimen-leaking split schemes.
- Do not optimize on the test set as if it were validation.
- Do not recommend background removal as a main next step unless there is a very strong reason; it has already underperformed.
- Do not treat healthy-aware score as the only objective for this phase; mAP50 improvement is the primary goal.
