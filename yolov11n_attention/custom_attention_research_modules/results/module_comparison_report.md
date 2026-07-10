# Module Comparison Report

Generated from completed notebooks in this workspace.

## Main Test Metrics

| Model | Aug | Full mAP50 | Diseased mAP50 | Full mAP50-95 | Diseased mAP50-95 | Healthy FP | FP masks/img | Disease miss | Healthy-aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline clean | clean_light | 0.441 | 0.466 | 0.161 | 0.169 | 36.6% | 0.415 | 12.5% | 0.393 |
| Baseline strong | strong | 0.481 | 0.506 | 0.171 | 0.179 | 36.6% | 0.390 | - | - |
| SimAM_CA strong | strong | 0.520 | 0.551 | 0.187 | 0.196 | 34.1% | 0.390 | - | - |
| CoTE | clean_light | 0.372 | 0.416 | 0.099 | 0.112 | 43.9% | 0.488 | 10.2% | 0.329 |
| LPSC | clean_light | 0.355 | 0.401 | 0.102 | 0.114 | 31.7% | 0.488 | 11.4% | 0.320 |
| SCSG | clean_light | 0.269 | 0.301 | 0.076 | 0.084 | 41.5% | 0.488 | 12.5% | 0.219 |
| CoTE strong | strong | 0.433 | 0.487 | 0.123 | 0.136 | 31.7% | 0.415 | 9.1% | 0.421 |
| LPSC strong | strong | 0.427 | 0.463 | 0.133 | 0.143 | 26.8% | 0.268 | 21.6% | 0.384 |
| CoLPSC | clean_light | 0.322 | 0.374 | 0.079 | 0.090 | 46.3% | 0.488 | 13.6% | 0.283 |
| CoLPSC strong | strong | 0.367 | 0.409 | 0.107 | 0.119 | 29.3% | 0.366 | 19.3% | 0.327 |

## Validation And Stability Diagnostics

| Model | Val mAP50 | Val mAP50-95 | Labeled val mAP50 | Val FP | Val healthy-aware | Epochs | Best epoch | Seg loss gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline clean | 0.502 | 0.173 | 0.492 | 17.1% | 0.427 | 60.00 | 52.00 | 2.154 |
| Baseline strong | 0.551 | 0.190 | - | - | - | 89.00 | 61.00 | 1.133 |
| SimAM_CA strong | 0.548 | 0.193 | - | - | - | 100.00 | 67.00 | 0.606 |
| CoTE | 0.425 | 0.137 | 0.433 | 29.3% | 0.350 | 98.00 | 74.00 | 1.355 |
| LPSC | 0.382 | 0.150 | 0.407 | 24.4% | 0.336 | 100.00 | 80.00 | 1.375 |
| SCSG | 0.444 | 0.129 | 0.445 | 19.5% | 0.369 | 83.00 | 69.00 | 0.919 |
| CoTE strong | 0.452 | 0.160 | 0.469 | 14.6% | 0.424 | 100.00 | 83.00 | 0.020 |
| LPSC strong | 0.397 | 0.127 | 0.405 | 12.2% | 0.353 | 100.00 | 83.00 | -0.026 |
| CoLPSC | 0.381 | 0.134 | 0.397 | 24.4% | 0.310 | 85.00 | 55.00 | 1.021 |
| CoLPSC strong | 0.452 | 0.156 | 0.462 | 17.1% | 0.400 | 100.00 | 90.00 | 0.124 |

## Quick Ranking

- Best overall full test mask mAP50: SimAM_CA strong (0.520)
- Best diseased-only mask mAP50: SimAM_CA strong (0.551)
- Lowest healthy FP-rate: LPSC strong (0.268)
- Lowest FP masks per healthy image: LPSC strong (0.268)
- Best healthy-aware test score among rows with this metric: CoTE strong (0.421)

## Deltas Versus Baseline Clean

- SimAM_CA strong: full mAP50 0.079, diseased mAP50 0.084, healthy FP -2.4% vs Baseline clean.
- CoTE strong: full mAP50 -0.008, diseased mAP50 0.020, healthy FP -4.9% vs Baseline clean.
- LPSC strong: full mAP50 -0.014, diseased mAP50 -0.003, healthy FP -9.8% vs Baseline clean.

## Interpretation

- SimAM_CA strong remains the best overall model by full test and diseased-only mask mAP50.
- CoTE strong is the best new research module for a balanced objective: it improves diseased-only mAP50 and healthy-aware score over the clean baseline while reducing healthy FP-rate.
- LPSC strong is the best new research module for low false positives: it has the lowest healthy FP-rate and FP masks per image among the custom attention experiments, but it raises disease miss rate.
- SCSG and CoLPSC are not recommended as primary candidates in the current form because their test mAP and healthy-aware scores trail the better alternatives.
- The gap between clean/light and strong augmentation shows augmentation is a major factor; architecture conclusions should be made against models trained under the same augmentation policy.
