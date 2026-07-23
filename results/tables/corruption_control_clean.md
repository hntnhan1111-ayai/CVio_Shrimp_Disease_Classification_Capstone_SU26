# Corruption-control clean test comparison

| Metric | Paired control baseline | RECSRA | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| mAP50 | 14.418% | 16.036% | +1.618 pp | +11.22% |
| mAP50_95 | 4.147% | 4.657% | +0.510 pp | +12.30% |
| mAP75 | 1.097% | 1.118% | +0.020 pp | +1.85% |
| precision | 20.428% | 24.342% | +3.914 pp | +19.16% |
| recall | 24.980% | 24.695% | -0.284 pp | -1.14% |
| AP_BG_mAP50_95 | 4.477% | 5.373% | +0.897 pp | +20.03% |
| AP_WSSV_mAP50_95 | 3.816% | 3.940% | +0.123 pp | +3.23% |

> **Baseline context:** paired_corruption_control_checkpoint.
> **Usage:** clean sanity check inside the robustness experiment; paired corruption evaluation; ten-corruption tables; top-five corruption rankings; per-severity robustness figures.
> These values are for the corruption benchmark paired control ONLY.
> Do NOT use these as the main clean-test headline baseline.
> The authoritative main baseline is in `clean_original_baseline_vs_recsra.json`.
