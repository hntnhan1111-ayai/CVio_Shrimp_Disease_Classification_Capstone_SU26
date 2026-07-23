# ONLY_fourier Running Metrics Report

This file collects the running metric tables from the current `ONLY_fourier` training logs into one readable report. The existing `training_log_summary.md` is treated as the canonical table for the older baseline and Fourier-only runs; newer sweep CSVs/notebook tables are appended as separate experiment families.

Common protocol unless noted: stratified grouped-specimen split, seed 42, `yolo11n-seg.pt`, 100 requested epochs, patience 40.

## Mode Key

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

## Overall Leaderboard by Labeled Test mAP50

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

## Canonical Historical Runs from training_log_summary.md

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

## Factorial A-H: Fourier High-pass s50 a0.10

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

## Mode G High-pass Local Sweep

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

## Mode G Fourier Tuning

| Run | Policy / Name | Mode | Fourier | Alpha | Sigma | Hook | Clean Aug | Eval | Best Ep | Ep Run | Train Img | mAP50 | mAP50-95 | Healthy FP | Miss | Count MAE | Healthy-aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G4 | - | G4 | - | - | - | - | - | - | - | - | - | 0.531771 | 0.170904 | 0.243902 | 0.102273 | 0.437500 | 0.470165 |
| G8 | - | G8 | - | - | - | - | - | - | - | - | - | 0.471856 | 0.174861 | 0.463415 | 0.034091 | 0.346591 | 0.403071 |
| G5 | - | G5 | - | - | - | - | - | - | - | - | - | 0.378245 | 0.134318 | 0.390244 | 0.125000 | 0.511364 | 0.294902 |

## Path 15: Fourier G4 x Strong Aug x SimAM+CA

Source analysis: `15_fourier_G4_strong_aug_attention_log_analysis.md`

| Run | Policy / Name | Mode | Fourier | Alpha | Sigma | Hook | Strong Aug | Attention | mAP50 | mAP50-95 | Healthy FP | FP/img | Miss | Count MAE | Healthy-aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | Fourier G4 + strong aug + hook off + baseline YOLO11n-seg | B | highpass_boost | 0.100000 | 50 | no | yes | no | 0.577126 | 0.208817 | 0.317073 | 0.365854 | 0.147727 | 0.435606 | 0.501480 |
| D | Fourier G4 + strong aug + hook off + SimAM+CA | D | highpass_boost | 0.100000 | 50 | no | yes | yes | 0.556110 | 0.198244 | 0.268293 | 0.268293 | 0.113636 | 0.354167 | 0.494527 |
| C | no Fourier + strong aug + hook off + SimAM+CA | C | none | - | - | no | yes | yes | 0.493754 | 0.171782 | 0.365854 | 0.414634 | 0.125000 | 0.314394 | 0.422699 |

## Source Inventory

| Source Group | Source Kind | Rows | Source File |
| --- | --- | --- | --- |
| canonical_markdown_summary | markdown_main_results | 16 | shrimp-leakage-aware-segmentation\ONLY_fourier\training_log\training_log_summary.md |
| path_15_fourier_G4_strong_aug_attention | notebook_training_log | 3 | C:\Users\Admin\Downloads\notebookae2e3c4caa.ipynb |
| factorial_A_H_highpass_s50_a0p10 | csv_summary | 8 | shrimp-leakage-aware-segmentation\ONLY_fourier\training_log\only_fourier_factorial_A_H_highpass_s50_a0p10_summary.csv |
| mode_G_fourier_tuning | notebook_rendered_table | 3 | shrimp-leakage-aware-segmentation\ONLY_fourier\training_log\11to20\013_mode_G_fourier_tuning\013-mode-g-fourier-tuning-seed42.ipynb |
| mode_G_highpass_local_sweep_seed42 | csv_summary | 9 | shrimp-leakage-aware-segmentation\ONLY_fourier\highpass_local_sweep\training_log\only_fourier_mode_G_highpass_local_sweep_seed42_summary.csv |

## Quick Read

- Best current mAP50 row is Path 15 Mode B at `0.577126`, which clears the previous `0.54` target and reaches the desired `0.56-0.57` range.
- Path 15 Mode D also clears `0.54` at `0.556110`, with better healthy FP, miss rate, and count MAE than Mode B.
- The hook-on clean baseline anchor remains strong at about `0.511424`, and Factorial D is essentially the same tier at about `0.512002`.
- The older global Fourier-only runs beat the strict no-augmentation baseline but do not approach the clean augmentation baselines.
- The missing control is Path 15 Mode A: no Fourier, strong aug, hook off, baseline architecture. Without it, Mode B proves the combined recipe is strong, but does not fully isolate Fourier from the strong augmentation policy.
