# Results

`artifacts/tables/legacy_top5_single_run.csv` is a transcription of the five
selected results in `BAO_CAO_TONG_HOP_MODULE_DO_CHINH_XAC_CAO.md`. It is not a
new experiment and every row is a single-seed historical result.

The historical clean baseline reported for context was full mask mAP50 0.441,
disease mask mAP50 0.466, disease mAP50-95 about 0.169, healthy FP 36.6%, and
HA 0.393. It must not be compared directly with a strong-augmentation result
unless a matched strong baseline is used.

`artifacts/tables/mrtu_top5_status.csv` records notebook availability and
whether a compact result package has been imported for the newer dataset.

## Imported expanded-dataset training results

The ZIPs added to the repository-level outputs/ directory supplied 11 completed
training histories on the expanded dataset. Compact histories, arguments and
source checksums are imported under artifacts/raw_runs/ and
artifacts/metadata/mrtu_output_import.json. The selected strong rows are:

| Candidate | Completed epoch | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|
| SimAM + CA strong | 100 | 0.58618 | 0.24261 |
| DPCA P4 strong | 200 | 0.52198 | 0.21097 |
| DPCA P3/P4/P5 strong | 200 | 0.54966 | 0.20841 |
| CoTE + BoundaryLite P4 strong | 200 | 0.54322 | 0.20909 |

These are final Ultralytics validation values from results.csv, seed 42.
They do not yet establish a cross-model result: the current expanded-dataset
output does not contain a matched baseline strong or LKA→SimAM run, and no
separately documented held-out test/healthy-FP evaluation was included. The
machine-readable files are artifacts/tables/mrtu_training_final_metrics.csv
and artifacts/tables/mrtu_selected_strong_runs.csv.
