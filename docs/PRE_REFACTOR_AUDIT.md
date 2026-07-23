# Pre-refactor audit snapshot

**Timestamp:** 2026-07-23T19:29:25+07:00
**Workspace:** D:\CVio_RECSRA_Shrimp_Disease_Detection
**Backup ZIP:** ..\CVio_RECSRA_PRE_REFACTOR_BACKUP_20260723_192925.zip
**Backup SHA-256:** 924C75A2CEA61816C90A2F0C2A2C00FB4B9FEDD9B666997834767C0E549C23F1

## 1. Tree summary

Top-level entries: 28 (including 2 ZIP archives, 1 extracted archive folder, 13 directories, 12 files).

Key directories:
- `.github/workflows/` — CI workflows
- `archive/original_experiment_source/` — original experiment scripts (training, robustness)
- `checkpoints/` — 3 `.pt` files + SHA256SUMS.txt + README.md
- `configs/` — data, eval, models, robustness, train
- `cvio_yolo11s_paper_ready_outputs_no_weights/` — legacy archive (20 MB, V007/SimAM material)
- `dataset_eda/` — EDA package (12 figures, 11 CSV tables, summary.json, manifest.json, eda_report.html)
- `docs/` — 12 documentation files
- `environment/` — 8 environment/runtime files
- `notebooks/` — notebooks
- `paper/tables/` — 1 LaTeX table (clean_results.tex)
- `reproducibility/` — empty
- `results/` — raw metrics, tables, figures, reports, qualitative
- `scripts/` — 11 script files
- `src/recsra/` — 7 source files

## 2. Duplicate archives

| File | Size | Notes |
|---|---|---|
| `dataset_eda.zip` | 1,163,379 B | EDA package ZIP |
| `cvio_yolo_dataset_eda_no_raw_images.zip` | 1,221,540 B | Likely duplicate EDA ZIP |
| `cvio_yolo11s_paper_ready_outputs_no_weights.zip` | 20,783,325 B | Legacy archive with V007/SimAM material |
| `shrimp_od_200ep_visual_report.zip` | 50,621,421 B | Large visual report ZIP |

## 3. Stale files

- `cvio_yolo11s_paper_ready_outputs_no_weights/` — extracted legacy archive at root (should be moved outside Git)
- `results/figures/original_robustness/` — old figure directory (should be cleaned)
- `results/figures/original_training/` — old figure directory (should be cleaned)
- `results/figures/paper/` — old paper figure directory (should be cleaned)
- `results/qualitative/top5_original/` — old qualitative samples
- `results/qualitative/top5_thesis_2x2/` — old qualitative samples
- `reproducibility/` — empty directory

## 4. Incorrect metric occurrences

### Wrong original baseline (14.418% / 4.147% used instead of 12.900% / 3.800%)

Files using the corruption-control baseline as the main clean-test baseline:

| File | Lines | Wrong values |
|---|---|---|
| `README.md` | 15-20 | 14.418%, 4.147%, 20.428%, 24.980%, 4.477%, 3.816% |
| `docs/results_dashboard.html` | 6-7 | 14.418%, 4.147%, 20.428%, 24.980%, 4.477%, 3.816% |
| `docs/METRIC_REPORTING.md` | 9, 13 | 14.418% in calculation examples |
| `docs/README_VI.md` | 5-6 | +1.618 pp, +0.510 pp (wrong gains) |
| `results/tables/clean_baseline_vs_recsra_percent.csv` | 2-8 | 14.418..., 4.147..., etc. |
| `results/tables/clean_baseline_vs_recsra.md` | 5-11 | 14.418%, 4.147%, etc. |
| `paper/tables/clean_results.tex` | 9-15 | 14.418, 4.147, etc. |
| `results/reports/PAPER_RESULTS_DRAFT.md` | 3 | 14.418%, 4.147% |
| `results/raw/clean_metrics.csv` | 2 | 0.14418... as baseline |
| `results/raw/all_clean_and_50_condition_metrics.csv` | 2 | 0.14418... as baseline |
| `results/raw/metrics_json/CLEAN/baseline.json` | 2 | 0.14418... |
| `results/raw/metrics_json/CLEAN/baseline.log` | 8 | 0.14418... |

### Correct original baseline (from HTML report)

The HTML report `cvio_yolo11s_paper_ready_outputs_no_weights/paper_final_outputs/paper_ready_summary_percent.html` contains:
- `YOLO11s original baseline` → `12.900` mAP50, `3.800` mAP50-95 (original benchmark row)

## 5. V007 occurrences

205 matches across the repository. All V007/SimAM references are in:
- `cvio_yolo11s_paper_ready_outputs_no_weights/` (legacy archive — 205 matches)
- No V007 references in public-facing files outside the legacy archive

## 6. Hardcoded local paths

`/home/drnguyenvinh/` appears in:
- `results/raw/metrics_json/CLEAN/baseline.json` (model_path, data_yaml)
- `results/raw/metrics_json/CLEAN/recsra.json` (model_path, data_yaml)
- `results/raw/metrics_json/CLEAN/baseline.log`
- `results/raw/clean_metrics.csv` (model_path, data_yaml)
- `results/raw/all_clean_and_50_condition_metrics.csv` (model_path, data_yaml)
- `results/raw/runtime_and_inputs.json` (dataset_root, baseline, recsra paths)
- `results/raw/training/args.original.yaml` (model, data, project, save_dir)
- `results/raw/training/winner_summary.json` (best_pt)
- `results/raw/training/attention_zoo_results.csv` (best_pt)
- `results/raw/training/results.csv` (no paths, but training log)
- `results/raw/top5_sample_metadata.json` (sample_path)
- `dataset_eda/summary.json` (data_yaml, dataset_root)
- `dataset_eda/manifest.json` (eda_out, zip_out, data_yaml, dataset_root)
- `dataset_eda/README.md` (data_yaml, dataset_root)
- `dataset_eda/tables/image_records.csv` (image_path, label_path)
- `dataset_eda/tables/label_records.csv` (label_path)
- `configs/data/data.original_provenance.yaml`
- `cvio_yolo11s_paper_ready_outputs_no_weights/` (many files)

## 7. Missing provenance

- `reproducibility/` directory is empty (no REPOSITORY_AUDIT.json, no PROJECT_FILE_SHA256.csv)
- `docs/METRIC_PROVENANCE.md` does not exist
- `docs/DATASET_EDA.md` does not exist
- `docs/ENVIRONMENT.md` does not exist
- `docs/FINAL_REFACTOR_AUDIT.md` does not exist
- `docs/GIT_RELEASE_RECORD.md` does not exist
- `dataset_eda/SOURCE_PROVENANCE.md` does not exist
- `results/raw/clean_original_baseline_vs_recsra.json` does not exist
- `results/raw/corruption_control_clean_metrics.json` does not exist
- `results/tables/clean_original_yolo11s_vs_recsra_percent.csv` does not exist
- `results/tables/clean_original_yolo11s_vs_recsra.md` does not exist
- `results/tables/corruption_control_clean_percent.csv` does not exist
- `results/tables/top5_corruptions.tex` does not exist
- `paper/tables/top5_corruptions.tex` does not exist
- `tests/` directory does not exist

## 8. Broken links

- README references `results/figures/paper/fig01_clean_test_comparison.png` and `fig05_top5_gain_by_severity.png` — these exist but use wrong baseline
- README references `docs/LIMITATIONS.md` — exists
- README references `docs/PUBLIC_RELEASE_CHECKLIST.md` — does not exist in docs/

## 9. Oversized files

| File | Size | Notes |
|---|---|---|
| `cvio_yolo11s_paper_ready_outputs_no_weights.zip` | 20.8 MB | Legacy archive |
| `shrimp_od_200ep_visual_report.zip` | 50.6 MB | Large visual report |
| `checkpoints/yolo11s_baseline_best.pt` | ? | Checkpoint (LFS) |
| `checkpoints/yolo11s_recsra_best.pt` | ? | Checkpoint (LFS) |
| `checkpoints/yolo11s_recsra_last.pt` | ? | Checkpoint (LFS) |

## 10. Compiled cache files

No `__pycache__/` or `*.pyc` files found in the current tree (they are in `.gitignore`).

## 11. Public files using decimals instead of percentages

- `results/raw/clean_metrics.csv` — uses decimals (correct for raw, but wrong baseline)
- `results/raw/all_clean_and_50_condition_metrics.csv` — uses decimals (correct for raw, but wrong baseline)
- `results/raw/metrics_json/CLEAN/*.json` — uses decimals (correct for raw, but wrong baseline)
- `results/tables/clean_baseline_vs_recsra_percent.csv` — uses decimals in CSV values (should be percentages)
- `results/tables/all10_corruptions_percent.csv` — uses decimals in CSV values (should be percentages)
- `results/tables/top5_corruptions_percent.csv` — uses decimals in CSV values (should be percentages)

## 12. Conflicting hardware claims

- `docs/HARDWARE.md` correctly states RTX 4090
- `environment/kaggle_t4x2_target.yaml` exists as a target profile
- No conflicting claims found in public docs

## 13. Missing EDA assets

EDA package is complete with all expected figures and tables. However:
- `dataset_eda/SOURCE_PROVENANCE.md` does not exist
- EDA README contains hardcoded `/home/drnguyenvinh/` paths
- EDA summary.json contains hardcoded `/home/drnguyenvinh/` paths
- EDA manifest.json contains hardcoded `/home/drnguyenvinh/` paths
- EDA image_records.csv contains hardcoded `/home/drnguyenvinh/` paths

## 14. Figure directory structure

Current:
- `results/figures/original_robustness/`
- `results/figures/original_training/`
- `results/figures/paper/`

Target:
- `results/figures/clean/`
- `results/figures/training/`
- `results/figures/robustness/`
- `results/figures/dataset_eda/`

## 15. Summary of required changes

1. **Critical:** Replace 14.418% / 4.147% with 12.900% / 3.800% as the original baseline in all public-facing files
2. **Critical:** Create canonical raw JSON source files
3. **Critical:** Move legacy archive outside Git
4. **Critical:** Remove all hardcoded `/home/drnguyenvinh/` paths from public files
5. **Critical:** Create all missing documentation files
6. **Important:** Reorganize figure directories
7. **Important:** Create tests
8. **Important:** Update scripts to use canonical source
9. **Important:** Create reproducibility manifests
10. **Important:** Set up Git and push
