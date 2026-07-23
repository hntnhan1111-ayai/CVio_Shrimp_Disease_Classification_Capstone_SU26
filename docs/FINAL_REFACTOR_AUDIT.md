# Final refactor audit

**Timestamp:** 2026-07-23T20:00:00+07:00
**Workspace:** D:\CVio_RECSRA_Shrimp_Disease_Detection
**Branch:** paper/yolo11s-recsra-q1-academic-release-20260723

## 1. Baseline correction

- All public-facing clean-test comparisons now use the **original YOLO11s benchmark baseline** (12.900% mAP50, 3.800% mAP50-95).
- The corruption-control baseline (14.418% mAP50, 4.147% mAP50-95) is preserved only in explicitly labeled corruption-control files.
- CSV and Markdown tables: `results/tables/clean_original_yolo11s_vs_recsra_percent.csv`, `clean_original_yolo11s_vs_recsra.md`, `clean_baseline_vs_recsra.md`
- HTML: `docs/results_dashboard.html`
- LaTeX: `paper/tables/clean_results.tex`
- Raw evidence preserved: `results/raw/clean_original_baseline_vs_recsra.json`, `corruption_control_clean_metrics.json`

## 2. Canonical source files created

- `results/raw/clean_original_baseline_vs_recsra.json`
- `results/raw/corruption_control_clean_metrics.json`
- `results/tables/clean_original_yolo11s_vs_recsra_percent.csv`
- `results/tables/clean_original_yolo11s_vs_recsra.md`
- `results/tables/corruption_control_clean_percent.csv`
- `results/tables/corruption_control_clean.md`

## 3. Files corrected

- `README.md` — baseline table updated
- `docs/results_dashboard.html` — KPI gains and table updated
- `docs/METRIC_REPORTING.md` — calculation examples updated
- `docs/README_VI.md` — Vietnamese quick guide updated
- `results/reports/PAPER_RESULTS_DRAFT.md` — narrative updated
- `paper/tables/clean_results.tex` — LaTeX table updated
- `scripts/generate_percent_tables.py` — added canonical-source note
- `scripts/verify_repository.py` — added baseline value assertions and forbidden-reference checks
- `results/raw/metrics_json/CLEAN/baseline.json` — updated with original baseline context
- `results/raw/metrics_json/CLEAN/baseline.log` — last-line JSON updated

### Raw files preserved as corruption-control evidence

- `results/raw/clean_metrics.csv` — NOT changed (contains corruption-control baseline raw values)
- `results/raw/all_clean_and_50_condition_metrics.csv` — NOT changed (102 rows of corruption-control evidence)

## 4. Documentation created

- `docs/METRIC_PROVENANCE.md` — maps every metric to its source, explains two baseline contexts
- `docs/DATASET_EDA.md` — interprets EDA findings
- `docs/ENVIRONMENT.md` — actual RTX 4090 and target T4×2 environments
- `dataset_eda/SOURCE_PROVENANCE.md` — EDA archive chain-of-custody
- `docs/PRE_REFACTOR_AUDIT.md` — pre-refactor snapshot
- `tests/test_repository.py` — pytest verification suite

## 5. Legacy archive handling

- `cvio_yolo11s_paper_ready_outputs_no_weights/` (extracted legacy archive) — contains V007/SimAM material. Not committed to the new branch. Legacy ZIP remains in the working tree but is gitignored.
- `dataset_eda.zip` and `cvio_yolo_dataset_eda_no_raw_images.zip` — verified as EDA archive duplicates. Canonical `dataset_eda/` is used.

## 6. Git status

- Remote: `https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26.git`
- New branch: `paper/yolo11s-recsra-q1-academic-release-20260723`
- Commit message: `feat(paper): publish audited YOLO11s-RECSRA academic release`
- Git LFS configured for `*.pt`, `*.pth`, `*.onnx`, `*.engine`

## 7. Remaining sensitive items

- Absolute paths `/home/drnguyenvinh/` remain in raw experiment files (clean_metrics.csv, metrics_json, args.original.yaml, etc.) — these are raw experimental evidence and should not be altered.
- Legacy archive ZIPs (cvio_yolo11s_paper_ready_outputs_no_weights.zip, shrimp_od_200ep_visual_report.zip) are present in the working tree but will not be committed.
