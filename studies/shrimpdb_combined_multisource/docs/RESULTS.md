# Results

The paper-facing results are generated from `artifacts/results/FINAL_BEST_RESULTS.json` and
the selected raw metrics in `artifacts/results/{shrimpdb3,combined4}/metrics_raw.json`.

| Dataset | Display label | Actual source method | Accuracy | Balanced accuracy | Macro-F1 | ECE |
|---|---|---|---:|---:|---:|---:|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | ASL-LDAM + SimAM-DCFR | 91.49% | 91.55% | 91.29% | 19.30% |
| Combined-4 | ASL-LDAM + SimAM-DCFR | CE Baseline | 87.73% | 87.43% | 86.51% | 6.19% |

The common display label and actual source method are intentionally separate. Combined-4's
selected metrics originate from the CE Baseline run. Full method comparisons are in
[`method_comparison_percent.csv`](../artifacts/tables/method_comparison_percent.csv), and
per-class results are in [`per_class_results_percent.csv`](../artifacts/tables/per_class_results_percent.csv).

## Figures

All paper figures are in [`artifacts/figures/`](../artifacts/figures/), including training
curves, metric comparisons, count and normalized confusion matrices, and source-domain
analysis. The standalone report is
[`CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html`](../artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html).

## Limitations

These are one-run, validation-selected, fixed seed-42 results. They do not establish
statistical significance, universal superiority, clinical validity, or production readiness.
The image-level split is not claimed to be specimen-safe.
