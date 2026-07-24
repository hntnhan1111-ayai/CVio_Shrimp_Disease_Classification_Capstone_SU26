# Results

All selected values below are generated from `artifacts/metadata/merged_result_registry.json`
and the imported `metrics_raw.json` files. Raw metrics, predictions, and confusion matrices
are preserved byte-for-byte from the merged package.

The refreshed standalone presentation report is
`artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html`.
Historical training-trajectory metadata remains in
`artifacts/final_application_model/effective_epochs.json`.

## Selected results

| Dataset | Display label | Actual source method | Accuracy | Balanced accuracy | Macro-F1 | ECE |
|---|---|---|---:|---:|---:|---:|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | ASL-LDAM + SimAM-DCFR | 91.49% | 91.55% | 91.29% | 19.30% |
| Combined-4 | ASL-LDAM + SimAM-DCFR | CE Baseline | 87.73% | 87.43% | 86.51% | 6.19% |

The common display label must not be read as the training method for Combined-4. Its
verified source is the CE baseline checkpoint
`adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad`.

## Full method comparison

See [`artifacts/tables/all_methods_comparison_percent.csv`](../artifacts/tables/all_methods_comparison_percent.csv).
The source-method column identifies the actual method for every comparison row. Selected
best-by-dataset values are 91.49% / 91.29% for ShrimpDB-3 and 87.73% / 86.51% for Combined-4
(accuracy / Macro-F1).

## Per-class, source-domain, and confusion-matrix evidence

The generated tables are:

- [`per_class_selected_shrimpdb3_percent.csv`](../artifacts/tables/per_class_selected_shrimpdb3_percent.csv)
- [`per_class_selected_combined4_percent.csv`](../artifacts/tables/per_class_selected_combined4_percent.csv)
- [`source_domain_selected_combined4_percent.csv`](../artifacts/tables/source_domain_selected_combined4_percent.csv)

Count and row-normalized confusion matrices are in
`artifacts/merged_best_by_dataset/evaluation/{shrimpdb3,combined4}` and are visualized by
figures 06–09 in the merged figure registry.

## Limitations

This is a single fixed seed-42, validation-selected benchmark. It does not establish
statistical significance, universal superiority, clinical validity, or production readiness.
The image-level split is not claimed to be specimen-safe, and cross-domain calibration and
label shift remain open evaluation concerns.
