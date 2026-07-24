# Merged Best-by-Dataset Results

The reviewed ZIP has SHA-256
`dc3e0c544e386008f62df6dbd12d806f9d00d854ff1f833b70365db7e7030e59`.
It selects the highest Macro-F1 result within each dataset and uses accuracy as the
tie-breaker. Its display label and verified source method are separate fields.

| Dataset | Display label | Actual source method | Label/source match | Accuracy | Macro-F1 | Checkpoint SHA-256 |
|---|---|---|---|---:|---:|---|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | ASL-LDAM + SimAM-DCFR | true | 91.49% | 91.29% | `ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36` |
| Combined-4 | ASL-LDAM + SimAM-DCFR | CE Baseline | false | 87.73% | 86.51% | `adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad` |

The Combined-4 selected metrics originate from the CE baseline run; the ShrimpDB-3
selected metrics originate from the ASL-LDAM + SimAM-DCFR run. No raw metric, prediction,
confusion matrix, or checkpoint was edited. Only repository-facing metadata and generated
presentation artifacts were added or corrected.

Machine-readable evidence is in `artifacts/merged_best_by_dataset/`, with the canonical
registry at `artifacts/metadata/merged_result_registry.json` and checkpoint metadata at
`model_registry/selected_best_by_dataset.json`.
