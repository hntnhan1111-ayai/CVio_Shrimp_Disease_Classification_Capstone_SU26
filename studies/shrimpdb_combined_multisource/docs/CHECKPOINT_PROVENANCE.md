# Checkpoint Provenance

Selected `.pt` files are intentionally excluded from Git. The external source paths,
filenames, SHA-256 values, class counts, and class order are recorded in
`model_registry/selected_best_by_dataset.json` and
`artifacts/tables/checkpoint_provenance.csv`.

| Dataset | Actual source method | Selected filename | Classes | SHA-256 |
|---|---|---|---:|---|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | `selected_best.pt` | 3 | `ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36` |
| Combined-4 | CE Baseline | `selected_best.pt` | 4 | `adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad` |

Before application use, retrieve the checkpoint from the merged package, hash it, verify
the registry entry, and bind the model to the registry class order. A filename or common
display label is not sufficient evidence of method identity.
