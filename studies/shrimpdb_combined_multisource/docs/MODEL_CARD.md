# Model Card and Checkpoint Distribution

This study contains YOLO26m-cls classification checkpoints selected from a merged
best-by-dataset package. Checkpoints are not tracked in Git. The authoritative registry is
[`model_registry/selected_best_by_dataset.json`](../model_registry/selected_best_by_dataset.json).

| Dataset | Display label | Actual source method | Classes | SHA-256 |
|---|---|---|---:|---|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | ASL-LDAM + SimAM-DCFR | 3 | `ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36` |
| Combined-4 | ASL-LDAM + SimAM-DCFR | CE Baseline | 4 | `adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad` |

Input is RGB 224 × 224. Class order is `Healthy, BG, WSSV` for ShrimpDB-3 and
`Healthy, BG, WSSV, WSSV_BG` for Combined-4. Retrieve a checkpoint from the reviewed
merged package, compute SHA-256, and compare it with the registry before use.

The common display label is presentation metadata only. The Combined-4 selected checkpoint
is a CE baseline model with no custom attention; it must not be integrated as an ASL-LDAM or
SimAM-DCFR model. Application code should load the dataset-specific registry entry and use
its class order, source method, and hash rather than selecting by filename.

## Intended use and limitations

Research evaluation and reproducibility only. These results are from one fixed seed and are
not a veterinary diagnostic claim, clinical validation, production-readiness claim, or proof
of universal superiority. Dataset shift, specimen correlation, label quality, calibration,
and external validity require additional studies.
