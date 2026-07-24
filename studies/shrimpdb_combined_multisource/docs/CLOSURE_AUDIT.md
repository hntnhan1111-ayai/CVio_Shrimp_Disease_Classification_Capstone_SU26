# Closure Audit

## Current merged-package closure

The merged result integration is represented by:

- `artifacts/metadata/merged_result_registry.json`
- `model_registry/selected_best_by_dataset.json`
- `artifacts/metadata/figure_registry.json`
- `artifacts/metadata/report_registry.json`
- `artifacts/metadata/closure_audit.json`
- `artifacts/metadata/method_identity_audit.{json,md}`
- `artifacts/metadata/label_correction_manifest.json`

The source ZIP was verified with SHA-256
`dc3e0c544e386008f62df6dbd12d806f9d00d854ff1f833b70365db7e7030e59`. Selected raw metric,
prediction, confusion-matrix, provenance, and checkpoint source files were not modified.
The repository contains no selected `.pt` files, raw dataset, ZIP, cache, or virtual
environment. Checkpoint distribution is external and hash-verified.

## Scientific interpretation

The common display label is `ASL-LDAM + SimAM-DCFR`. ShrimpDB-3's selected source method is
ASL-LDAM + SimAM-DCFR; Combined-4's selected source method is CE Baseline. The selected
values are 91.49% / 91.29% and 87.73% / 86.51% for accuracy / Macro-F1 respectively.
This is a fixed seed-42, validation-selected result package and does not establish
statistical significance, universal superiority, clinical validity, or production readiness.

## Reproduction checks

```powershell
uv sync --frozen
uv run python tools/generate_merged_best_artifacts.py
uv run pytest -q
git diff --check
```
