# Contributing to CVio

Thank you for improving the research record. Keep repository-facing content in English and preserve scientific provenance.

## Workflow

1. Open or reference an issue for non-trivial work.
2. Create a focused branch such as `docs/<topic>`, `fix/<topic>`, `feat/<topic>`, or `experiment/<id>`.
3. Make small, imperative commits.
4. Update documentation and tests alongside behavior.
5. Run all repository checks before opening a pull request.

Do not commit datasets, private images, credentials, caches, checkpoints, exported weights, or generated environments unless maintainers have explicitly approved their license, privacy, and size.

## Experiment submissions

Use the experiment issue form and include:

- full commit SHA and config path;
- dataset version, fingerprint, split-manifest hash, and leakage checks;
- seed, environment lock, hardware, and exact command;
- checkpoint path and cryptographic hash;
- metrics file, predictions, logs, thresholds, and failure notes;
- comparison label: paper-reported, team reimplementation, proposed method, ablation, or selected final run;
- evidence that test data did not guide selection.

Measured values must never be reformatted in a way that changes units, averaging, precision meaning, or provenance.

## Dataset changes

Dataset changes require a reviewed dataset card, source and license evidence, a new immutable version/fingerprint, updated counts, a split impact analysis, and privacy review. Never replace a split silently.

## Documentation and assets

Use descriptive alt text. Add only original or clearly licensed assets and update `docs/assets/README.md`. Do not edit generated visuals without updating their generator.

Run:

```powershell
python scripts/docs/check_readme_links.py
python scripts/docs/check_assets.py
python scripts/docs/validate_result_tables.py
python scripts/docs/check_markdown.py
git diff --check
```

## Pull-request checklist

- Scope and scientific impact are explained.
- Commands, tests, and results are reproducible.
- Result provenance is complete or values remain explicitly pending.
- Docs, citation, and asset inventory are updated.
- No secrets, private data, restricted assets, or unsupported claims are introduced.
