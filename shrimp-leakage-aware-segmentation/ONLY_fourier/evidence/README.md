# ONLY_fourier Evidence Intake

This is an audit intake area, not the canonical experiment tree.

- `only_fourier_artifact_manifest.csv` records repository and Downloads artifacts, SHA-256 hashes, notebook output status, and duplicate relationships.
- `experiment_coverage.csv` and `experiment_coverage.md` validate the notebook-to-output mapping for experiments 01-46.
- `duplicate_cleanup_candidates.json` records the exact duplicate/cache files removed after coverage validation.
- `intake_downloads/` contains only Downloads artifacts that were not exact duplicates of repository files.
- Original filenames are preserved. Downloads files were copied, not removed.
- Canonical experiment files remain under the grouped training-log folders; `training_log/EXPERIMENT_INDEX.md` records the selection priority and known gaps.
