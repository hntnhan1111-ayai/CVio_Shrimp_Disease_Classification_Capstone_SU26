# Repository audit

## Passed

- Source evidence manifest: 321 entries, verified before refactoring.
- Training history: 200 epochs.
- Metrics: 2 clean rows plus 100 corruption rows = 102 rows.
- Corruption coverage: 10 families × 5 severities × 2 models.
- Top-five: all selected families win mAP50 and mAP50-95 at 5/5 severities.
- Locked checkpoint SHA-256 values verified.
- Python files compile successfully.
- Public tables use percentages; raw data retains decimal values.

## Deliberate reductions

- The 115-MB text log is stored as a compressed `.gz` file.
- Deterministically generated corruption datasets are excluded.
- Nested ZIP duplicates are excluded.
- A full vendored Ultralytics tree is replaced by an exact version pin and idempotent patch.

## Evidence gap

The source collection did not include standalone validation/test artifact directories. Numeric validation/test metrics are available and regeneration scripts are included.
