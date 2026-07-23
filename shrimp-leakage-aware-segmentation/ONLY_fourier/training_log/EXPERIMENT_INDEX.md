# ONLY_fourier Experiment Index

This index is the canonical navigation point for the grouped training logs. Raw evidence is catalogued in `../evidence/only_fourier_artifact_manifest.csv`.

| Group | Experiments | Research purpose | Canonical location |
|---|---|---|---|
| 01-10 | 001-010 | Strict controls, image-level high-pass, band-pass, damping, illumination, and train-copy variants | `training_log/1to10/` |
| 11-20 | 011-022 | Alpha/config selection, Factorial A-H, local sweep, strong augmentation, attention, and FDDem feature methods | `training_log/11to20/` |
| 21-30 | 021-029 | FDDem confirmation/hybrid ideas and N=1 HSV/geometric/erasing Fourier interaction | `training_log/21to30/` |
| 31-40 | 031-040 | Offline N=1 order studies and first N=2 augmentation pairs | `training_log/31to40/` |
| 41-50 | 041-046 | Remaining N=2 pairs, noise robustness, and inference timing | `training_log/41to50/` |

## Canonical Selection Rules

When multiple files exist for one experiment, use this order:

1. Executed notebook with outputs.
2. Raw CSV, JSON, or TXT metrics/log.
3. Analysis Markdown.
4. Unexecuted generator notebook or script.

Partial CSVs are retained as provenance and are not treated as final metrics when a second-run paper row exists.

## Known Coverage Gaps

- Experiment 16 has a notebook and a Downloads summary, but no executed notebook output is retained in the canonical tree.
- Experiments 19-20 have generator/notebook evidence and Downloads paper rows, but no executed notebook output is retained.
- Experiments 28 and 30 have generated notebooks but no confirmed executed metrics artifact.
- Some N=1 experiments have both first-run partial and second-run final outputs; both are indexed, with the second-run paper row preferred.
- The earlier flat `training_log/` paths were replaced by the grouped folders in the current uncommitted move. Do not restore the flat copies after this index is accepted.
