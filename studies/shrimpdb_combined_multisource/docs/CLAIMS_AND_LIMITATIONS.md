# Claims and Limitations

## Supported claims

- Under the merged package's fixed-seed selection rule, the ShrimpDB-3 selected result is
  ASL-LDAM + SimAM-DCFR: 91.49% accuracy and 91.29% Macro-F1.
- The Combined-4 selected result is displayed under the common label ASL-LDAM + SimAM-DCFR,
  but its verified source method is CE Baseline: 87.73% accuracy and 86.51% Macro-F1.
- The selection rule is highest Macro-F1 within each dataset, with accuracy as tie-breaker.

## Not supported

These results do not establish state-of-the-art performance, universal superiority,
statistical significance, clinical validity, or production readiness. They are not a
replacement for expert diagnosis.

## Limitations

Evaluation uses one fixed seed (42), one validation-selected checkpoint per selected result,
and a source-wise image-level split. The split is not asserted to be specimen-safe. Dataset
shift, class imbalance, label harmonization assumptions, calibration, and external validity
may affect performance. Confidence intervals in the source artifacts are descriptive and do
not replace multi-seed or study-level uncertainty analysis.
