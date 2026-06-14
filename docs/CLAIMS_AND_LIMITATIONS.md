# Claims and Limitations

## Supported Claims

- On the fixed seed-42 Stage-1 split, YOLO26m-cls with ASL-LDAM and
  SimAM-DCFR improved over the native CE baseline in Macro-F1, accuracy, and
  Cohen's Kappa.
- The top-5 corruptions were selected from the weakest CE-baseline mean
  Macro-F1 values across severities 1-3.
- The repository provides clean-test, corruption, XAI, efficiency, and
  architecture-comparison artifacts for research reproducibility.

## Required Scope Statements

- Results are seed-42-only.
- No multi-seed mean ± std claim is made.
- No statistical significance claim across seeds is made.
- The paper runtime was Kaggle T4x2 GPU with Python 3.12.3.
- The Kaggle dataset was already background-removed using U2Net/rembg.
- No trained weights or checkpoints are included.

## Limitations

- Results may vary slightly because of GPU nondeterminism, library versions,
  pretrained asset changes, and Ultralytics internals.
- The split is image-level and is not asserted to prevent identity-level
  leakage when multiple images originate from the same shrimp.
- Exact package versions for missing local dependencies are not invented.
  Regenerate `docs/ENVIRONMENT.md` in the target Kaggle runtime.
- The main result is specific to the evaluated four-class dataset and does not
  establish general superiority across architectures, datasets, or seeds.
- Generated corruptions approximate noisy imaging conditions; they do not
  cover all real acquisition failures.
- TIMM baseline provenance includes a user-provided seed-42 result table.

## Non-Intended Use

This repository and its models are not clinical or veterinary diagnostic
tools. They are intended for research reproducibility and method evaluation.

Avoid claims of statistical significance, state of the art, robustness across
random seeds, globally best architecture, or validated field deployment.
