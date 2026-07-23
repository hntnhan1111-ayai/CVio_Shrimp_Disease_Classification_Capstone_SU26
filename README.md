# CVio Shrimp Disease Segmentation

This workspace contains the active, paper-facing shrimp disease segmentation project and older exploratory work.

## Active Project

The current source of truth is [`shrimp-leakage-aware-segmentation/`](shrimp-leakage-aware-segmentation/).

Read these files first:

1. [`shrimp-leakage-aware-segmentation/README.md`](shrimp-leakage-aware-segmentation/README.md) for the project contract and run order.
2. [`shrimp-leakage-aware-segmentation/research/optimization_deep_research_2026.md`](shrimp-leakage-aware-segmentation/research/optimization_deep_research_2026.md) for the current optimization plan.
3. [`shrimp-leakage-aware-segmentation/notebooks/README.md`](shrimp-leakage-aware-segmentation/notebooks/README.md) for the notebook catalogue.
4. [`shrimp-leakage-aware-segmentation/ONLY_fourier/fourier_consolidated_effects_report.md`](shrimp-leakage-aware-segmentation/ONLY_fourier/fourier_consolidated_effects_report.md) for the consolidated Fourier findings.
5. [`shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/EXPERIMENT_INDEX.md`](shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/EXPERIMENT_INDEX.md) for the grouped Fourier experiment archive and coverage status.

### Active repository layout

```text
shrimp-leakage-aware-segmentation/
├── notebooks/       Reproducible training and evaluation notebooks
├── ONLY_fourier/    Fourier experiments, evidence, reports, and grouped logs
├── research/        Current optimization plan
├── paper/           Manuscript, references, figures, and LaTeX support files
├── requirements.txt Python package requirements for local inspection
├── CITATION.cff     Citation metadata
└── LICENSE          Project license
```

## Reproducibility Contract

The paper-facing experiments use:

- `YOLO11n-seg` for the primary segmentation baseline
- the hand-labeled shrimp disease dataset hosted on [Roboflow](https://universe.roboflow.com/lets-try-this/shrimpdishandsegv2)
- disease-stratified grouped-specimen splitting, so images of one shrimp never cross train, validation, and test
- seed `42` as the primary run, with multi-seed confirmation reserved for selected candidates
- healthy-aware diagnostics in addition to mask mAP, including healthy false-positive rate and disease miss rate

Do not commit Roboflow credentials. Cloud notebooks expect the secret `ROBOFLOW_API_KEY`.

## Running Experiments

The notebooks are intended for Kaggle or Colab. Start with the grouped clean-light baseline, then follow the notebook guide for split-policy, model, and augmentation experiments. For the Fourier direction, use the grouped logs and reports under `ONLY_fourier`; do not treat generated Kaggle/Colab output directories as repository source.

Preserve these outputs after cloud runs:

- summary and paper-row CSV files
- partial CSV files when a run is interrupted
- split manifests and split fingerprints
- `results.csv` and selected `best.pt` checkpoints

Large training outputs are not part of the active repository unless explicitly retained as evidence.

## Archived Work

[`archived/`](archived/) contains earlier research packages and deployment experiments that are kept for reference. This includes the former augmentation research package, LiteRT/Android work, mobile accuracy optimization, ONNX models, and older standalone notebooks.

Archived files are not the active implementation or the current paper source of truth. New experiments and documentation should be added under `shrimp-leakage-aware-segmentation/`.

## License and Citation

See [`shrimp-leakage-aware-segmentation/LICENSE`](shrimp-leakage-aware-segmentation/LICENSE) and [`shrimp-leakage-aware-segmentation/CITATION.cff`](shrimp-leakage-aware-segmentation/CITATION.cff). Dataset images and annotations remain subject to their original hosting and dataset terms.
