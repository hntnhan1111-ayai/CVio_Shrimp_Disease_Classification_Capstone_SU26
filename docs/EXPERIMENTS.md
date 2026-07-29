# Experiments

## Experiment contract

Every submitted experiment must include:

| Field | Requirement |
|---|---|
| Experiment ID | Stable, unique identifier |
| Commit | Full Git SHA of executable code |
| Track | Classification or instance segmentation |
| Dataset | Version and fingerprint |
| Split | Manifest path and hash |
| Config | Versioned repository path and config hash |
| Randomness | Seed and deterministic settings |
| Environment | OS, Python, framework, package lock, hardware |
| Training | Input, preprocessing, augmentation, optimizer, schedule, epochs |
| Selection | Validation-only checkpoint rule |
| Evaluation | Exact command and metric implementation |
| Artifact | Checkpoint path and cryptographic hash |
| Thresholds | Confidence, IoU, calibration, or decision thresholds |
| Evidence | Logs, predictions, metrics file, and failure notes |

## Registry

| Experiment ID | Track | Purpose | Evidence | Status |
|---|---|---|---|---|
| TBD-CLS-001 | Classification | Integrated baseline | Not submitted | Pending verification |
| TBD-SEG-001 | Instance segmentation | Integrated baseline | Not submitted | Pending verification |

## Comparison policy

- **Paper-reported:** transcribed from a cited publication; never mixed with local measurements.
- **Team reimplementation:** locally executed interpretation of an external method.
- **Proposed method:** frozen CVio definition with versioned implementation.
- **Ablation:** one declared factor differs from its parent experiment.
- **Selected final run:** selected by a stated validation rule before final test evaluation.

Test data must not guide model selection, thresholds, early stopping, or architecture decisions. Failed and partial runs remain visible in the registry when they influenced decisions.
