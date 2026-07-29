# Results

No integrated result has passed provenance verification on `main`.

## Verified result registry

| Experiment ID | Track | Dataset / split | Model | Seed | Primary metric | Checkpoint | Config | Status |
|---|---|---|---|---:|---|---|---|---|
| TBD-CLS-001 | Classification | TBD | TBD | TBD | TBD | TBD | TBD | Pending verification |
| TBD-SEG-001 | Instance segmentation | TBD | TBD | TBD | TBD | TBD | TBD | Pending verification |

## Publication gate

A row becomes `Verified` only when:

1. all required experiment fields are present;
2. the dataset and split fingerprints resolve;
3. the checkpoint and config hashes match;
4. the evaluation command reruns against immutable inputs;
5. the metrics file agrees with the displayed value and unit;
6. task, averaging, threshold, and uncertainty definitions are explicit;
7. failures and excluded samples are documented;
8. an independent reviewer signs off in the pull request.

## Status vocabulary

`Verified`, `Pending rerun`, `Paper-reported`, `Team reimplementation`, `Ablation`, and `Selected final run` are evidence labels. More than one can apply, but `Verified` requires the publication gate.

## Qualitative reporting

Future qualitative panels must state experiment ID, sample ID, dataset split, selection method, predicted and reference labels, threshold, and visualization method. Include representative failure cases and avoid selecting only visually favorable outputs.

Raw result artifacts should be registered under `docs/assets/results/` only when they are small, redistributable, and directly traceable.
