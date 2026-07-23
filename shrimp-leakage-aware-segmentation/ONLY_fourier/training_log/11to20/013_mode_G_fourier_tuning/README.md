# Mode G Fourier Tuning

Run order:

1. `013_mode_G_fourier_tuning_seed42.ipynb`

Fixed experiment context:

- stratified grouped-specimen split
- seed 42
- `yolo11n-seg.pt`
- clean-light YOLO augmentation on
- default Ultralytics Albumentations hook on
- epochs 100
- patience 40

Candidate list inside `FOURIER_TUNING_MODES`:

| Mode | Fourier transform | Sigma | Alpha | Purpose |
|---|---|---:|---:|---|
| G0 | none | | | Mode D reference |
| G1 | high-pass boost | 50 | 0.03 | very weak boost |
| G2 | high-pass boost | 50 | 0.05 | balanced weak boost candidate |
| G3 | high-pass boost | 50 | 0.07 | midpoint before current G |
| G4 | high-pass boost | 50 | 0.10 | current Mode G replication |
| G5 | high-pass boost | 50 | 0.15 | stronger boost stress test |
| G6 | high-frequency damping | 50 | 0.05 | mild damping |
| G7 | high-frequency damping | 50 | 0.10 | prior damping candidate in Mode G context |
| G8 | high-frequency damping | 70 | 0.10 | wider damping cutoff |

Promote a setting only if it improves healthy-aware score against G0/Mode D without a meaningful healthy false-positive increase, and then confirm the finalist across seeds 123 and 3407.
