# Mode G High-Pass Local Sweep

Run order:

1. `014_mode_G_highpass_local_sweep_seed42.ipynb`

Primary objective:

- Beat the current teammate mAP50 anchor of about `0.54`.
- Stretch target: labeled test mask mAP50 `0.56-0.57`.

Fixed experiment context:

- stratified grouped-specimen split
- seed 42
- `yolo11n-seg.pt`
- clean-light YOLO augmentation on
- default Ultralytics Albumentations hook on
- epochs 100
- patience 40
- Fourier preprocessing applied to train, valid, and test

Candidate list inside `HIGHPASS_LOCAL_SWEEP_MODES`:

| Mode | Transform | Sigma | Alpha | Purpose |
|---|---|---:|---:|---|
| T0 | high-pass boost | 50 | 0.10 | repeat current G4 anchor |
| T1 | high-pass boost | 35 | 0.08 | sharper cutoff, mild boost |
| T2 | high-pass boost | 35 | 0.10 | sharper cutoff, same alpha as G4 |
| T3 | high-pass boost | 35 | 0.12 | sharper cutoff, stronger boost below failed 0.15 |
| T4 | high-pass boost | 40 | 0.08 | local candidate |
| T5 | high-pass boost | 40 | 0.10 | strongest prior guess |
| T6 | high-pass boost | 40 | 0.12 | stronger local candidate |
| T7 | high-pass boost | 45 | 0.10 | close to G4 with smaller cutoff shift |
| T8 | high-pass boost | 45 | 0.12 | close to G4 with mild stronger boost |
| T9 | high-pass boost | 55 | 0.10 | smoother cutoff near G4 |

Selection rule:

- Primary: labeled test mask mAP50.
- Strong promote: mAP50 `>= 0.56`.
- Acceptable promote: mAP50 `>= 0.55`.
- Beat-teammate threshold: mAP50 `> 0.54`.
- Secondary diagnostics: healthy FP, mAP50-95, disease miss rate, and count MAE.

This is a discovery sweep against seed 42. If a winner emerges, freeze the setting and confirm against the clean baseline and G4 over seeds 123 and 3407.
