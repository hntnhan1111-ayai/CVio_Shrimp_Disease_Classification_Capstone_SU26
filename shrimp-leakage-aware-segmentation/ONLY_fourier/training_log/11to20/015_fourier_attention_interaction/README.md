# Fourier G4 x SimAM+CA Interaction

Run order:

1. `015_fourier_G4_simam_ca_interaction_seed42.ipynb`

Primary objective:

- Test whether Fourier G4 still improves mAP50 after adding the teammate's stronger augmentation and SimAM+CA attention head.
- Beat the teammate SimAM+CA anchor: labeled test mask mAP50 about `0.5507`.
- Stretch target: labeled test mask mAP50 `0.56-0.57`.

Fixed experiment context:

- stratified grouped-specimen split
- seed 42
- base weights: `yolo11n-seg.pt`
- epochs 100
- patience 30
- explicit strong YOLO augmentation from the teammate notebook
- default Ultralytics Albumentations hook off
- best.pt evaluation

Strong augmentation policy:

- `erasing=0.15`
- `fliplr=0.5`
- `flipud=0.3`
- `hsv_h=0.05`
- `hsv_s=0.50`
- `hsv_v=0.40`
- `degrees=10`
- `translate=0.10`
- `scale=0.50`
- `mosaic=0`
- `mixup=0`
- `copy_paste=0`

Candidate list:

| Mode | Fourier G4 | SimAM+CA | Purpose |
|---|---|---|---|
| A | off | off | strong-aug hook-off baseline |
| B | on | off | Fourier effect under strong aug |
| C | off | on | attention effect under strong aug |
| D | on | on | Fourier + attention interaction |

Fourier G4:

- transform: high-pass boost
- sigma: 50
- alpha: 0.10
- applied to train, valid, and test

Selection rule:

- Primary: labeled test mask mAP50.
- Strong promote: mAP50 `>= 0.56`.
- Useful promote: mAP50 `> 0.5507`.
- Secondary diagnostics: mAP50-95, healthy FP, disease miss rate, count MAE, healthy-aware score.

Interpretation logic:

- If B > A, Fourier helps under strong augmentation.
- If C > A, SimAM+CA reproduces the teammate's architecture gain.
- If D > C, Fourier adds value beyond the attention architecture.
- If D reaches `0.56-0.57`, freeze this setting and confirm across seeds 123 and 3407.
