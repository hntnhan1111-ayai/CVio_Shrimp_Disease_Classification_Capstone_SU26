# W1 and Strong-Augmentation Spectrum Interaction

## Recorded training observation

| Regime | No Fourier labeled mAP50 | W1 labeled mAP50 | W1 delta |
|---|---:|---:|---:|
| No augmentation, hook off | 0.235374 | 0.345105 | +0.109731 |
| Strong augmentation, hook off | 0.595095 | 0.577126 | -0.017969 |

The seed-42 difference-in-differences interaction contrast is `-0.127700` (-12.77 percentage points).

## Measured image-spectrum interaction

- W1 high-band energy-share change without augmentation: `+0.00067017`.
- Incremental W1 high-band energy-share change after matched strong augmentation: `+0.00015073`.
- W1 clipping-fraction change without augmentation: `+0.00238098`.
- Incremental W1 clipping-fraction change after matched strong augmentation: `+0.00026408`.
- W1 mean-gradient change without augmentation: `+0.01097869`.
- Incremental W1 mean-gradient change after matched strong augmentation: `+0.00615721`.

## Interpretation boundary

These measurements show how the image distribution and spectrum change under paired preprocessing paths. They support or challenge a proposed mechanism, but they do not prove that the measured spectral change caused the model metric difference. The performance rows are seed-42 observations; multi-seed training is required before describing the negative interaction as statistically confirmed.

The strong-policy visualization excludes erasing because the Ultralytics 8.4.62 segmentation transform builder does not use the classification erasing setting. Consult `ultralytics_augmentation_contract.json` and the saved `v8_transforms` source before final report wording.
