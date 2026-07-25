# Noisy test samples from the CoTE-Gate notebook dataset

Source: Roboflow `lets-try-this/shrimpdishandsegv2`, version 1, `yolo26` export. The grouped, disease-stratified seed-42 split was recreated without moving source files: 129 test images (Healthy 41, BG 24, WSSV 38, WSSV_BG 26).

This folder contains 60 lossless PNG variants from 3 representative test images per disease, using all five paper corruptions at severity 2.

`recreated_notebook_test_split.csv` records every source in the recreated test split; `manifest.csv` records each generated PNG, its source, split index, seed, parameter, and SHA-256.

## Paper-compatible settings

- `impulse_noise`: `amount` = `0.035`
- `gaussian_noise`: `sigma` = `0.08`
- `contrast_reduction`: `factor` = `0.55`
- `defocus_blur`: `radius` = `2.0`
- `low_light`: `factor` = `0.45`

PNG avoids adding JPEG-compression artifacts after corruption. Source images are read-only.
