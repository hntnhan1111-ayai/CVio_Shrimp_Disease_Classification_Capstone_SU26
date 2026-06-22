# Notebook Guide

These notebooks reproduce the paper-facing experiments for leakage-aware shrimp disease instance segmentation.

## Run Order

1. `baseline/yolo11n_grouped_clean_baseline.ipynb`
   - One-row paper baseline.
   - YOLO11n-seg, stratified grouped-specimen split, seed 42.

2. `split-policy/random_and_stratified_random_seed_variance.ipynb`
   - Leakage-prone references.
   - Plain random image split and stratified random image split across seeds 42, 123, and 3407.

3. `split-policy/grouped_specimen_seed_stability.ipynb`
   - Leakage-free grouped split across seeds 42, 123, and 3407.

4. `model-sweep/grouped_model_sweep.ipynb`
   - YOLOv8, YOLO11, and YOLO26 n/s comparison under the grouped split.

5. Optimization screening notebooks:
   - `yolo_aug_policy_search/yolo_aug_policy_search.ipynb`
   - `photometric_randaug/photometric_randaug_screening.ipynb`
   - `copy_paste_focused_sweep/copy_paste_focused_sweep.ipynb`
   - `frequency_wavelet_revisit/frequency_wavelet_revisit.ipynb`

6. Architecture interaction notebook:
   - `architecture/simam_ca_split_policy_sweep_rtx4090.ipynb`
   - YOLO11n-seg vs YOLO11n-seg + SimAM-CA under plain random, stratified random, and stratified grouped-specimen splits across seeds 42, 123, and 3407.
   - `architecture/simam_ca_baseline_recipe_diagnostic_kaggle.ipynb`
   - Seed-43 Kaggle diagnostic for SimAM-CA with baseline clean-light augmentation, hook on/off controls, and teammate heavy-augmentation control.

7. Noise robustness notebook:
   - `noise_ablation/yolo11n_grouped_clean_baseline_noise_ablation_kaggle.ipynb`
   - Retrains the grouped clean-light YOLO11n baseline, then evaluates the trained checkpoint on clean and five noisy test conditions.

## Shared Conventions

- Paper-facing baseline uses `DISABLE_ULTRALYTICS_ALBUMENTATIONS = False`.
- Dataset download expects a Roboflow API key from `ROBOFLOW_API_KEY` or a Kaggle secret.
- No API key should be committed in notebooks.
- Test metrics are report-only. Model/candidate selection should use validation healthy-aware score.
- Keep `SMOKE_RUN = True` only for syntax/runtime checks, not for paper metrics.
- Preserve generated report CSVs and split manifests after cloud runs.

## Main Output Types

- `*_summary.csv`: full experiment report.
- `*_partial.csv`: interrupted-run recovery report.
- `*_paper_table.csv` or `*_paper_row.csv`: compact paper-facing subset.
- `split_manifests/`: split membership, summary, and fingerprint files for leakage auditing.
