# Baseline Optimization Folder Structure

This folder is the working area for optimization experiments after the fair grouped-specimen baseline has been established.

The purpose is to keep optimization notebooks, run outputs, exported reports, and reusable generation rules separate from the main leakage-analysis evidence.

## Recommended Structure

```text
baseline-optim/
  folder_structure.md
  skills/
    shrimp-seg-optimization-notebook/
      SKILL.md

  notebooks/
    yolo_aug_policy_search/
    copy_paste_sweep/
    photometric_randaug/
    frequency_wavelet_revisit/
    background_removal/
    multi_seed_confirmation/

  configs/
    baseline/
    yolo_aug_policy_search/
    copy_paste_sweep/
    photometric_randaug/
    frequency_wavelet_revisit/
    background_removal/

  results/
    yolo_aug_policy_search/
    copy_paste_sweep/
    photometric_randaug/
    frequency_wavelet_revisit/
    background_removal/
    multi_seed_confirmation/

  reports/
    paper_tables/
    figures/
    run_summaries/
    failure_analysis/

  exports/
    kaggle/
    colab/
```

## Folder Roles

| Folder | Purpose |
|---|---|
| `skills/` | Reusable Codex-style rules for generating optimization notebooks consistently. |
| `notebooks/` | Generated Kaggle/Colab-ready notebooks. Keep one notebook per experiment family. |
| `configs/` | Saved run configs, experiment dictionaries, selected hyperparameters, and frozen baseline settings. |
| `results/` | Downloaded CSVs, `results.csv`, metrics summaries, partial logs, and checkpoint references from each run. |
| `reports/` | Human-readable findings, paper tables, figures, and analysis notes. |
| `exports/` | Zipped artifacts downloaded from Kaggle/Colab sessions. |

## Notebook Naming Convention

Use this pattern:

```text
[AIP491_01]yolo_seg_<phase>_group_stratified_<platform>.ipynb
```

Examples:

```text
[AIP491_01]yolo_seg_yolo_aug_policy_search_group_stratified_colab_kaggle.ipynb
[AIP491_01]yolo_seg_copy_paste_sweep_group_stratified_colab_kaggle.ipynb
[AIP491_01]yolo_seg_photometric_randaug_group_stratified_colab_kaggle.ipynb
[AIP491_01]yolo_seg_frequency_wavelet_revisit_group_stratified_colab_kaggle.ipynb
```

Use `colab_kaggle` when the notebook is written to support both platforms through runtime flags.

## Experiment Phase Order

Recommended execution order:

1. `yolo_aug_policy_search`
2. `copy_paste_sweep`
3. `photometric_randaug`
4. `frequency_wavelet_revisit`
5. `background_removal`
6. `multi_seed_confirmation`

The first phase should focus on tuning the augmentation behavior that YOLO already supports safely for segmentation.

## Required Artifacts Per Run

Each completed training session should export:

- main summary CSV
- paper-table CSV
- partial CSV if interrupted
- split manifest CSV/JSON
- run config JSON/YAML
- training `results.csv`
- best checkpoint path and, when feasible, `best.pt`
- metrics on full validation set
- metrics on labeled-only validation set
- metrics on full test set
- metrics on labeled-only test set
- healthy false-positive report
- disease count/miss report
- final export zip

## Metric Selection Rule

Do not select models by labeled test mAP50 alone.

Preferred decision order:

1. Select candidates using validation healthy-aware score.
2. Check labeled validation/test mask mAP50.
3. Check healthy false-positive rate.
4. Check disease missed-image rate and count MAE.
5. Confirm top candidates across grouped-specimen seeds.

## Paper-Facing Principle

Every optimization should answer this question:

> Does this method improve shrimp disease segmentation under fair unseen-specimen evaluation, or does it only look useful under easier/leaky conditions?

