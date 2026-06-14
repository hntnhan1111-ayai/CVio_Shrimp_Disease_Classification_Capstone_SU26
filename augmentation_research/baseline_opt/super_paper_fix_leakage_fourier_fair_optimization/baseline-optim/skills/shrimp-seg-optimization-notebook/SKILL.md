---
name: shrimp-seg-optimization-notebook
description: Use when generating or modifying Kaggle/Colab notebooks for leakage-safe shrimp disease YOLO segmentation optimization, including grouped-specimen splits, augmentation/preprocessing sweeps, boolean run controls, platform-aware paths, reproducible metrics, and artifact exports.
---

# Shrimp Segmentation Optimization Notebook

Use this skill for notebooks that optimize the hand-labeled shrimp disease segmentation baseline under specimen-grouped evaluation.

## Experimental Contract

Default baseline:

- Dataset: hand-labeled shrimp disease segmentation dataset.
- Split: stratified grouped-specimen split.
- Model: `yolo11n-seg`.
- Ultralytics version: pin to the version used in the accepted baseline unless the user explicitly changes it.
- Ultralytics hidden Albumentations hook: enabled.
- Baseline augmentation: clean-light YOLO augmentation.
- Main seed: `42`; final candidates should be repeated across multiple grouped-specimen seeds.
- Evaluation must include full-set and labeled-only validation/test metrics.

Do not use random image-level split for optimization unless the user explicitly asks for a leakage demonstration.

## Notebook Design Rules

Every generated notebook must be RUN ALL compatible.

Use top-level config cells with:

```python
class RuntimeMode(str, Enum):
    COLAB = "colab"
    KAGGLE = "kaggle"

RUNTIME_MODE = RuntimeMode.COLAB
SMOKE_TEST = False
```

Use boolean flags for experiment groups:

```python
RUN_BASELINE_CHECK = False
RUN_YOLO_AUG_POLICY_SEARCH = True
RUN_COPY_PASTE_SWEEP = False
RUN_PHOTOMETRIC_RANDAUG = False
RUN_FREQUENCY_WAVELET = False
RUN_BACKGROUND_REMOVAL = False
RUN_MULTI_SEED_CONFIRMATION = False
```

Represent iterative experiments as lists of dictionaries:

```python
EXPERIMENTS = [
    {
        "key": "copy_paste_low_flip",
        "enabled": True,
        "train_args": {"copy_paste": 0.10, "copy_paste_mode": "flip"},
        "preprocess": None,
        "notes": "Low-risk built-in segmentation copy-paste.",
    },
]
```

Never hard-code one-off runs inside training loops. A user should be able to enable/disable a run by changing only booleans or the `enabled` field.

## Platform-Aware Paths

Resolve paths from `RUNTIME_MODE`.

Required pattern:

```python
if RUNTIME_MODE == RuntimeMode.COLAB:
    ROOT_DIR = Path("/content/shrimp_yolo_seg_baseline_optim")
elif RUNTIME_MODE == RuntimeMode.KAGGLE:
    ROOT_DIR = Path("/kaggle/working/shrimp_yolo_seg_baseline_optim")
else:
    raise ValueError(RUNTIME_MODE)

DATASET_DIR = ROOT_DIR / "dataset_raw"
WORK_DIR = ROOT_DIR / "work"
REPORT_DIR = ROOT_DIR / "reports"
EXPORT_DIR = ROOT_DIR / "exports"
```

Package handling should be platform-aware:

- Colab: install missing packages explicitly.
- Kaggle: prefer installed packages, but pin/reinstall only when needed for baseline consistency.
- Always print package versions after installation/import.

## Data and Split Rules

The notebook must:

- download/load the hand-labeled segmentation dataset,
- normalize Roboflow hash filenames when needed,
- parse specimen IDs using the accepted hand-labeled/Mendeley naming convention,
- build stratified grouped-specimen train/valid/test splits,
- keep all images from one specimen in one split,
- preserve co-infection labels,
- remove stale YOLO label caches after rewriting split folders,
- save split manifests and summaries.

If a split cannot perfectly stratify classes because the dataset is small, print the compromise clearly.

## Training Rules

Each experiment must:

- create an isolated run directory,
- save the exact train args to JSON/YAML,
- print the final train config before training,
- use the same base model and image size unless the experiment explicitly changes them,
- restore/evaluate `best.pt`, not just the last epoch,
- save `results.csv`,
- tolerate interruption by writing partial summary CSVs after each completed run.

Use `SMOKE_TEST` to shrink epochs/patience and run at most one enabled experiment. Smoke mode must preserve the same code path as full training.

## Evaluation Rules

Evaluate every completed checkpoint on:

- full validation set,
- labeled-only validation set,
- full test set,
- labeled-only test set,
- healthy-only subset for false positives.

Required metrics:

- box mAP50,
- box mAP50-95,
- mask mAP50,
- mask mAP50-95,
- healthy false-positive rate,
- healthy false-positive masks per image,
- disease missed-image rate,
- over-prediction image rate,
- under-prediction image rate,
- mask count MAE,
- healthy-aware score.

Selection rule:

1. Rank by validation healthy-aware score.
2. Reject candidates with substantially worse healthy false positives unless mAP gain is large and explicitly justified.
3. Use test metrics for final reporting, not iterative selection.
4. Confirm top candidates across multiple grouped-specimen seeds.

## Optimization Phase Guidance

Preferred order:

1. YOLO augmentation policy search.
2. Copy-paste focused sweep.
3. Segmentation-safe photometric RandAugment-style policies.
4. Frequency/wavelet revisit.
5. Background-removal augmentation.
6. Multi-seed confirmation.

For YOLO augmentation policy search, prioritize:

- `copy_paste`,
- `copy_paste_mode`,
- `mosaic`,
- `close_mosaic`,
- `hsv_s`,
- `hsv_v`,
- `scale`,
- `translate`,
- low `mixup`.

For RandAugment-style segmentation policies, only use image-only photometric transforms unless mask handling is explicitly implemented and tested.

## Export Rules

Every notebook must create a final zip export containing:

- summary CSV,
- paper-table CSV,
- partial CSV,
- split manifests,
- run configs,
- training `results.csv`,
- healthy false-positive report,
- disease count/miss report,
- best checkpoint paths and optionally checkpoint files,
- generated figures when present.

Print a final "download these files" section with exact paths.

## Researcher-Friendly Practices

- Prefer small screening runs before expensive sweeps.
- Keep test set untouched for final comparison when possible.
- Add visual sanity checks for any custom image preprocessing.
- Do not introduce custom transforms that can desynchronize masks.
- Log enough information that a run can be understood after a Colab/Kaggle session dies.
- If a method improves mAP but worsens healthy false positives, report it as a tradeoff, not a win.
- Keep negative ablations; they are useful evidence under fair evaluation.

