# Troubleshooting

## Export: TensorFlow/tf-keras version mismatch

**Symptom:** Export crashes or fails with TensorFlow/tf-keras import errors.

**Cause:** The base or Python 3.13 environment has incompatible versions.

**Fix:** Use a clean Python 3.12.2 venv with `requirements-export-litert.txt`.

```bash
python -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -r requirements-export-litert.txt
```

## Export: weights not found

The export script requires a trained `best.pt`. If you do not have one:

```bash
python scripts/03_train_yolo26m_asl_ldam_simam_dcfr.py --device 0 --skip-if-complete
```

Full training is not required for the refactor; use the provided TFLite files
under `export/` if you only need inference.

## Dataset root not found

`scripts/01_prepare_dataset_and_split.py` searches recursively for a folder
containing all four class directories (`1. Healthy`, `2. BG`, `3. WSSV`,
`4. WSSV_BG`, or the plain names). If it fails, pass `--data-root` explicitly:

```bash
python scripts/01_prepare_dataset_and_split.py --data-root /path/to/processed_images --seed 42
```

## KaggleHub authentication/download failure

Confirm Internet access and Kaggle credentials, then rerun
`scripts/01_prepare_dataset_and_split.py` (it downloads via KaggleHub when
`--data-root` is not given).

## Run directory already exists (exist_ok)

Training scripts default to `--skip-if-complete`: they skip a run if
`status.json` reports `ok` and `best.pt` exists. To force a fresh run:

```bash
python scripts/03_train_yolo26m_asl_ldam_simam_dcfr.py --force
```

To resume an interrupted run:

```bash
python scripts/03_train_yolo26m_asl_ldam_simam_dcfr.py --resume
```

## CUDA out of memory

Reduce `batch` in the YAML config and set `workers` to 2. The default paper
batch is 32; the runner does not silently change reported settings.

## Missing `yolo26m-cls` asset

The installed Ultralytics version may not expose that model name. Record the
version and use the release compatible with the original experiment runtime;
do not silently substitute another architecture.

## Tests skip due to missing optional libraries

Tests skip PyTorch-only and interpreter-only checks when those libraries are
absent. Syntax, corruption, and structure tests must still pass. Run the full
suite with:

```bash
python -m pytest tests -q
```

## Long Bash heredocs error in terminal

Prefer the provided `.py` and `.sh` scripts over pasting long heredocs. The
scripts under `scripts/` are the supported entry points.
