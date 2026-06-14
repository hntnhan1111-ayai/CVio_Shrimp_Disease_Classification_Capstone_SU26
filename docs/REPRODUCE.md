# Reproduce the Seed-42 Paper Artifacts

## Kaggle Notebook Setup

Use a Kaggle Notebook with T4x2 GPU and Python 3.12.3. Enable Internet access
for KaggleHub and pretrained model downloads.

```bash
git clone https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26.git
cd CVio_Shrimp_Disease_Classification_Capstone_SU26
git switch paper/asl-ldam
python -m pip install -r requirements.txt
```

Set portable paths:

```bash
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/datasets/processed-images}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs}"
export SEED="${SEED:-42}"
export DEVICE="${DEVICE:-auto}"
```

`DEVICE=auto` resolves to `0,1` with two CUDA GPUs, `0` with one GPU, or CPU.

## Dataset Download

```bash
python scripts/00_download_dataset.py
```

Set `DATA_DIR` to the printed KaggleHub path or copy/link that path into the
default `datasets/processed-images` location.

The Kaggle data are already background-removed with U2Net/rembg. Do not run
background removal in the default workflow. Optional experimentation:

```bash
pip install "rembg[cpu]"
```

## Environment Setup

```bash
python scripts/99_env_report.py
```

This writes:

- `artifacts/logs_sample/environment_report.json`
- `docs/ENVIRONMENT.md`

## Smoke Test

```bash
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m cvio_asl_ldam.models.yolo_train \
  --config configs/train/yolo26m_ce.yaml \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --smoke-test
```

Run lightweight repository tests:

```bash
PYTHONPATH=src pytest -q
```

## Fixed Split

```bash
bash scripts/01_make_seed42_split.sh
```

The command audits 1,149 readable images, applies the curated split assignment,
and writes `artifacts/manifests/split_manifest_seed42_generated.csv`.

## Full Clean-Test Reproduction

YOLO26m native cross-entropy:

```bash
MODEL=yolo26m-cls SEED=42 bash scripts/02_run_yolo_baselines.sh
```

Main method:

```bash
bash scripts/04_run_yolo26m_asl_ldam.sh
```

TIMM baseline example:

```bash
MODEL=convnext_tiny_in22k SEED=42 bash scripts/03_run_timm_baselines.sh
```

## Noise Robustness Reproduction

```bash
bash scripts/05_run_noise_eval.sh
```

The runner evaluates only:

1. `impulse_noise`
2. `gaussian_noise`
3. `contrast_reduction`
4. `defocus_blur`
5. `low_light`

Each corruption is evaluated at severities 1, 2, and 3.

## XAI Reproduction

```bash
bash scripts/06_run_xai.sh
```

Override the checkpoint when necessary:

```bash
WEIGHTS=/path/to/best.pt bash scripts/06_run_xai.sh
```

## Result Collection

```bash
bash scripts/07_collect_paper_results.sh
```

The collector copies only small CSV, JSON, and PNG outputs. It excludes
checkpoints and model files.

## Expected Outputs

- `runs/training/*/paper_evaluation/clean_test_metrics.json`
- `runs/training/*/paper_evaluation/clean_test_predictions.csv`
- `runs/training/*/paper_evaluation/clean_test_confusion_matrix.{csv,png}`
- `runs/noise_robustness/top5_noise_by_severity.csv`
- `runs/noise_robustness/top5_noise_mean_summary.csv`
- `runs/xai/*__gradcam.png`
- curated outputs under `artifacts/` after collection

## Common Errors and Fixes

**Dataset root not found**

Set `DATA_DIR` to a folder containing the four class directories. Both
`Healthy` and numbered names such as `1. Healthy` are supported.

**KaggleHub authentication/download failure**

Confirm Internet access and Kaggle credentials, then rerun
`scripts/00_download_dataset.py`.

**CUDA out of memory**

Reduce `batch` in the YAML config and set workers to 2. The default paper batch
is 32; the portable runner does not silently change reported settings.

**Two-GPU custom trainer issue**

Set `DEVICE=0` to isolate whether the installed Ultralytics release changed DDP
serialization behavior. Regenerate the environment report and retain the log.

**Missing `yolo26m-cls` asset**

The installed Ultralytics version may not expose that model name. Record the
version and use the release compatible with the original experiment runtime;
do not silently substitute another architecture.

**Optional heavy libraries unavailable during tests**

Tests skip PyTorch-only checks when PyTorch is absent. Syntax and NumPy
corruption tests must still pass.
