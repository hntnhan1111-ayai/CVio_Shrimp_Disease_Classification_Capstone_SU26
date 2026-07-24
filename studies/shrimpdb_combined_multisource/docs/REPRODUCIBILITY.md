# Reproducibility

## Merged-package validation

Run `uv run python tools/generate_merged_best_artifacts.py` to validate the reviewed ZIP,
verify the selected metrics against `metrics_raw.json` and provenance, import only selected
evaluation artifacts, and regenerate tables, figures, registries, and the HTML report. The
command does not copy checkpoints or edit raw metrics. Run `uv run pytest -q` afterward.

This document provides instructions for reproducing the **shrimpdb_combined_multisource**
study, including environment setup, package versions, and command references.

## 1. Runtime Environment

The reported results were generated on a **Kaggle Notebook** runtime. The exact runtime
environment is documented below.

### 1.1 Hardware

| Component | Specification |
|---|---|
| Platform | Kaggle Notebook runtime |
| Accelerator | 2 x Tesla T4 |
| GPU memory per device | ~14.56 GiB |
| Distributed backend | torchrun, NCCL peer access enabled |

> The reported run used two Tesla T4 GPUs.

### 1.2 Software Versions

All versions are taken from the verified runtime configuration (`configs/study.yaml` and
`artifacts/final_application_model/effective_training_config.yaml`).

| Package | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| TorchVision | 0.25.0+cu128 |
| Ultralytics | 8.4.75 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.10.0 |
| Pillow | 11.3.0 |
| PyYAML | 6.0.3 |

## 2. Local Setup

The project uses `uv` for package and environment management.

### 2.1 Commands (Windows PowerShell)

```powershell
Set-Location "D:\CVio\CVio_ShrimpDB_Combined_Research_Repo\studies\shrimpdb_combined_multisource"
uv python install 3.12
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv sync --frozen
uv run python scripts\00_check_environment.py
uv run pytest -q
```

### 2.2 Commands (Linux/macOS)

```bash
cd D:/CVio/CVio_ShrimpDB_Combined_Research_Repo/studies/shrimpdb_combined_multisource
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync --frozen
uv run python scripts/00_check_environment.py
uv run pytest -q
```

## 3. Kaggle Reproduction

To reproduce on Kaggle, the datasets must first be added to the notebook environment via the
Kaggle Datasets panel:

- `vohoangtu/shrimpdb`
- `uynnhy/processed-images`

### 3.1 Kaggle Notebook Commands

```bash
uv run python scripts\01_audit_datasets.py --config configs\study.yaml
uv run python scripts\02_prepare_splits.py --config configs\study.yaml
uv run python scripts\03_train_shrimpdb3.py --config configs\study.yaml
uv run python scripts\04_train_combined4.py --config configs\study.yaml
uv run python scripts\05_evaluate_checkpoints.py --config configs\study.yaml
uv run python scripts\06_generate_academic_artifacts.py --config configs\study.yaml
uv run python scripts\07_generate_html_report.py --config configs\study.yaml
uv run python scripts\08_package_results.py --config configs\study.yaml
```

## 4. Determinism Notes

- **Fixed seed**: All experiments use a fixed random seed of 42.
- **Deterministic algorithms**: The training configuration sets `deterministic: true`, which
  requests deterministic CUDA algorithms where available in PyTorch.
- **No multi-seed statistics**: This repository contains fixed seed-42 results only. It does
  not report mean ± standard deviation across multiple random seeds, nor does it claim
  statistical significance.
- **Residual nondeterminism**: Despite the above settings, minor numerical differences may
  occur across different GPU models, CUDA versions, or PyTorch releases. Results should be
  treated as reference points rather than exact reproductions on dissimilar hardware.

## 5. Dataset Acquisition

Datasets must be obtained directly from Kaggle:

```bash
# Kaggle CLI (if installed)
kaggle datasets download vohoangtu/shrimpdb -p datasets/shrimpdb --unzip
kaggle datasets download uynnhy/processed-images -p datasets/shrimpdiseasedb --unzip
```

Alternatively, the scripts use `kagglehub` where available. The dataset root paths can be
overridden via environment variables:

- `SHRIMPDB_ROOT` for ShrimpDB
- `SHRIMPDISEASEDB_ROOT` for ShrimpDiseaseDB

## 6. Configuration

All experiment parameters are defined in `configs/study.yaml`. The training configuration
(`effective_training_config.yaml`) is generated at runtime and stored in
`artifacts/final_application_model/`.
