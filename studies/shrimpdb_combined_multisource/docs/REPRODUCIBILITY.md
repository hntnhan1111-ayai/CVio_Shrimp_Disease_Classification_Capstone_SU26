# Reproducibility

The reported execution used a Kaggle Notebook, 2 × Tesla T4 GPUs, seed 42, Python 3.12.13,
PyTorch 2.10.0+cu128, TorchVision 0.25.0+cu128, CUDA 12.8, and Ultralytics 8.4.75.

## Local installation

```powershell
Set-Location "D:\CVio\CVio_ShrimpDB_Combined_Research_Repo\studies\shrimpdb_combined_multisource"
uv python install 3.12
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv sync --frozen
```

## Reproduction commands

```powershell
uv run python scripts\01_audit_datasets.py --config configs\study.yaml
uv run python scripts\02_prepare_splits.py --config configs\study.yaml
uv run python scripts\03_train_shrimpdb3.py --config configs\study.yaml
uv run python scripts\04_train_combined4.py --config configs\study.yaml
uv run python scripts\05_evaluate_checkpoints.py --config configs\study.yaml
uv run python scripts\generate_paper_artifacts.py
uv run pytest -q
```

For Kaggle, attach `vohoangtu/shrimpdb`, enable Internet, select a GPU accelerator, and use
Save Version -> Run All. The notebook downloads `uynnhy/processed-images` when it is not
attached.

## Reproducibility scope

Raw datasets and `.pt` checkpoints are not redistributed. Results are fixed seed-42,
image-level experiments; no multi-seed uncertainty or specimen-level split guarantee is
claimed.
