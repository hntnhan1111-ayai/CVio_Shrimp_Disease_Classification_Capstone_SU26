# Troubleshooting

This document addresses common issues encountered when reproducing the
**shrimpdb_combined_multisource** study.

## 1. GPU Detection Issues

### 1.1 CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

1. Verify that the runtime has a CUDA-capable GPU. On Kaggle, ensure the accelerator is set
   to GPU (not TPU or None) in the notebook settings.
2. Check that the PyTorch version includes CUDA support: `python -c "import torch; print(torch.version.cuda)"`
3. If using a local machine, ensure that NVIDIA drivers are installed and that the CUDA
   version matches the PyTorch build (2.10.0+cu128 requires CUDA 12.8).

### 1.2 Mismatched GPU Count

The training configuration specifies `device: auto` and `gpu_count: 2`. If only one GPU is
available, the effective global batch size will be halved. To enforce a single GPU:

```bash
uv run python scripts\03_train_shrimpdb3.py --config configs\study.yaml --device 0
```

The scripts accept a `--device` argument that overrides the `auto` detection.

### 1.3 Peer Access / NCCL Errors

If distributed training fails with NCCL errors:

1. Check that `disable_nccl_p2p` is `false` in the runtime configuration.
2. On some Kaggle runtimes, peer access between Tesla T4 GPUs is available but may require
   environment variable adjustment. The script sets `NCCL_P2P_DISABLE=0` by default.
3. If issues persist, run with a single GPU as a fallback.

## 2. Out of Memory (OOM)

### 2.1 Symptoms

The training process is killed with a CUDA out-of-memory error, typically during the first
epoch.

### 2.2 Solutions

1. **Reduce batch size**: The reported global batch size is 32 across 2 GPUs (16 per GPU).
   Reducing the batch size will require adjusting the learning rate proportionally.
2. **Disable AMP**: If automatic mixed precision is causing memory issues on specific
   hardware, set `amp: false` in `configs/study.yaml`.
3. **Reduce image size**: The input size is 224 x 224. Reducing to 192 x 192 will lower
   memory usage.
4. **Disable caching**: The configuration already sets `cache: false`. Do not enable
   `disk_cache` or `ram_cache` if memory is constrained.
5. **Reduce workers**: Lower `workers` from 4 to 2 or 0 to reduce data-loader memory
   overhead.

## 3. Dataset Path Issues

### 3.1 Dataset Not Found

If the scripts report that the dataset root is not found:

1. Verify that the dataset was downloaded to the expected path:
   - ShrimpDB: `datasets/shrimpdb/`
   - ShrimpDiseaseDB: `datasets/shrimpdiseasedb/`
2. Alternatively, set environment variables before running:
   ```powershell
   $env:SHRIMPDB_ROOT = "D:\path\to\shrimpdb"
   $env:SHRIMPDISEASEDB_ROOT = "D:\path\to\shrimpdiseasedb"
   ```
3. The script will search for common Kaggle-exported directory structures. If your directory
   structure differs, update the `root` fields in `configs/study.yaml`.

### 3.2 Class Count Mismatch

If the audit script reports a count mismatch:

1. Verify that the dataset version matches the expected counts in `configs/study.yaml`.
2. The expected counts are: ShrimpDB (Den_Mang 125, Dom_Den 103, Dom_Trang 173,
   Hoai_Tu_Co 115, Hoai_tu_gan 61, Tom_BT 74) and ShrimpDiseaseDB (Healthy 403, BG 198,
   WSSV 328, WSSV_BG 220).
3. If the dataset has been updated by the curator since the study was conducted, counts may
   differ. Update `expected_counts` in `configs/study.yaml` accordingly.

## 4. Split Already Exists

If `02_prepare_splits.py` reports that the split already exists:

1. The script is guarded against overwriting existing split manifests.
2. To force regeneration, delete the `artifacts/experiments/` directory for the relevant
   experiment and re-run.
3. Alternatively, check for the `--force` flag if available in the script.

## 5. Checkpoint Selection Confusion

If `05_evaluate_checkpoints.py` reports metrics that differ from the documented values:

1. Verify that the checkpoint SHA-256 matches the documented checkpoint:
   ```bash
   sha256sum path/to/checkpoint.pt
   ```
2. Ensure that the evaluation is using the test split, not the validation split.
3. The `last.pt` and `best.pt` checkpoints will produce different metrics. Use
   `best.pt` for the canonical application metrics.

## 6. Import Errors

If `uv run python scripts\00_check_environment.py` fails with import errors:

1. Ensure the virtual environment is activated: `.\.venv\Scripts\Activate.ps1`
2. Run `uv sync --frozen` to install all pinned dependencies.
3. Verify that Python 3.12 is the active interpreter: `python --version`

## 7. Kaggle Notebook Time Limits

Kaggle Notebooks have session time limits (typically 9-12 hours). The full training and
evaluation pipeline may exceed this limit. Strategies:

1. Run training and evaluation in separate notebook sessions, saving checkpoints between
   sessions.
2. Use Kaggle's persistent storage (`/kaggle/working/`) to preserve artifacts between
   sessions.
3. Monitor training progress via the `runs/` directory, which contains `status.json` files
   written by each script.

## 8. Determinism Failures

If results differ across runs despite `deterministic: true`:

1. This is expected behaviour. PyTorch does not guarantee full determinism across all
   operations, particularly on newer GPU architectures or CUDA versions.
2. For reproducibility of the exact reported numbers, the study must be rerun on the same
   hardware (Kaggle Notebook with 2 x Tesla T4) and software versions listed in
   `REPRODUCIBILITY.md`.
