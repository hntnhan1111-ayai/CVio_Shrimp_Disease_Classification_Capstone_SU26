# Improving Lightweight Shrimp Disease Classification with Co-Infection-Aware Losses and RandAugment

This repository contains clean, researcher-facing Python scripts for reproducing the shrimp disease classification experiments on Kaggle T4x2 GPUs with Python 3.12.

The code uses the processed Kaggle dataset `uynnhy/processed-images` only. It avoids background-removal, instance-mask, phone-latency, and deployment-export pipelines.

## How To Run On Kaggle

Run the scripts in this order:

```bash
python shrimp_scripts/run_01_prepare_dataset.py --output_dir /kaggle/working/shrimp_outputs
python shrimp_scripts/run_02_train_core_ablation.py --output_dir /kaggle/working/shrimp_outputs --resume
python shrimp_scripts/run_03_train_yolo_family.py --output_dir /kaggle/working/shrimp_outputs --resume
python shrimp_scripts/run_04_train_lightweight_models.py --output_dir /kaggle/working/shrimp_outputs --resume
python shrimp_scripts/run_05_generate_reports_and_xai.py --output_dir /kaggle/working/shrimp_outputs
```

Step 1 prepares data only. Step 2 is the main paper ablation. Step 3 is YOLO family comparison. Step 4 is lightweight/mobile-friendly model comparison. Step 5 generates tables, figures, XAI, Excel, report, and zip.

All runner scripts print timestamped progress by default and append JSONL events to `progress_log.jsonl` in the selected `--output_dir`. Use `--no_progress` to silence progress bars and progress events during lightweight validation.

Stage 03 supports filtered diagnostic runs and artifact verification. For example, to run the YOLOv26m-cls CE class-mapping diagnostic in a fresh output directory:

```bash
python shrimp_scripts/run_03_train_yolo_family.py --output_dir /kaggle/working/shrimp_outputs_yolo_classmap_diagnostic --resume --progress --model yolo26m-cls --loss baseline_ce
python shrimp_scripts/run_03_train_yolo_family.py --output_dir /kaggle/working/shrimp_outputs_yolo_classmap_diagnostic --model yolo26m-cls --loss baseline_ce --verify_artifacts --progress
```

The Kaggle notebook `kaggle_step_by_step_runner.ipynb` provides the same real-stage command order with command logs written to `/kaggle/working/notebook_command_logs`.

Stage 04 also has one opt-in diagnostic extra for `convnext_tiny_in22k` with PairwiseCoInfectionRankingASL using the same Torch baseline transform recipe. It is not part of the default 51-run lightweight paper plan. Launch only that run with:

```bash
python shrimp_scripts/run_04_train_lightweight_models.py --output_dir /kaggle/working/shrimp_outputs --resume --progress --run_id lightweight_diagnostic_timm_convnext_tiny_in22k_pairwise_seed42_repeat1
```

Additional ASL-derived loss screening scripts live in `experiments/asl_custom_loss_screening/`. They are a short pre-final screening experiment and do not change the main Stage 02/03/04 experiment definitions.

## Core Ablation

Core comparison is the main experiment of the paper. It contains 6 runs:

- ConvNeXt/ShrimpXNet-style model x 3 loss conditions with the Torch baseline transform recipe
- YOLOv26m-cls model x 3 loss conditions with the YOLO-default RandAugment training recipe

The 3 loss conditions are:

1. CE
2. ASLSingleLabel
3. PairwiseCoInfectionRankingASL

This experiment tests whether the co-infection-aware loss improves classification of BG, WSSV, and WSSV_BG under each backend's intended training recipe. YOLO classification runs keep Ultralytics `auto_augment='randaugment'` when supported; Torch/TIMM runs do not use RandAugment in this workflow.

## File Guide

| File | Purpose |
|---|---|
| `shrimp_scripts/config.py` | Defines dataset settings, class names, model lists, training hyperparameters, output paths, and experiment conditions. |
| `shrimp_scripts/utils.py` | Provides reusable utilities for JSON/CSV I/O, hashing, random seeds, environment logging, resume checks, and zip creation. |
| `shrimp_scripts/dataset.py` | Downloads `uynnhy/processed-images`, verifies class counts, builds manifests, creates train/val/test split, and prepares YOLO-format folders. |
| `shrimp_scripts/losses.py` | Implements CE, ASLSingleLabel, and PairwiseCoInfectionRankingASL loss functions. |
| `shrimp_scripts/models_torch.py` | Implements ConvNeXt/ShrimpXNet-style and lightweight PyTorch/TIMM/Torchvision training and prediction. |
| `shrimp_scripts/models_yolo.py` | Implements YOLO classification training, Ultralytics `auto_augment` handling, custom loss injection, and YOLO prediction. |
| `shrimp_scripts/evaluate.py` | Computes metrics, prediction tables, classification reports, confusion matrices, and collects run outputs. |
| `shrimp_scripts/xai.py` | Generates selected Grad-CAM-style heatmaps for supported Torch and YOLO classification models. |
| `shrimp_scripts/report.py` | Builds final paper tables, figures, Excel files, reproducibility report, and downloadable zip. |
| `shrimp_scripts/progress.py` | Provides timestamped console/JSONL progress logging and optional tqdm progress bars that degrade gracefully when tqdm is unavailable. |
| `shrimp_scripts/run_01_prepare_dataset.py` | Entry point for dataset download, dataset audit, split creation, and YOLO dataset preparation. |
| `shrimp_scripts/run_02_train_core_ablation.py` | Entry point for the main core comparison: ConvNeXt and YOLOv26m across CE, ASL, and Pairwise loss under each backend's intended training recipe. |
| `shrimp_scripts/run_03_train_yolo_family.py` | Entry point for YOLO family comparison across YOLO classification m-variants. |
| `shrimp_scripts/run_04_train_lightweight_models.py` | Entry point for lightweight/mobile-friendly model comparison with resume and chunking support. |
| `shrimp_scripts/run_05_generate_reports_and_xai.py` | Entry point for final result aggregation, XAI generation, tables, figures, reports, and zip packaging. |
| `kaggle_step_by_step_runner.ipynb` | Kaggle runner notebook with real-stage switches defaulting to `True`, safe resume validation, and no default output deletion. |

## Dataset Audit

`run_01_prepare_dataset.py` fails fast unless the processed dataset contains exactly:

| Class folder | Class name | Expected images |
|---|---:|---:|
| `1. Healthy` | Healthy | 403 |
| `2. BG` | BG | 198 |
| `3. WSSV` | WSSV | 328 |
| `4. WSSV_BG` | WSSV_BG | 220 |

The expected total is 1149 images. Because the dataset is already processed, `processed_path == source_path` and `processed_md5 == source_md5`.

## Resume Rules

A run is skipped only when strict final-artifact validation passes. Required evidence includes `status.json` with `status="completed"`, current `run_audit.json` config hash, `metrics.json` with finite top-level final test metrics, `val_predictions.csv`, `test_predictions.csv`, `classification_report.csv`, `confusion_matrix.csv` or `confusion_matrix.json`, `confusion_counts.csv`, and the appropriate best checkpoint. YOLO runs also require `class_order_audit.json` with `audit_passed=true` and no swap-diagnostic failure.

If validation fails, only that run directory is archived under `runs/_archived_incomplete_runs/` and the run is launched fresh. The scripts do not delete the whole output directory during normal resume execution.

## YOLO Class-Order Audit

YOLO classification folders are copied as `00_Healthy`, `01_BG`, `02_WSSV`, and `03_WSSV_BG` to prevent alphabetical folder-order drift. Each YOLO run writes `class_order_audit.json`, records Ultralytics-discovered names and dataset order, decodes prediction indices back into the project class order, and runs a Healthy/BG swap diagnostic. If swapping predicted indices 0 and 1 improves test Macro-F1 by more than 0.20, the run fails instead of reporting suspicious metrics.

## Validation

```bash
python -m compileall shrimp_scripts
python shrimp_scripts/run_01_prepare_dataset.py --dry_run
python shrimp_scripts/run_02_train_core_ablation.py --list_runs
python shrimp_scripts/run_03_train_yolo_family.py --list_runs
python shrimp_scripts/run_04_train_lightweight_models.py --list_runs
python shrimp_scripts/run_05_generate_reports_and_xai.py --dry_run
python shrimp_scripts/run_02_train_core_ablation.py --output_dir /kaggle/working/shrimp_outputs --validate_resume
python shrimp_scripts/run_03_train_yolo_family.py --output_dir /kaggle/working/shrimp_outputs --validate_resume
python shrimp_scripts/run_04_train_lightweight_models.py --output_dir /kaggle/working/shrimp_outputs --validate_resume
python experiments/asl_custom_loss_screening/run_asl_custom_screen.py --output_dir /kaggle/working/shrimp_outputs_asl_custom_screening --validate_resume
python -c "import shrimp_scripts.config, shrimp_scripts.utils, shrimp_scripts.dataset, shrimp_scripts.losses, shrimp_scripts.evaluate, shrimp_scripts.models_torch, shrimp_scripts.models_yolo, shrimp_scripts.xai, shrimp_scripts.report"
```

## Notes On Claims

ASLSingleLabel is an existing ASL baseline, not a custom contribution. PairwiseCoInfectionRankingASL should be described as a dataset-specific co-infection-aware ASL variant unless stronger results and literature review justify broader claims.
