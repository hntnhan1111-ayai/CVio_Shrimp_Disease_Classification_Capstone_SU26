# ASL-LDAM: A Class-Imbalance-Aware Loss for Robust Shrimp Disease Image Classification Under Noisy Imaging Conditions

**ASL-LDAM with SimAM-DCFR Attention for Robust YOLO-Based Shrimp Disease Image Classification Under Noisy Imaging Conditions**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Seed](https://img.shields.io/badge/Seed-42-orange)
![Task](https://img.shields.io/badge/Task-Shrimp%20Disease%20Classification-purple)
![Status](https://img.shields.io/badge/Status-Paper%20Artifacts-informational)

> [!IMPORTANT]
> This repository contains **fixed seed-42 results only**. It does not claim
> mean ± standard deviation across seeds or statistical significance. The split
> is image-level and is not asserted to be animal/specimen-safe. This code is
> for research reproducibility, not veterinary diagnosis.

> [!WARNING]
> No raw dataset, corrupted image folders, or trained `.pt` checkpoints are
> included. The two reviewer-facing TFLite files under `export/` are committed.

## Overview

Four-class shrimp disease image classification:

1. `Healthy`
2. `BG`
3. `WSSV`
4. `WSSV_BG`

Main method: **YOLO26m-cls + ASL-LDAM + SimAM-DCFR**, evaluated on a fixed
seed-42 image-level split and under the top-5 imaging corruptions.

## What is included / not included

**Included**

- Importable `src/cvio_asl_ldam` package (ASL-LDAM loss, SimAM-DCFR attention,
  dataset audit/split, metrics, corruptions, export sanity check)
- Python scripts `00?07` for the full reviewer workflow
- Configs (`configs/`), split manifest, result tables, figures, XAI panels
- Two TFLite files in `export/` (FP32 ~40 MB, FP16 ~20 MB)
- Docs and lightweight tests

**Not included**

- Raw dataset (download from Kaggle `uynnhy/processed-images`)
- Trained `.pt`/`.onnx`/SavedModel weights
- Training run folders, caches, corrupted image folders

## Dataset

- Kaggle dataset: `uynnhy/processed-images` (already background-removed with U2Net/rembg)
- Class counts: Healthy 403, BG 198, WSSV 328, WSSV_BG 220 (total 1,149)
- Fixed seed-42 Stage-1 **image-level** split:

| Split | Healthy | BG | WSSV | WSSV_BG | Total |
|---|---:|---:|---:|---:|---:|
| train | 282 | 139 | 229 | 154 | 804 |
| val | 60 | 30 | 49 | 33 | 172 |
| test | 61 | 29 | 50 | 33 | 173 |

The default workflow does **not** rerun background removal. Details: [DATA.md](docs/DATA.md).

## Environment

- Paper runtime: Kaggle T4x2 GPU, Python 3.12.3
- Export runtime: **Python 3.12.2** with pinned versions (see [EXPORT_LITERT.md](docs/EXPORT_LITERT.md))
- Record your environment:

```bash
python scripts/00_check_env.py
```

Install training dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick start: reproduce split only

```bash
python scripts/01_prepare_dataset_and_split.py --seed 42
```

This downloads via KaggleHub (or uses `--data-root`), robustly detects the
class root, validates counts (403/198/328/220), creates the seed-42 split
(804/172/173), materializes `runs/prepared_seed42/{train,val,test}`, and writes
`artifacts/manifests/split_manifest_seed42.csv` plus
`artifacts/metadata/dataset_audit.json`.

## Train main method

```bash
python scripts/03_train_yolo26m_asl_ldam_simam_dcfr.py --device 0 --skip-if-complete
```

Run-overwrite guards: `--skip-if-complete` (default), `--force`, `--resume`,
`--run-name`. Each run writes `status.json`. Full training was **not** rerun
for this refactor; use the provided TFLite files or train fresh.

Reported main-method setting: seed=42, imgsz=224, epochs=30, patience=15,
batch=32, optimizer=AdamW, lr0=0.00125, cos_lr=True,
auto_augment=randaugment, erasing=0.4, workers=4. ASL-LDAM hyperparameters:
gamma_pos=0.0, gamma_neg=4.0, label_smoothing=0.1, LDAM max margin=0.5,
LDAM scale=30.0.

## Optional: train CE baseline

```bash
python scripts/02_train_yolo26m_ce_baseline.py --device 0 --skip-if-complete
```

## Evaluate clean test

```bash
python scripts/04_eval_clean.py \
  --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt
```

## Evaluate top-5 noise robustness

```bash
python scripts/05_eval_noise_top5.py \
  --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt
```

Corruptions (severities 1?3): `impulse_noise`, `gaussian_noise`,
`contrast_reduction`, `defocus_blur`, `low_light`. Corruptions are generated
in memory; no corrupted image folders are saved.

## Export LiteRT/TFLite

Use a clean Python 3.12.2 venv (3.13/base caused TensorFlow/tf-keras problems):

```bash
python -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -r requirements-export-litert.txt

python scripts/06_export_litert_fp32_fp16.py \
  --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt \
  --out-dir export \
  --imgsz 224
```

The export script never trains. See [EXPORT_LITERT.md](docs/EXPORT_LITERT.md).

## Use provided TFLite files

```
export/yolo26m_asl_ldam_simam_dcfr_fp32.tflite
export/yolo26m_asl_ldam_simam_dcfr_fp16.tflite
```

- **FP32** (~40 MB) is safest for baseline deployment testing.
- **FP16** (~20 MB) is smaller and may be faster on compatible GPU/mobile delegates.
- Input: RGB, 224?224, NHWC `[1,224,224,3]`, `float32`.
- Output: `[1,4]` logits; apply softmax for probabilities.
- Class order: `0: Healthy`, `1: BG`, `2: WSSV`, `3: WSSV_BG`.
- FP16 keeps internal weights as float16; input/output may remain float32.

Sanity check:

```bash
python - <<'PY'
import sys; sys.path.insert(0, "src")
from cvio_asl_ldam.export import tflite_sanity_check
print(tflite_sanity_check("export/yolo26m_asl_ldam_simam_dcfr_fp32.tflite"))
PY
```

See [export/README.md](export/README.md).

## Results

Main result (fixed seed-42):

| Metric | Baseline CE | Best: ASL-LDAM + SimAM-DCFR | Delta |
|---|---:|---:|---:|
| Macro-F1 | 0.890200 | 0.910137 | +0.019937 |
| Accuracy | 0.890200 | 0.913295 | +0.023095 |
| Cohen's Kappa | 0.850500 | 0.881593 | +0.031093 |

Source: [key result CSV](artifacts/tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv).
Full baseline and noise tables: [RESULTS.md](docs/RESULTS.md).

## Troubleshooting

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Common issues: export env
mismatches, dataset-root detection, run-overwrite guards, CUDA OOM.

## Claims and limitations

- This repository contains fixed seed-42 results only. It does not claim mean ± standard deviation across seeds or statistical significance.
- Results are seed-42-only. No multi-seed mean ± std or statistical significance is claimed.
- The split is image-level and is not asserted to be animal/specimen-safe.
- Training results may vary slightly due to GPU/library nondeterminism.
- Exported TFLite files are provided for reviewer convenience.
- Raw dataset and model weights are not included (TFLite files are committed).
- This is research code, not a veterinary diagnostic tool.

Full statements: [CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md),
[MODEL_CARD.md](docs/MODEL_CARD.md).

## Repository structure

```text
configs/      dataset, training, noise, XAI configuration
src/          importable ASL-LDAM research package (losses, attention, data, export, utils)
scripts/      00-07 Python entrypoints + run_*.sh wrappers
export/       reviewer-facing TFLite files + sanity checks
artifacts/    manifests, tables, figures, XAI, metadata
docs/         REPRODUCE, DATA, EXPORT_LITERT, CLAIMS_AND_LIMITATIONS, MODEL_CARD, TROUBLESHOOTING
tests/        lightweight tests (no full dataset required)
```

## Citation / license / contact

```bibtex
@software{nguyen2026asl_ldam_shrimp,
  title = {ASL-LDAM with SimAM-DCFR Attention for Robust YOLO-Based Shrimp Disease Image Classification Under Noisy Imaging Conditions},
  author = {Nguyen, Vinh Dinh and Nguyen, Phong Van and Tran, Nhan Huu and Le Thi, Nhu Huynh},
  year = {2026},
  version = {0.1.0-paper-asl-ldam},
  url = {https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26}
}
```

Code: AGPL-3.0-or-later. Dataset not redistributed. Use the issue tracker for questions.
