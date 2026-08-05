# Reproduction and verification

## Reproduction contract

The repository fixes seed 42, image size 224, class order, and an image-level
split. Reproduction has two distinct goals:

1. rerun training/evaluation from the SDI-4 dataset; and
2. verify the identity and output shape of released EXT-3 checkpoints.

The official SDI-4 checkpoint binaries are unresolved, so no command below is
claimed to load the official-final 0.910137 checkpoint.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/00_check_env.py
```

The recorded training runtime was Kaggle T4×2 with Python 3.12.3. Evaluation
of the rejected `4305e491...38c1` checkpoint was repeated with Ultralytics
8.4.72 and 8.4.103 and produced identical predictions.

## Prepare the fixed SDI-4 split

```bash
python scripts/01_prepare_dataset_and_split.py --seed 42
```

The script downloads Kaggle dataset `uynnhy/processed-images` unless
`--data-root` is supplied, validates total class counts 403/198/328/220, and
creates train/validation/test sizes 804/172/173. It retains the manifest at
`artifacts/manifests/split_manifest_seed42.csv`.

## Train

```bash
# CE
python scripts/02_train_yolo26m_ce_baseline.py --device 0 --skip-if-complete

# ASL-LDAM + SimAM-DCFR
python scripts/03_train_yolo26m_asl_ldam_simam_dcfr.py --device 0 --skip-if-complete
```

Each run uses a separate name and writes `status.json`. Inspect the checkpoint's
stored run name, classes, and architecture after training; do not promote a file
based on its path alone.

## Evaluate SDI-4 clean test

```bash
python scripts/04_eval_clean.py \
  --weights runs/training/<run>/weights/best.pt \
  --manifest artifacts/manifests/split_manifest_seed42.csv \
  --output-dir artifacts/evaluation/<run>/clean
```

The release audit declared an absolute tolerance of `1e-6` before evaluation.
For official-final reproduction, all of the following must hold:

- exactly 173 test images;
- class order Healthy, BG, WSSV, WSSV_BG;
- proposed architecture includes the late SimAM-DCFR wrapper;
- Accuracy equals 0.913295 within tolerance;
- Macro-F1 equals 0.910137 within tolerance.

The evaluator saves predictions, metrics/classification report, count and
row-normalized confusion matrices, command, environment, and checkpoint hash.

## Evaluate controlled corruptions

```bash
python scripts/05_eval_noise_top5.py \
  --weights runs/training/<run>/weights/best.pt
```

The five corruptions are impulse noise, Gaussian noise, contrast reduction,
defocus blur, and low light at severities 1–3. Existing retained corruption
tables belong to a historical 0.905448 clean package, not the unresolved
official-final checkpoint.

## Verify released EXT-3 files

```powershell
git lfs pull --include="weights/ext3_original/*.pt"
Get-FileHash -Algorithm SHA256 weights\ext3_original\*.pt
git lfs fsck
```

Expected hashes are recorded in `weights/SHA256SUMS.txt`. For trusted project
checkpoints, register repository safe globals before loading:

```python
from cvio_asl_ldam.attention.patch_yolo import register_checkpoint_safe_globals

register_checkpoint_safe_globals()
```

Do not load untrusted third-party pickle files. Both released EXT-3 models must
produce shape `(1, 3)` for an input `(1, 3, 224, 224)` and use class order
Healthy/BG/WSSV.

## Regenerate documentation figures

```bash
python scripts/docs/generate_release_figures.py
```

The script reads retained CSV files, uses fixed styling, and rewrites the four
quantitative PNGs and four SVG diagrams under `docs/assets/`.

## Export a newly trained, verified checkpoint

```bash
python scripts/06_export_litert_fp32_fp16.py \
  --weights runs/training/<verified-run>/weights/best.pt \
  --out-dir export --imgsz 224
```

Export only after binding the source checkpoint hash and verifying architecture,
class order, and clean metrics. The currently tracked TFLite files do not pass
the official-proposed source-binding gate; see [EXPORT_LITERT.md](EXPORT_LITERT.md).

## Local QA

```bash
python -m pytest tests -q
python -m compileall -q src scripts
python scripts/check_readme_links.py
git lfs fsck
git diff --check
```

Full training is intentionally not part of the lightweight QA suite.
