# Reproduce the Seed-42 Paper Artifacts

This is the reviewer-facing reproduction workflow. It uses the numbered Python
scripts under `scripts/`. Legacy `.sh` wrappers remain for back-compatibility
but the Python scripts are the supported entry points.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python scripts/00_check_env.py     # writes artifacts/metadata/env_report.json
```

Paper runtime: Kaggle T4x2 GPU, Python 3.12.3. Export runtime: Python 3.12.2
(see [EXPORT_LITERT.md](EXPORT_LITERT.md)).

## 1. Prepare the fixed seed-42 split

```bash
python scripts/01_prepare_dataset_and_split.py --seed 42
```

Downloads via KaggleHub (`uynnhy/processed-images`) unless `--data-root` is
given, robustly detects the class root, validates counts (403/198/328/220),
and creates the split (804/172/173). Outputs:

- `artifacts/manifests/split_manifest_seed42.csv`
- `artifacts/metadata/dataset_audit.json`
- `runs/prepared_seed42/{train,val,test}`

Skips if already complete (use `--force` to re-run).

## 2. Train the main method

```bash
python scripts/03_train_yolo26m_asl_ldam_simam_dcfr.py --device 0 --skip-if-complete
```

Run-overwrite guards: `--skip-if-complete` (default), `--force`, `--resume`,
`--run-name`. Writes `status.json` per run.

## 3. (Optional) Train the CE baseline

```bash
python scripts/02_train_yolo26m_ce_baseline.py --device 0 --skip-if-complete
```

## 4. Evaluate the clean test set

```bash
python scripts/04_eval_clean.py \
  --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt
```

## 5. Evaluate top-5 noise robustness

```bash
python scripts/05_eval_noise_top5.py \
  --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt
```

Corruptions: `impulse_noise`, `gaussian_noise`, `contrast_reduction`,
`defocus_blur`, `low_light` at severities 1?3. Generated in memory.

## 6. Export LiteRT/TFLite

Use a clean Python 3.12.2 venv:

```bash
python -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -r requirements-export-litert.txt

python scripts/06_export_litert_fp32_fp16.py \
  --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt \
  --out-dir export --imgsz 224
```

The export script never trains. See [EXPORT_LITERT.md](EXPORT_LITERT.md).

## 7. Package reviewer artifacts

```bash
python scripts/07_package_review_artifacts.py
```

## Smoke tests

```bash
bash scripts/run_smoke_tests.sh
# or
PYTHONPATH=src python -m pytest tests -q
```

## Expected outputs

- `runs/training/*/weights/best.pt`
- `artifacts/evaluation/clean/clean_test_metrics.json`
- `artifacts/evaluation/noise/top5_noise_by_severity.csv`
- `export/yolo26m_asl_ldam_simam_dcfr_fp32.tflite`
- `export/yolo26m_asl_ldam_simam_dcfr_fp16.tflite`
- `export/tflite_sanity_check.json`

## Notes

- Full training was **not** rerun for this refactor; use the provided TFLite
  files or train fresh.
- Results may vary slightly with GPU/library nondeterminism.
- Legacy `.sh` scripts (`01_make_seed42_split.sh`, `02_run_yolo_baselines.sh`,
  etc.) remain for back-compatibility but are superseded by the Python scripts.
