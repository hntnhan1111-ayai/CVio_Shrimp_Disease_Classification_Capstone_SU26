# CVio YOLO11s-RECSRA: Robust Shrimp Disease Detection

> **RECSRA** — Robust Energy-Guided Circular–Spatial Residual Attention

This publication-oriented repository detects **BG** and **WSSV** manifestations from images of individual shrimp photographed onshore with consumer mobile phones. It contains the frozen method, exact checkpoints, clean-test evidence, a ten-family mobile-camera corruption benchmark, raw metrics, percentage-formatted tables, paper figures, and reproducibility scripts.

![Clean comparison](results/figures/paper/fig01_clean_test_comparison.png)

## Clean test results

The matched checkpoint pair uses the same test protocol: `imgsz=1536`, `IoU=0.55`, `conf=0.0005`, `max_det=600`, batch size `1`, and FP32.

| Metric | YOLO11s baseline | YOLO11s-RECSRA | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| mAP50 | 12.900% | **16.036%** | **+3.136 pp** | **+24.31%** |
| mAP50-95 | 3.800% | **4.657%** | **+0.857 pp** | **+22.54%** |
| Precision | 22.400% | **24.342%** | +1.942 pp | +8.67% |
| Recall | 21.800% | 24.695% | +2.895 pp | +13.28% |
| BG AP50-95 | 4.000% | **5.373%** | +1.373 pp | +34.33% |
| WSSV AP50-95 | 3.500% | **3.940%** | +0.440 pp | +12.56% |

Raw files retain decimal metrics for numerical processing. Public tables use percentages; direct differences use percentage points (`pp`).

## Top-five mobile corruption results

![Top-five severity consistency](results/figures/paper/fig05_top5_gain_by_severity.png)

| Rank | Corruption | Baseline mean mAP50-95 | RECSRA mean mAP50-95 | Gain | Relative gain | Wins |
|---|---|---|---|---|---|---|
| 1 | N07 — jpeg recompression | 3.600% | 4.132% | +0.532 pp | +14.78% | 5/5 |
| 2 | N06 — auto white balance color cast | 3.725% | 4.211% | +0.485 pp | +13.03% | 5/5 |
| 3 | N01 — low light sensor noise | 0.906% | 1.385% | +0.479 pp | +52.86% | 5/5 |
| 4 | N03 — localized specular flash glare | 3.975% | 4.440% | +0.464 pp | +11.68% | 5/5 |
| 5 | N10 — lens smudge fingerprint | 3.385% | 3.786% | +0.401 pp | +11.86% | 5/5 |

Each selected family has positive RECSRA mAP50 and mAP50-95 differences at all five severity levels. Ranking uses mean mAP50-95 gain across severities 1–5.

## RECSRA design

RECSRA combines RMS energy-based channel description, circular local channel interaction, grouped horizontal–vertical spatial evidence, boundary-density evidence, and bounded residual feature modulation. It replaces C3k2 stages at backbone layers `4`, `6`, and `8`.

The winning configuration uses channel kernel `5`, spatial kernel `5`, `8` groups, residual strength `0.15`, temperature `1.0`, and boundary weight `0.1`.

The historical names `SLDRA` and `C3k2SLDRA` are retained only for serialized checkpoint compatibility. Academic names are `RECSRA` and `C3k2RECSRA`.

## Repository map

```text
configs/          Model, data, train, evaluation, and corruption configurations
src/recsra/       RECSRA implementation and Ultralytics compatibility layer
scripts/          Setup, training, evaluation, robustness, and verification commands
checkpoints/      Locked baseline, best, and final-epoch checkpoints
results/          Raw metrics, percentage tables, figures, logs, and samples
paper/            LaTeX-ready tables and publication asset index
reproducibility/  Dataset fingerprints, provenance, and integrity reports
environment/      Exact package freeze plus actual and target hardware profiles
archive/          Original experiment scripts preserved for traceability
```

## Verify the package

```bash
git lfs install
PYTHONPATH=src python scripts/verify_repository.py
sha256sum -c checkpoints/SHA256SUMS.txt
```

## Install

```bash
bash scripts/setup_environment.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

## Clean evaluation

```bash
PYTHONPATH=src python scripts/evaluate_clean.py   --data /absolute/path/to/data.yaml   --device 0
```

## Reproduce RECSRA training

```bash
PYTHONPATH=src python scripts/train_recsra.py   --data /absolute/path/to/data.yaml   --device 0
```

The frozen pretrained flow is:

```text
YOLO(model YAML)
→ model.load(baseline checkpoint)
→ direct transfer + C3k2 wrapper remapping
→ trainer receives 529/529 prepared tensors
→ 200 epochs
→ validation/test evaluation at 1536 pixels
```

## Reproduce mobile robustness

Current artifacts were evaluated on an RTX 4090:

```bash
export DATASET_ROOT=/absolute/path/to/dataset
bash scripts/reproduce_rtx4090.sh
```

A future Kaggle T4×2 target is supplied:

```bash
export DATASET_ROOT=/kaggle/input/<dataset>
bash scripts/reproduce_kaggle_t4x2_target.sh
```

The T4×2 script verifies actual T4 hardware and records runtime metadata. Existing RTX 4090 artifacts must not be relabeled as T4×2 results.

## Dataset

The dataset is not redistributed. Frozen counts are:

- Train: **523 images**
- Validation: **112 images**
- Test: **111 images**
- Classes: **BG** and **WSSV**

Use `configs/data/data.template.yaml` and the SHA-256 dataset manifest.

## Limitations

- One reported training seed (`42`).
- Corruption severities are ordered stress levels, not independent statistical replicates.
- Top-five qualitative figures use one shared WSSV source image for controlled illustration.
- Clean recall is slightly below the matched baseline even though mAP and precision improve.
- Standalone validation/test plot directories were absent from the source evidence; numeric metrics and regeneration scripts are included.
- Mobile-device latency, energy, and thermal behavior remain to be measured.

See `docs/LIMITATIONS.md`.

## Recommended branch

`paper/yolo11s-recsra-mobile-robustness`

## Citation and licensing

Use `CITATION.cff`. Review `LICENSE_NOTICE.md` and `THIRD_PARTY_NOTICES.md` before public release.
