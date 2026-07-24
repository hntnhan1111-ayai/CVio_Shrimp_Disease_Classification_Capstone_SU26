# ShrimpDB Combined Multi-Source Classification

Nguyen, Vinh Dinh; Nguyen, Phong Van; Tran, Nhan Huu; Le Thi, Nhu Huynh.

[![Python](https://img.shields.io/badge/Python-3.12.13-blue)](#installation) [![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0%2Bcu128-green)](#runtime) [![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.75-red)](#runtime) [![Kaggle](https://img.shields.io/badge/Runtime-Kaggle-20beff)](#kaggle-reproduction) [![uv](https://img.shields.io/badge/managed%20by-uv-6f42c1)](#installation)

Research repository for fixed-seed YOLO26m-cls experiments on ShrimpDB and
ShrimpDiseaseDB. The accompanying report and citation metadata are maintained at the
[repository root](../../README.md) and in [CITATION.cff](../../CITATION.cff).

## Abstract

We evaluate a CE baseline and ASL-LDAM with SimAM-DCFR attention for shrimp disease image
classification. ShrimpDB is harmonized to three classes and combined with ShrimpDiseaseDB
using source-wise stratified partitions. The reported execution uses seed 42 and a
70%/15%/15% train/validation/test split. A merged package selects the highest Macro-F1
result independently for each dataset, with accuracy as tie-breaker.

The merged archive uses a common display label while retaining the verified source method.
Consequently, the selected ShrimpDB-3 result is ASL-LDAM + SimAM-DCFR, while the selected
Combined-4 result is from the CE baseline archive. This distinction is preserved in
`artifacts/results/FINAL_BEST_RESULTS.json` and in the main table.

## Highlights

- ShrimpDB-3 selected result: 91.49% accuracy and 91.29% Macro-F1.
- Combined-4 selected result: 87.73% accuracy and 86.51% Macro-F1.
- Results are validation-selected, fixed seed-42 observations, not multi-seed estimates.

## Main results

Generated from `artifacts/results/FINAL_BEST_RESULTS.json` and the selected raw metrics:

| Dataset | Display label | Actual source method | Accuracy | Balanced Accuracy | Macro-F1 | ECE |
|---|---|---|---:|---:|---:|---:|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | ASL-LDAM + SimAM-DCFR | 91.49% | 91.55% | 91.29% | 19.30% |
| Combined-4 | ASL-LDAM + SimAM-DCFR | CE Baseline | 87.73% | 87.43% | 86.51% | 6.19% |

The common display label is not a method claim for Combined-4. Its selected checkpoint is
CE Baseline and `label_matches_source_method` is `false`.

## Datasets and split

- [ShrimpDB](https://www.kaggle.com/datasets/vohoangtu/shrimpdb)
- [ShrimpDiseaseDB / processed-images](https://www.kaggle.com/datasets/uynnhy/processed-images)

ShrimpDB labels are harmonized as `Tom_BT -> Healthy`, `Den_Mang -> BG`, and
`Dom_Trang -> WSSV`. The split target is 70% train, 15% validation, and 15% test with
`seed = 42`. Each source is partitioned before combination:

`Combined train = ShrimpDB train + ShrimpDiseaseDB train`
`Combined validation = ShrimpDB validation + ShrimpDiseaseDB validation`
`Combined test = ShrimpDB test + ShrimpDiseaseDB test`

| Dataset / split | Healthy | BG | WSSV | WSSV_BG | Total |
|---|---:|---:|---:|---:|---:|
| ShrimpDB-3 train | 49 | 78 | 94 | — | 221 |
| ShrimpDB-3 validation | 10 | 17 | 20 | — | 47 |
| ShrimpDB-3 test | 11 | 16 | 20 | — | 47 |
| Combined-4 train | 331 | 217 | 323 | 154 | 1,025 |
| Combined-4 validation | 70 | 47 | 69 | 33 | 219 |
| Combined-4 test | 72 | 45 | 70 | 33 | 220 |

The machine-readable version is [`dataset_split_counts.csv`](artifacts/tables/dataset_split_counts.csv).

## Methodology

Both experiments use YOLO26m-cls, classification task, 224px input, 30 epochs, patience 15,
batch 32, four workers, AMP, AdamW, `lr0=0.00125`, `lrf=0.01`, cosine learning rate,
RandAugment, erasing 0.4, deterministic seed 42, and no cache.

| Method | Loss | Attention |
|---|---|---|
| CE baseline | CrossEntropyLoss | None |
| ASL-LDAM + SimAM-DCFR | ASL-LDAM (`gamma_pos=0`, `gamma_neg=4`, `label_smoothing=0.1`, `ldam_max_m=0.5`, `ldam_scale=30`) | SimAM-DCFR (`e_lambda=0.0001`) |

See [METHODOLOGY.md](docs/METHODOLOGY.md) and [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Runtime

The reported run used a Kaggle Notebook with 2 × Tesla T4 GPUs, global batch 32, and seed 42.

| Component | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| TorchVision | 0.25.0+cu128 |
| CUDA | 12.8 |
| Ultralytics | 8.4.75 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.10.0 |
| Pillow | 11.3.0 |
| PyYAML | 6.0.3 |

## Quantitative and qualitative artifacts

Tables are in [`artifacts/tables/main_results_percent.csv`](artifacts/tables/main_results_percent.csv)
and the neighboring table files: main results, method comparison,
per-class results, split counts, source-domain metrics, and checkpoint summary. Imported raw
evaluation files are in `artifacts/results/`, including predictions,
confusion-matrix CSVs/PNGs, and provenance JSON files.

Figures are linked directly from the repository:

| Figure | Description |
|---|---|
| [fig01_dataset_split_distribution.png](artifacts/figures/fig01_dataset_split_distribution.png) | Dataset split distribution |
| [fig02_selected_accuracy_by_dataset.png](artifacts/figures/fig02_selected_accuracy_by_dataset.png) | Selected accuracy |
| [fig03_selected_macro_f1_by_dataset.png](artifacts/figures/fig03_selected_macro_f1_by_dataset.png) | Selected Macro-F1 |
| [fig04_all_methods_shrimpdb3_comparison.png](artifacts/figures/fig04_all_methods_shrimpdb3_comparison.png) | ShrimpDB-3 method comparison |
| [fig05_all_methods_combined4_comparison.png](artifacts/figures/fig05_all_methods_combined4_comparison.png) | Combined-4 method comparison |
| [fig06_shrimpdb3_confusion_matrix_counts.png](artifacts/figures/fig06_shrimpdb3_confusion_matrix_counts.png) | ShrimpDB-3 count confusion matrix |
| [fig07_shrimpdb3_confusion_matrix_normalized.png](artifacts/figures/fig07_shrimpdb3_confusion_matrix_normalized.png) | ShrimpDB-3 normalized confusion matrix |
| [fig08_combined4_confusion_matrix_counts.png](artifacts/figures/fig08_combined4_confusion_matrix_counts.png) | Combined-4 count confusion matrix |
| [fig09_combined4_confusion_matrix_normalized.png](artifacts/figures/fig09_combined4_confusion_matrix_normalized.png) | Combined-4 normalized confusion matrix |
| [fig10_combined4_source_domain_comparison.png](artifacts/figures/fig10_combined4_source_domain_comparison.png) | Source-domain comparison |
| [fig11_checkpoint_provenance_diagram.png](artifacts/figures/fig11_checkpoint_provenance_diagram.png) | Checkpoint/source mapping |
| [fig02_shrimpdb3_training_curves.png](artifacts/figures/fig02_shrimpdb3_training_curves.png) | ShrimpDB-3 training curves |
| [fig03_combined4_training_curves.png](artifacts/figures/fig03_combined4_training_curves.png) | Combined-4 training curves |

The standalone HTML report is [`artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html`](artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html).

## Installation

```powershell
Set-Location "D:\CVio\CVio_ShrimpDB_Combined_Research_Repo\studies\shrimpdb_combined_multisource"
uv python install 3.12
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv sync --frozen
```

## Reproduction

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
attached. The reported execution used 2 × Tesla T4.

## Checkpoints

Checkpoint files are not tracked. The selected checkpoints are documented in
[`checkpoint_summary.csv`](artifacts/tables/checkpoint_summary.csv):

- ShrimpDB-3, ASL-LDAM + SimAM-DCFR: `ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36`.
- Combined-4, CE Baseline: `adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad`.

Retrieve a checkpoint from the reviewed result package, verify its SHA-256, and use the
class order in the checkpoint summary. Do not select a checkpoint by display label alone.

## Limitations

These are one-run, fixed seed-42 results. They do not establish statistical significance,
state-of-the-art performance, universal superiority, clinical validity, or production
readiness. The split is image-level and is not claimed to be specimen-safe. Dataset shift,
label harmonization, calibration, and external validity remain limitations.

## Citation and license

```bibtex
@software{nguyen2026shrimpdb_combined,
  title = {ShrimpDB Combined Multi-Source Classification},
  author = {Nguyen, Vinh Dinh and Nguyen, Phong Van and Tran, Nhan Huu and Le Thi, Nhu Huynh},
  year = {2026},
  url = {https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26}
}
```

Code is licensed under [AGPL-3.0-or-later](../../LICENSE). Datasets and checkpoints are not redistributed.
