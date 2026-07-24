# ShrimpDB Combined Multi-Source Benchmark

[![Python](https://img.shields.io/badge/Python-3.12.13-blue)](#verified-runtime) [![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0%2Bcu128-green)](#verified-runtime) [![TorchVision](https://img.shields.io/badge/TorchVision-0.25.0%2Bcu128-green)](#verified-runtime) [![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.75-red)](#verified-runtime) [![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900)](#verified-runtime) [![Kaggle](https://img.shields.io/badge/Runtime-Kaggle-20beff)](#kaggle-reproduction) [![uv](https://img.shields.io/badge/managed%20by-uv-6f42c1)](#reproduction) [![pandas](https://img.shields.io/badge/pandas-2.3.3-150458)](#verified-runtime) [![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6.1-f7931e)](#verified-runtime) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.0-11557c)](#verified-runtime) [![OpenCV](https://img.shields.io/badge/OpenCV-research-5c3ee8)](#reproduction) [![GitHub Actions](https://img.shields.io/github/actions/workflow/status/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26/shrimpdb-combined-ci.yml?branch=paper%2Fshrimpdb-combined-asl-ldam-simam-dcfr&label=CI)](../../actions)

## Abstract

This study evaluates YOLO26m-cls for shrimp disease image classification under a fixed
seed-42, source-wise 70/15/15 split protocol. The repository preserves two method families:
CrossEntropyLoss (CE baseline, no custom attention) and ASL-LDAM with SimAM-DCFR attention.
The newly integrated merged package selects the highest Macro-F1 result independently for
each dataset, with accuracy as the tie-breaker.

The merged package uses the common display label **ASL-LDAM + SimAM-DCFR** while retaining
the verified `actual_source_method` and checkpoint SHA-256. Therefore, the selected
ShrimpDB-3 result is ASL-LDAM + SimAM-DCFR, whereas the selected Combined-4 result is from
the CE baseline archive. This distinction is intentional and is enforced by the registries
and tests.

## Research questions and contributions

The study asks how the two training methods behave on ShrimpDB-3 and Combined-4, how
source-wise partitioning affects reproducibility, and whether a merged best-by-dataset
selection can be presented without obscuring method provenance. Contributions include
source-wise stratified splitting, label harmonization, executable method-identity auditing,
provenance-preserving result integration, machine-readable tables, and checkpoint hash
records. Results are fixed seed-42 observations; they do not establish statistical
significance or universal superiority.

## Repository structure

`configs/` contains dataset and training configuration; `src/` contains reusable
training/evaluation code; `scripts/` contains audit, split, training, evaluation, reporting,
and packaging entry points; `artifacts/` contains generated evidence; `artifacts/merged_best_by_dataset/`
contains only selected evaluation files and provenance; `model_registry/` records external
checkpoint metadata; `docs/` contains academic documentation; `tests/` validates registries,
tables, figures, links, and repository hygiene.

## Datasets and label harmonization

- [ShrimpDB](https://www.kaggle.com/datasets/vohoangtu/shrimpdb)
- [ShrimpDiseaseDB / processed-images](https://www.kaggle.com/datasets/uynnhy/processed-images)

For ShrimpDB, `Tom_BT -> Healthy`, `Den_Mang -> BG`, and `Dom_Trang -> WSSV`.
ShrimpDiseaseDB contributes `Healthy`, `BG`, `WSSV`, and `WSSV_BG`.

## Dataset audit and split protocol

The target split is 70% train, 15% validation, and 15% test with `seed = 42`. Each source
is partitioned independently before combination:

`Combined train = ShrimpDB train + ShrimpDiseaseDB train`
`Combined val = ShrimpDB val + ShrimpDiseaseDB val`
`Combined test = ShrimpDB test + ShrimpDiseaseDB test`

| Dataset / split | Healthy | BG | WSSV | WSSV_BG | Total |
|---|---:|---:|---:|---:|---:|
| ShrimpDB-3 train | 49 | 78 | 94 | — | 221 |
| ShrimpDB-3 validation | 10 | 17 | 20 | — | 47 |
| ShrimpDB-3 test | 11 | 16 | 20 | — | 47 |
| Combined-4 train | 331 | 217 | 323 | 154 | 1,025 |
| Combined-4 validation | 70 | 47 | 69 | 33 | 219 |
| Combined-4 test | 72 | 45 | 70 | 33 | 220 |

## Methods and shared training configuration

Both methods use `yolo26m-cls`, task `classify`, image size 224, 30 epochs, seed 42,
deterministic execution, patience 15, batch 32, workers 4, AMP, AdamW, `lr0=0.00125`,
`lrf=0.01`, cosine learning rate, no cache, RandAugment, erasing 0.4, and plots enabled.

| Method | Loss | Attention |
|---|---|---|
| CE baseline | CrossEntropyLoss | none |
| ASL-LDAM + SimAM-DCFR | ASL-LDAM (`gamma_pos=0`, `gamma_neg=4`, `label_smoothing=0.1`, `ldam_max_m=0.5`, `ldam_scale=30`) | SimAM-DCFR (`e_lambda=0.0001`) |

The executable identity audit and historical correction trail are in
`artifacts/metadata/method_identity_audit.{json,md}` and
`artifacts/metadata/label_correction_manifest.json`. The merged package adds a separate
verified registry at `artifacts/metadata/merged_result_registry.json`.

## Merged best-by-dataset selection

Selection is the highest Macro-F1 within each dataset, with accuracy as the tie-breaker.
The table below is generated from `FINAL_BEST_RESULTS.json`; raw decimal metrics remain in
the JSON evaluation artifacts.

| Dataset | Display label | Actual source method | Accuracy | Macro-F1 | Label/source match |
|---|---|---|---:|---:|---|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | ASL-LDAM + SimAM-DCFR | 91.49% | 91.29% | Yes |
| Combined-4 | ASL-LDAM + SimAM-DCFR | CE Baseline | 87.73% | 86.51% | No |

The merged package applies a common display label to the selected best-by-dataset results
while retaining the verified source method and checkpoint SHA-256 for each result. The
Combined-4 selected metrics originate from the CE baseline run; the ShrimpDB-3 selected
metrics originate from the ASL-LDAM + SimAM-DCFR run.

### Selected benchmark metrics

| Dataset / source method | Accuracy | Balanced accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|
| ShrimpDB-3 / ASL-LDAM + SimAM-DCFR | 91.49% | 91.55% | 91.29% | 19.30% |
| Combined-4 / CE Baseline | 87.73% | 87.43% | 86.51% | 6.19% |

Full method comparisons, including the non-selected result for each dataset, are generated
in `artifacts/tables/all_methods_comparison_percent.csv`. Per-class and source-domain
tables are generated from the imported selected evaluation files.

## Verified runtime

The reported execution was a Kaggle Notebook run using 2 × Tesla T4 GPUs, global batch 32,
and seed 42.

| Component | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| TorchVision | 0.25.0+cu128 |
| CUDA reported by PyTorch | 12.8 |
| Ultralytics | 8.4.75 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.10.0 |
| Pillow | 11.3.0 |
| PyYAML | 6.0.3 |

## Figures and reports

The selected figure set is in `artifacts/merged_best_by_dataset/figures/` and is registered
with SHA-256, source artifact, checkpoint, split, display label, and actual source method in
`artifacts/metadata/figure_registry.json`. It includes split distribution, selected metric
comparisons, all-method comparisons, count and normalized confusion matrices, source-domain
comparison, and checkpoint provenance. The generated HTML report is
`artifacts/merged_best_by_dataset/reports/merged_best_by_dataset_report.html`.
The compatibility report path is
`artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html`.

## Reproduction

```powershell
Set-Location "D:\CVio\CVio_ShrimpDB_Combined_Research_Repo\studies\shrimpdb_combined_multisource"
uv python install 3.12
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv sync --frozen
uv run pytest -q
```

Separate workflow commands are:

```powershell
uv run python scripts\01_audit_datasets.py --config configs\study.yaml       # dataset audit
uv run python scripts\02_prepare_splits.py --config configs\study.yaml        # split preparation
uv run python scripts\03_train_shrimpdb3.py --config configs\study.yaml       # CE/ASL training entry points
uv run python scripts\04_train_combined4.py --config configs\study.yaml
uv run python scripts\05_evaluate_checkpoints.py --config configs\study.yaml  # evaluation
uv run python tools\audit_method_identity.py                                  # identity audit
uv run python tools\generate_merged_best_artifacts.py                          # merged validation, tables, figures, report
uv run python scripts\07_generate_html_report.py --config configs\study.yaml  # legacy report workflow
```

For Kaggle reproduction, attach `vohoangtu/shrimpdb`, enable Internet, select a GPU
accelerator, and use **Save Version -> Run All**. The notebook downloads
`uynnhy/processed-images` when it is not attached. The reported execution used 2 × Tesla T4.

## Checkpoint provenance and application integration

The selected checkpoint files are not tracked in Git. `model_registry/selected_best_by_dataset.json`
records the external source path, filename, SHA-256, class count, class order, display label,
and actual source method. The ShrimpDB-3 selected checkpoint hash is
`ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36`; the Combined-4 selected
checkpoint hash is `adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad`.
Retrieve a checkpoint from the merged package, verify its hash, and use the registry's class
order. Do not treat the common display label as evidence that the Combined-4 checkpoint uses
ASL-LDAM or SimAM-DCFR; it is a CE baseline checkpoint.

## Claims and limitations

These are validation-selected, fixed seed-42 results from one reported run. They do not
establish statistical significance, universal superiority, clinical validity, or production
readiness. The split is image-level and is not asserted to be specimen-safe; dataset shift,
label noise, calibration error, and external validity remain limitations. See
`docs/CLAIMS_AND_LIMITATIONS.md` and `docs/MERGED_BEST_BY_DATASET_RESULTS.md`.

## Citation and license

```bibtex
@software{nguyen2026shrimpdb_combined,
  title = {ShrimpDB Combined Multi-Source Classification Benchmark},
  year = {2026},
  url = {https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26}
}
```

Code: [AGPL-3.0-or-later](../../LICENSE). Datasets are not redistributed.
