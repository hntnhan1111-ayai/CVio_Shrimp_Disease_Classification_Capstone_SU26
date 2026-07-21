# Results


## Standalone HTML Report

The full standalone HTML report (figures, tables, limitations, provenance) is
available at `artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html`.
Its SHA-256, byte size, embedded image count, and section coverage are recorded in
`artifacts/metadata/report_registry.json`.

This document presents the evaluation results for both experiments in the
**shrimpdb_combined_multisource** study. All numbers are taken directly from the verified
artifact files in `artifacts/evaluation/`.

## 1. ShrimpDB-3 Experiment (best.pt, test split)

The ShrimpDB-3 experiment trains on three classes (`Healthy`, `BG`, `WSSV`) sourced exclusively
from ShrimpDB. The test split contains **47 images**.

### 1.1 Overall Metrics

| Metric | Value |
|---|---:|
| Accuracy | 91.49% |
| Balanced accuracy | 91.55% |
| Macro precision | 91.41% |
| Macro recall | 91.55% |
| Macro-F1 | 91.29% |
| Weighted-F1 | 91.63% |
| Cohen's kappa | 86.94% |
| MCC | 87.19% |
| Top-2 accuracy | 100.00% |
| ECE (15 bins) | 19.30% |

### 1.2 Per-Class Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Healthy | 90.91% | 90.91% | 90.91% | 11 |
| BG | 83.33% | 93.75% | 88.24% | 16 |
| WSSV | 100.00% | 90.00% | 94.74% | 20 |
| **Macro avg** | **91.41%** | **91.55%** | **91.29%** | 47 |
| **Weighted avg** | **92.20%** | **91.49%** | **91.63%** | 47 |

### 1.3 Confusion Matrix (counts)

|  | Pred: Healthy | Pred: BG | Pred: WSSV |
|---|---:|---:|---:|
| **True: Healthy** | 10 | 1 | 0 |
| **True: BG** | 1 | 15 | 0 |
| **True: WSSV** | 0 | 2 | 18 |

## 2. Combined-4 Experiment (best.pt, test split)

The Combined-4 experiment trains on four classes (`Healthy`, `BG`, `WSSV`, `WSSV_BG`) sourced
from both ShrimpDB (after label harmonization) and ShrimpDiseaseDB. The test split contains
**220 images**.

### 2.1 Overall Metrics

| Metric | Value |
|---|---:|
| Accuracy | 83.18% |
| Balanced accuracy | 83.85% |
| Macro precision | 81.80% |
| Macro recall | 83.85% |
| Macro-F1 | 82.18% |
| Weighted-F1 | 83.19% |
| Cohen's kappa | 77.15% |
| MCC | 77.51% |
| Top-2 accuracy | 95.45% |
| ECE (15 bins) | 33.46% |

### 2.2 Per-Class Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Healthy | 88.00% | 91.67% | 89.80% | 72 |
| BG | 80.85% | 84.44% | 82.61% | 45 |
| WSSV | 90.91% | 71.43% | 80.00% | 70 |
| WSSV_BG | 67.44% | 87.88% | 76.32% | 33 |
| **Macro avg** | **81.80%** | **83.85%** | **82.18%** | 220 |
| **Weighted avg** | **84.38%** | **83.18%** | **83.19%** | 220 |

### 2.3 Confusion Matrix (counts)

|  | Pred: Healthy | Pred: BG | Pred: WSSV | Pred: WSSV_BG |
|---|---:|---:|---:|---:|
| **True: Healthy** | 66 | 0 | 3 | 3 |
| **True: BG** | 5 | 38 | 0 | 2 |
| **True: WSSV** | 3 | 8 | 50 | 9 |
| **True: WSSV_BG** | 1 | 1 | 2 | 29 |

## 3. Matched ShrimpDB Test-Domain Comparison

This section compares the ShrimpDB-only model and the Combined-4 model on the **same 47 images**
from the ShrimpDB test split (three classes only: `Healthy`, `BG`, `WSSV`). This comparison
controls for the test domain to assess cross-domain transfer effects when training data is
augmented with ShrimpDiseaseDB.

### 3.1 Matched-Domain Metrics

| Metric | ShrimpDB-only | Combined-4 | Absolute Delta | Relative Change |
|---|---:|---:|---:|---:|
| Accuracy | 91.49% | 85.11% | -6.38 pp | -6.98% |
| Balanced accuracy | 91.55% | 87.92% | -3.64 pp | -3.97% |
| Macro-F1 | 91.29% | 64.77% | -26.53 pp | -29.06% |

### 3.2 Interpretation

The Combined-4 model, which was trained on both ShrimpDB and ShrimpDiseaseDB data, shows
degraded performance on the ShrimpDB test domain relative to the ShrimpDB-only model. The
largest reduction is observed in Macro-F1 (absolute change: -26.53 percentage points, relative
change: -29.06%). This performance reduction may reflect a cross-domain trade-off: the
Combined-4 model optimises for a broader, four-class objective that includes the `WSSV_BG`
class from ShrimpDiseaseDB, which is absent from the ShrimpDB test set. The lower Macro-F1 is
influenced by the `WSSV_BG` class having zero support in the ShrimpDB test domain, which
reduces the standard macro-average across all four classes.

The reduction should be interpreted as a possible negative transfer or domain mismatch effect
rather than a definitive conclusion, given the fixed-seed, single-run nature of this study.

## 4. Source-Domain Analysis (Combined-4 best.pt)

The Combined-4 model's test performance decomposed by source dataset:

| Source Dataset | Images | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| ShrimpDB | 47 | 85.11% | 87.92% | 64.77% |
| ShrimpDiseaseDB | 173 | 82.66% | 82.34% | 81.95% |

### 4.1 ShrimpDB Source-Domain Per-Class Metrics (Combined-4)

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Healthy | 91.67% | 100.00% | 95.65% | 11 |
| BG | 71.43% | 93.75% | 81.08% | 16 |
| WSSV | 100.00% | 70.00% | 82.35% | 20 |
| WSSV_BG | 0.00% | 0.00% | 0.00% | 0 |
| **Macro avg** | **65.77%** | **65.94%** | **64.77%** | 47 |

Note: `WSSV_BG` has zero support in the ShrimpDB test set; its F1-score is therefore 0.00%,
which reduces the macro-average. This is consistent with the dataset composition.

### 4.2 ShrimpDiseaseDB Source-Domain Per-Class Metrics (Combined-4)

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Healthy | 87.30% | 90.16% | 88.71% | 61 |
| BG | 88.46% | 79.31% | 83.64% | 29 |
| WSSV | 87.80% | 72.00% | 79.12% | 50 |
| WSSV_BG | 67.44% | 87.88% | 76.32% | 33 |
| **Macro avg** | **82.75%** | **82.34%** | **81.95%** | 173 |

## 5. Checkpoint Audit

### 5.1 Combined-4 best.pt vs last.pt

| Checkpoint | Accuracy | Balanced Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|
| best.pt (validation-selected) | 83.18% | 83.85% | 82.18% | 33.46% |
| last.pt (epoch 30 final) | 88.18% | 88.18% | 87.61% | 38.08% |

The Combined-4 training completed the full 30-epoch budget (patience=15 was not triggered).
Combined-4 best.pt corresponds to epoch 26 (minimum val/loss across epochs 1-30; see artifacts/final_application_model/effective_epochs.json).
However, `last.pt` exhibits higher test metrics than `best.pt` under the current evaluation.

### 5.2 Checkpoint Selection Warning

Selecting `last.pt` based on test metrics is not valid practice, as it constitutes a
post-hoc selection that invalidates the test set as an unbiased estimator. The
validation-selected `best.pt` is the appropriate checkpoint for any application use. Future
reruns should define checkpoint selection using validation Macro-F1 or balanced accuracy
_before_ test evaluation, and should select the checkpoint that maximises the chosen
validation metric.

### 5.3 Final Application Checkpoint

The application candidate checkpoint is:

- **File**: `artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt`
- **SHA-256**: `9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606`
- **Experiment**: Combined-4 (validation-selected best.pt)
- **Class order**: Healthy (0), BG (1), WSSV (2), WSSV_BG (3)
- **Input**: RGB, 224 x 224
