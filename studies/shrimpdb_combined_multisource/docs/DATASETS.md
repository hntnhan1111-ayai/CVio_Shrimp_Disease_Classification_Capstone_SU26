# Datasets

This document describes the datasets used in the study **shrimpdb_combined_multisource**, including
sources, label harmonization, raw counts, and exclusions.

## 1. Data Sources

### 1.1 ShrimpDB

- **Repository**: [https://www.kaggle.com/datasets/vohoangtu/shrimpdb](https://www.kaggle.com/datasets/vohoangtu/shrimpdb)
- **Kaggle slug**: `vohoangtu/shrimpdb`
- **Role in this study**: Independent external-dataset training on three clinically aligned classes (ShrimpDB-3 experiment).

#### Raw Class Counts

| Original Class | Count |
|---:|:---|
| Den_Mang | 125 |
| Dom_Den | 103 |
| Dom_Trang | 173 |
| Hoai_Tu_Co | 115 |
| Hoai_tu_gan | 61 |
| Tom_BT | 74 |
| **Total** | **651** |

#### Label Harmonization

For the ShrimpDB-3 experiment, the five original ShrimpDB classes were mapped to three
clinically aligned target classes:

| Original Label | Target Label | Count |
|---|---:|:---|
| Tom_BT | Healthy | 74 |
| Den_Mang | BG | 125 |
| Dom_Trang | WSSV | 173 |
| Dom_Den | _excluded_ | 103 |
| Hoai_Tu_Co | _excluded_ | 115 |
| Hoai_tu_gan | _excluded_ | 61 |

**Excluded classes**: `Dom_Den`, `Hoai_Tu_Co`, and `Hoai_tu_gan` were excluded from the
ShrimpDB-3 experiment. The excluded labels were not represented in the harmonized target
class space and were not included in any training, validation, or test partition.

After label harmonization and exclusion, the ShrimpDB-3 dataset comprises **315 images**
distributed across the three target classes (`Healthy`, `BG`, `WSSV`).

### 1.2 ShrimpDiseaseDB

- **Repository**: [https://www.kaggle.com/datasets/uynnhy/processed-images](https://www.kaggle.com/datasets/uynnhy/processed-images)
- **Kaggle slug**: `uynnhy/processed-images`
- **Role in this study**: Primary multi-source dataset for the Combined-4 experiment.

#### Raw Class Counts

| Class | Count |
|---:|:---|
| Healthy | 403 |
| BG | 198 |
| WSSV | 328 |
| WSSV_BG | 220 |
| **Total** | **1,149** |

The `WSSV_BG` class (co-infected with both WSSV and baculovirus) is unique to ShrimpDiseaseDB
and was retained as a distinct class in the Combined-4 experiment.

## 2. Dataset Audit

Both datasets were audited using the `01_audit_datasets.py` script prior to splitting. The
audit verified that the actual on-disk image counts matched the expected counts listed above.
No drift was detected between expected and actual counts for either dataset.

### 2.1 Exact-Duplicate Detection

Exact-duplicate detection was performed within each label group (same-label duplicates) and
across label groups (cross-label duplicates). Detected duplicates were quarantined and excluded
from training. The audit artifacts are stored in `artifacts/audit/`.

**Important caveat**: Exact-duplicate detection confirms the removal of identical image files.
It does not prove the removal of all near-duplicate or augmented variants. The audit procedure
identifies byte-level identical images only.

## 3. Dataset Summary Tables

### 3.1 ShrimpDB-3 Post-Harmonization Counts

| Class | Count |
|---:|:---|
| Healthy | 74 |
| BG | 125 |
| WSSV | 173 |
| **Total** | **315** |

### 3.2 ShrimpDiseaseDB Counts

| Class | Count |
|---:|:---|
| Healthy | 403 |
| BG | 198 |
| WSSV | 328 |
| WSSV_BG | 220 |
| **Total** | **1,149** |

### 3.3 Combined-4 Post-Harmonization Counts

The Combined-4 experiment combines ShrimpDB-3 (after label harmonization) with ShrimpDiseaseDB:

| Class | ShrimpDB-3 | ShrimpDiseaseDB | Combined |
|---:|---:|---:|---:|
| Healthy | 74 | 403 | 477 |
| BG | 125 | 198 | 323 |
| WSSV | 173 | 328 | 501 |
| WSSV_BG | 0 | 220 | 220 |
| **Total** | **315** | **1,149** | **1,464** |

Note: The `WSSV_BG` class is contributed exclusively by ShrimpDiseaseDB; ShrimpDB contains no
co-infected images in its raw distribution.

## 4. Notes

- Dataset not redistributed. Raw images must be downloaded directly from the Kaggle sources
  listed above.
- The study does not modify the underlying images except as required for the label
  harmonization mapping described in Section 1.1.
- ShrimpDiseaseDB was already pre-processed (background removal via U2Net/rembg) by the
  dataset curators; this study does not re-apply background removal.
