# Dataset EDA

## Dataset overview

The canonical dataset contains **746 images** of individual shrimp photographed onshore using consumer mobile phones. All source images are **2048 × 2048** pixels.

### Split proportions

| Split | Images | Percentage |
|---|---:|---:|
| Train | 523 | 70.11% |
| Validation | 112 | 15.01% |
| Test | 111 | 14.88% |

### Class distribution

| Class | Boxes | Percentage |
|---|---:|---:|
| BG | 2,121 | 38.09% |
| WSSV | 3,448 | 61.91% |

### Box statistics

| Metric | Value |
|---|---:|
| Total boxes | 5,569 |
| Mean boxes per image | 7.465 |
| Median boxes per image | 6 |
| Mean box area | 0.0997% |
| Median box area | 0.0491% |

### Size buckets

| Size bucket | Boxes | Percentage |
|---|---:|---:|
| Tiny (< 0.1% area) | 3,933 | 70.62% |
| Small (0.1–1% area) | 1,598 | 28.69% |
| Medium (1–5% area) | 38 | 0.68% |

## Implications for high-resolution detection

The dataset is dominated by **tiny lesions**: approximately 70.62% of all boxes occupy less than 0.1% of the 2048×2048 image area. This concentration creates a challenging small-object detection task. RECSRA's channel and spatial evidence pathways are intended to improve feature discrimination at very small scales, but EDA alone does not prove that any specific attention mechanism solves this — it only establishes the task's difficulty.

## Label audit

- Audit issue images: 0
- Empty/missing-label images: 0
- Duplicate labels: verified in `dataset_eda/tables/duplicate_labels.csv`

## Spatial distribution

The box center heatmap (`dataset_eda/figures/fig_box_center_heatmap.png`) shows the spatial concentration of annotated lesions. This matters because RECSRA's spatial evidence module attends to spatial layout, but the heatmap alone does not validate spatial attention efficacy.

## Class imbalance

WSSV boxes outnumber BG boxes by approximately 1.6:1 (61.91% vs 38.09%). The Focal Loss component in the training protocol (via the classification weight `cls=0.35`) partially addresses this.

## Figures and tables

Full EDA figures and tables are in `dataset_eda/figures/` and `dataset_eda/tables/`. The interactive report is `dataset_eda/eda_report.html`.

## Recommended reading

- `dataset_eda/figures/fig_box_size_bucket.png` — justifies small-lesion challenge
- `dataset_eda/figures/fig_class_percent_total.png` — class imbalance overview
- `dataset_eda/figures/fig_box_center_heatmap.png` — spatial distribution
- `dataset_eda/figures/fig_split_image_count.png` — split proportions
- `dataset_eda/tables/dataset_overview.csv` — numeric overview
- `dataset_eda/tables/class_counts_by_split.csv` — per-split class counts
