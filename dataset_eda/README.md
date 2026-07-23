# CVio YOLO Dataset EDA Package

Generated: 2026-07-03T08:02:14

## Dataset

- data.yaml: `/home/drnguyenvinh/notebooks/shrimp_yolo_canonical_bg_wssv_2cls_split70_15_15/data.yaml`
- dataset root: `/home/drnguyenvinh/notebooks/shrimp_yolo_canonical_bg_wssv_2cls_split70_15_15`
- classes: `0:BG, 1:WSSV`

## Main counts

- total images: 746
- total boxes: 5569
- train images: 523
- val images: 112
- test images: 111
- audit issue images: 0

## Included

- CSV tables under `tables/`
- paper-ready plots under `figures/`
- `eda_report.html`
- `summary.json`

## Excluded by default

- raw dataset images
- raw labels folder copy
- model weights

## Paper usage

Use the box area, size bucket, and center heatmap figures to justify that the task is small-lesion detection.
Use the split and class-count tables in the Dataset and Experimental Setup sections.
Use the audit tables to report label quality and potential annotation issues.
