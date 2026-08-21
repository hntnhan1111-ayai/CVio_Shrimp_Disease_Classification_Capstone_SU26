# Dataset and protocol catalog

The project currently has two distinct evaluation contexts. They must not be
pooled into one result table or compared as though they share a split.

## 1. Legacy report protocol

The report records a 129-image test set: 41 healthy negatives and 88
disease/labeled images, evaluated for BG/WSSV instance segmentation. It is the
source of the historical top-five table. The original `data.yaml`, immutable
split manifest and preprocessing recipe are not in this workspace, so this
protocol is historical-only until those inputs are restored.

## 2. Expanded dataset: ShrimpDisBD-TigerShrimp_MrTuDat v1

The local audit records this expanded/combined 1,452-image, two-class YOLO
export with 403 empty label images. The workspace contains evidence of filename
groups and near duplicates, so reruns must use a persisted leakage-safe grouped
split. The imported output results belong to this dataset. Raw images, labels
and the export ZIP remain outside this new layer.

The two current notebook tracks are preserved in place:

- `yolov11n_attention/expanded_grouped_data/`: expanded/combined dataset notebooks using the grouped near-duplicate split.
- `yolov11n_attention/mrtu_grouped_data/`: MrTu dataset leakage-safe module notebooks.

Use the configs in `../configs/datasets/` as the catalog. Before training,
provide an external YOLO `data.yaml`; do not commit the raw dataset.
