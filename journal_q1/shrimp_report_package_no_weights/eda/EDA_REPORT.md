# ShrimpDiseaseImageBD EDA Report

Generated: 2026-06-15T19:53:30

## Dataset source
- Kaggle dataset: `nhanayai/shrimpdiseaseimagebd`
- Local raw path: `/home/drnguyenvinh/notebooks/shrimp_dataset_eda_label_model_analysis_v1/kaggle_raw/shrimpdiseaseimagebd`

## Raw class folders
- Healthy: 0 images | `missing`
- BG: 0 images | `missing`
- WSSV: 0 images | `missing`
- BG_WSSV: 0 images | `missing`

## Annotated disease folders
- BG: 198 images, 198 labels | `/home/drnguyenvinh/notebooks/shrimp_dataset_eda_label_model_analysis_v1/kaggle_raw/shrimpdiseaseimagebd/ShrimpDiseaseImageBD An Image Dataset for Computer Vision-Based Detection of Shrimp Diseases in Bangladesh/Root/Annotated Diseased Shrimp Images/Annotated Diseased Shrimp Images/1. BG`
- WSSV: 328 images, 328 labels | `/home/drnguyenvinh/notebooks/shrimp_dataset_eda_label_model_analysis_v1/kaggle_raw/shrimpdiseaseimagebd/ShrimpDiseaseImageBD An Image Dataset for Computer Vision-Based Detection of Shrimp Diseases in Bangladesh/Root/Annotated Diseased Shrimp Images/Annotated Diseased Shrimp Images/2. WSSV`
- WSSV_BG: 220 images, 220 labels | `/home/drnguyenvinh/notebooks/shrimp_dataset_eda_label_model_analysis_v1/kaggle_raw/shrimpdiseaseimagebd/ShrimpDiseaseImageBD An Image Dataset for Computer Vision-Based Detection of Shrimp Diseases in Bangladesh/Root/Annotated Diseased Shrimp Images/Annotated Diseased Shrimp Images/4. WSSV_BG`

## Canonical OD mapping
- `0 = BG`
- `1 = WSSV`
- `BG/source class 0 -> BG`
- `WSSV/source class 0 -> WSSV`
- `WSSV_BG/source class 0 -> WSSV`, `WSSV_BG/source class 1 -> BG`

## Main warnings
- No malformed/out-of-range labels found by this audit.
- Healthy has no disease-region boxes and should not be treated as a box-level OD class.
- For final mobile system, use a Healthy classification gate before the disease detector.
- For final paper, report both box-level OD metrics and image-level diagnosis metrics.

## Output folders
- Figures: `/home/drnguyenvinh/notebooks/shrimp_dataset_eda_label_model_analysis_v1/eda_outputs/figures`
- Tables: `/home/drnguyenvinh/notebooks/shrimp_dataset_eda_label_model_analysis_v1/eda_outputs/tables`
- Galleries: `/home/drnguyenvinh/notebooks/shrimp_dataset_eda_label_model_analysis_v1/eda_outputs/galleries`
- Overlays: `/home/drnguyenvinh/notebooks/shrimp_dataset_eda_label_model_analysis_v1/eda_outputs/overlays`