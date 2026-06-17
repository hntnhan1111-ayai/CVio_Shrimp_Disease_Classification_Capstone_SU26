# ShrimpDiseaseImageBD 14-Model 100-Epoch Canonical OD Report

This folder contains a standalone static HTML dashboard for the 14-model, 100-epoch canonical BG/WSSV object-detection benchmark.

## Open the report

Double-click:

```text
journal_q1_od_100ep_report/index.html
```

Or serve locally:

```powershell
cd journal_q1_od_100ep_report
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Data source

Source package:

```text
shrimp_od_14models_100ep_canonical_results_no_weights/
```

The report uses `final_model_comparison_14models_100ep_canonical.csv` as the source of truth. CSV/JSON/Markdown/config files are copied into `data/`; safe PNG/JPG diagnostics are copied into `assets/`.

## Key results

- All 14 models completed 100 epochs.
- Best detector localization: `yolov9t` with mAP50-95 ≈ 0.0439.
- Best mAP50 and AP_WSSV: `yolo26s`, but it is heavier and weaker for diagnosis/miss-rate.
- Best mobile-balanced candidate: `yolov5nu`.
- Best image-level diagnosis: `yolov8s`, but its localization mAP is not the best.
- mAP remains low, so precise lesion localization is still an open research problem.

## Excluded artifacts

The report intentionally excludes raw dataset images, YOLO dataset images/labels, model weights/checkpoints, exported model files, `node_modules`, and `dist`.

## Next experiments

Run validation-only threshold tuning, resolution ablation at 640/1024/1280, small-object interventions such as SAHI/sliced inference, a Healthy classification gate, and an Android benchmark on a fixed 50-image set across three devices.
