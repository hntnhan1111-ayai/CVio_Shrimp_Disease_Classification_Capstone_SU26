# ShrimpDiseaseImageBD Object Detection EDA and YOLO Screening Report

This branch packages a static research dashboard, derived EDA artifacts, notebooks, and result summaries for disease-region localization in **ShrimpDiseaseImageBD**.

## Branch purpose

`paper/journal_Q1` supports Q1-journal-oriented experiment review and planning. The current evidence is a controlled 30-epoch YOLO screening benchmark, not final paper training.

## Task formulation

- Object-detection classes are **BG** and **WSSV** disease regions.
- Healthy is an image-level state, not a bounding-box class, and should be handled by a later classification gate.
- WSSV_BG is treated as co-occurrence of BG and WSSV evidence.
- Folder-local source IDs are canonically remapped before training.

## Current findings

- YOLOv8n is the best current screening model under this specific protocol.
- Absolute mAP remains low, indicating difficult disease-region localization and substantial small-object risk.
- YOLOv13n/s did not produce valid accuracy results. The packaged logs show missing model files during loading, so these runs are environment/model compatibility failures.
- No result in this branch establishes deployment readiness.

## Run the report

```powershell
cd journal_q1/shrimp-od-report-web
npm install
npm run dev
npm run build
npm run preview
```

## Repository structure

```text
journal_q1/
├── shrimp-od-report-web/                 # React/Vite static dashboard
├── shrimp_report_package_no_weights/     # Safe derived EDA and result artifacts
├── *.ipynb                               # Lightweight experiment notebooks
└── shrimp_eda_yolo_results_no_weights.zip
```

## Data and artifact policy

The raw ShrimpDiseaseImageBD dataset is not committed. Only derived EDA/report artifacts are included. Raw datasets, Kaggle caches, model checkpoints, exported models, `node_modules`, and build output are intentionally excluded.

## Reproducibility and next experiments

The package and notebooks document the current 30-epoch screening. Journal-grade experiments should train the top candidates for 100–200 epochs, use multiple seeds, tune thresholds only on validation data, evaluate the final test set once, and report device-specific latency and model footprint. The next system component should be a Healthy classification gate followed by disease-region detection and BG/WSSV evidence aggregation.

## Citation and acknowledgments

- ShrimpDiseaseImageBD dataset citation: **TODO: insert the verified dataset citation supplied by the dataset authors.**
- YOLO / Ultralytics citation: **TODO: insert the verified citation for the exact implementations used.**

## License

This branch does not add or modify a repository license. Dataset and upstream framework terms remain applicable.
