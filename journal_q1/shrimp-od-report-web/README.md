# ShrimpDiseaseImageBD OD Web Report

Static React + Vite + TypeScript dashboard for the disease-region EDA and 30-epoch YOLO screening package. It parses packaged CSV files at runtime and displays derived figures, contact sheets, error logs, and training diagnostics without a backend.

## Run locally

```powershell
npm install
npm run dev
npm run build
npm run preview
```

The static research artifacts are under `public/report-data/`. Missing CSVs or images produce visible warnings instead of crashing the report.

## Scope

- Canonical object classes: `0 = BG`, `1 = WSSV`
- Healthy: image-level state reserved for a later classification gate
- WSSV_BG: co-occurrence of BG and WSSV evidence
- Benchmark status: 30-epoch model screening, not final journal training

No raw dataset, checkpoints, weights, or exported model files are included.
