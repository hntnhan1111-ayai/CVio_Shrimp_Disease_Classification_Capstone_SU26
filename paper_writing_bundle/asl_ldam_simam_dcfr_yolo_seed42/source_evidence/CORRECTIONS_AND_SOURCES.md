# Corrections and Sources for Curated Paper Package

Updated: 2026-06-12T11:22:27.412499

## Corrected YOLO baseline source

The previous curated package accidentally read `baseline_ce_native_ultralytics` rows from a later noise-evaluation output where YOLO26m CE was 0.850541 Macro-F1. For the final paper package, YOLO baseline comparison is corrected using:

`shrimp_results_summary_final_v3.zip::csv/Stage1_YOLO.csv`

Key corrected row:

- YOLO26m-cls CE baseline: Macro-F1 = 0.890200, Accuracy = 0.890200, Kappa = 0.850500.

## Corrected YOLO26m best method key result

The final paper key result uses the user-confirmed seed-42 metrics:

- Model: YOLO26m-cls
- Best method: ASL-LDAM + SimAM-DCFR (`asl_ldam_simam_dcfr_top1`)
- Macro-F1 = 0.910137
- Accuracy = 0.913295
- Kappa = 0.881593
- Delta vs CE: +0.019937 Macro-F1, +0.023095 Accuracy, +0.031093 Kappa.

See `tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv`.

## Corrected TIMM baseline source

Missing TIMM baseline rows were filled from the previous seed-42 TIMM baseline table supplied by the user as a screenshot. The screenshot is included at:

`source_evidence/timm_baseline_previous_seed42_table_screenshot.png`

## Scope

All final tables are seed-42 only. Do not claim mean±std or statistical significance across seeds.
