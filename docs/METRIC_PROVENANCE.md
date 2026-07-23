# Metric provenance

This document maps every major public metric to its authoritative source file and explains why two baseline contexts exist.

## Two baseline contexts

### Context A: Original YOLO11s 14-model benchmark baseline

**Authoritative source:** `results/raw/clean_original_baseline_vs_recsra.json`

This is the recorded original YOLO11s baseline row from the independent 14-model benchmark, reproduced in the HTML report:
`cvio_yolo11s_paper_ready_outputs_no_weights/paper_final_outputs/paper_ready_summary_percent.html`

**Headline values:**
- mAP50: 12.900%
- mAP50-95: 3.800%
- Precision: 22.400%
- Recall: 21.800%
- BG AP50-95: 4.000%
- WSSV AP50-95: 3.500%
- FPS: 169.5
- Model size: 18.39 MB
- Parameters: 9.43 M

**Usage:** Main clean-test comparison, README headline, paper main table, MODEL_CARD headline, clean-result figures, efficiency comparison, main contribution statement.

**Canonical public table:**
- `results/tables/clean_original_yolo11s_vs_recsra_percent.csv`
- `results/tables/clean_original_yolo11s_vs_recsra.md`
- `paper/tables/clean_results.tex`

**Legacy alias:** `results/tables/clean_baseline_vs_recsra_percent.csv` and `clean_baseline_vs_recsra.md` now also point to Context A (regenerated from the canonical source).

### Context B: Paired corruption-control checkpoint baseline

**Authoritative source:** `results/raw/corruption_control_clean_metrics.json`
**Raw evaluation:** `results/raw/clean_metrics.csv`, `results/raw/metrics_json/CLEAN/baseline.json`

This is the matched baseline checkpoint that was locked and re-evaluated under the corruption benchmark protocol. Its clean metrics differ from Context A because it was evaluated as part of the same corruption protocol.

**Headline values:**
- mAP50: 14.418%
- mAP50-95: 4.147%
- Precision: 20.428%
- Recall: 24.980%
- BG AP50-95: 4.477%
- WSSV AP50-95: 3.816%

**Usage:** Clean sanity check inside the robustness experiment; paired corruption evaluation; ten-corruption tables; top-five corruption rankings; per-severity robustness figures.

**Canonical public table:**
- `results/tables/corruption_control_clean_percent.csv`

**Raw evidence:** `results/raw/all_clean_and_50_condition_metrics.csv` (102 rows: 2 clean + 100 corruption)

## Source-precedence order

1. Raw checkpoint metadata, raw JSON, raw CSV, actual run args.yaml, actual test output
2. Locked experiment package and SHA-256 manifests
3. Original YOLO11s baseline row in the HTML report
4. Canonical EDA summary.json and CSV tables
5. Generated Markdown, HTML, LaTeX, or figures
6. Legacy paper-ready archive

Generated files never override raw evidence.

## RECSRA current best method

**Experiment ID:** SD001_A_RCCM_EV015FAIR_200E_SEED42
**Model name:** YOLO11s-RECSRA
**Checkpoint:** `checkpoints/yolo11s_recsra_best.pt`
**SHA-256:** `c1652101bb870a0b174b569ac47a70bb8cfafe119577e1ca2fe2de87f57824ff`

**Authoritative raw source:** `results/raw/metrics_json/CLEAN/recsra.json`
**Canonical source:** `results/raw/clean_original_baseline_vs_recsra.json`

## Obsolete results

V007 C3k2SimAM results (mAP50 16.391%, mAP50-95 4.775%) are excluded from the current research narrative. They are preserved only in the legacy archive `cvio_yolo11s_paper_ready_outputs_no_weights/` for traceability and must not appear in public-facing files.

The historical identifiers `0.163909` (mAP50) and `0.047748` (mAP50-95) correspond to V007 and are prohibited from public narrative.
