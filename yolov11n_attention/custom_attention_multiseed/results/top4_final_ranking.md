# Top-4 Multiseed Final Ranking

This report is generated from threshold-sweep CSV outputs. It does not train or evaluate models by itself.

- Required seeds: `42, 3407, 2026`
- Default threshold for multiseed summary: `conf=0.25`, `iou=0.7`
- Final winners are not claimed unless every top-4 module has all three default-threshold seed rows.

## Data Completeness

No completed sweep rows were found yet. Run training and evaluation first.

```bash
python custom_attention_multiseed/run_top4_multiseed.py
python custom_attention_multiseed/evaluate_top4_threshold_sweep.py
python custom_attention_multiseed/aggregate_multiseed_results.py
```
