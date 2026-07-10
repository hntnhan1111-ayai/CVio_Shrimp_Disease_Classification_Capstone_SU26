# Triplet Attention Segment Head - Multiseed Verification

## Direction

Best performance candidate from the single-seed sweep.

## Source Module

- Source folder: `custom_attention/triplet_attention_segment_head`
- Model YAML: `custom_attention/triplet_attention_segment_head/yolov11n_triplet_segment.yaml`
- Source class: `TripletAttention` from `custom_attention/common/attention_modules.py`
- Placement: P3/P4/P5 before `Segment`

## Multiseed Scope

This folder does not define a new attention module. It only provides a focused
entrypoint for rerunning the existing Triplet module with seeds `42`, `3407`,
and `2026`. Threshold sweep and aggregation are separate shared steps after
checkpoints are produced.

## Notebook

Open:

```text
triplet_attention_segment_head_multiseed.ipynb
```

The notebook is standalone: it contains the copied Triplet module setup and
runs the three seeds directly in notebook cells. It does not call
`run_top4_multiseed.py` or other shared pipeline scripts.

## Expected Trade-Off

Triplet had the strongest single-seed mask mAP and healthy-aware score, but it
also showed a large train/validation segmentation-loss gap. The multiseed run is
mainly checking whether that performance is repeatable or seed-sensitive.
