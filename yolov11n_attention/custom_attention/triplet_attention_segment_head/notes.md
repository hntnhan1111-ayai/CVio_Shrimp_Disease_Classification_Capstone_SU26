# Triplet Attention Segment Head

## Module

`TripletAttention` from `custom_attention/common/attention_modules.py`.

## Placement

Before Segment head on P3/P4/P5.

## Why this position

The module is placed near the mask-producing path so it can refine features that
feed the YOLO11n `Segment` head while leaving the grouped data split and baseline
training/evaluation protocol unchanged.

## Risks

Extra spatial gates can raise latency and may not improve strict mask IoU.

## Files changed or added

- `triplet_attention_segment_head.ipynb`: copied from the clean baseline and patched with custom attention setup, YAML build, smoke build, and shape sanity check cells.
- `yolov11n_triplet_segment.yaml`: YOLO11n-seg architecture with the custom attention placement.
- `notes.md`: this experiment note.
- Shared implementation: `../common/attention_modules.py`.

No installed Ultralytics source file is modified on disk; parser registration is
done in memory inside the notebook.

## How to run

Open `triplet_attention_segment_head.ipynb` and run all cells from top to bottom.

## Sanity pass condition

- The notebook cell named `Shape sanity check before training` passes for
  `[2, 64, 80, 80]`, `[2, 128, 40, 40]`, and `[2, 256, 20, 20]`.
- The model smoke-build cell prints `Smoke build OK`.
