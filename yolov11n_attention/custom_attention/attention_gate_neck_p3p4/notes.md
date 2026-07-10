# Attention Gate Neck P3/P4

## Module

`AttentionGate` from `custom_attention/common/attention_modules.py`.

## Placement

After neck P3 and P4 fusion outputs; P4/P5 semantic features act as gates.

## Why this position

The module is placed near the mask-producing path so it can refine features that
feed the YOLO11n `Segment` head while leaving the grouped data split and baseline
training/evaluation protocol unchanged.

## Risks

Gate can suppress small lesions if semantic feature is too coarse.

## Files changed or added

- `attention_gate_neck_p3p4.ipynb`: copied from the clean baseline and patched with custom attention setup, YAML build, smoke build, and shape sanity check cells.
- `yolov11n_attention_gate_neck.yaml`: YOLO11n-seg architecture with the custom attention placement.
- `notes.md`: this experiment note.
- Shared implementation: `../common/attention_modules.py`.

No installed Ultralytics source file is modified on disk; parser registration is
done in memory inside the notebook.

## How to run

Open `attention_gate_neck_p3p4.ipynb` and run all cells from top to bottom.

## Sanity pass condition

- The notebook cell named `Shape sanity check before training` passes for
  `[2, 64, 80, 80]`, `[2, 128, 40, 40]`, and `[2, 256, 20, 20]`.
- The model smoke-build cell prints `Smoke build OK`.
