# CESA-Lite Segment Head - Multiseed Verification

## Direction

High mask-quality candidate from the legacy segment-head attention group.

## Source Module

- Source folder: `custom_attention/cesa_lite_segment_head`
- Model YAML: `custom_attention/cesa_lite_segment_head/yolov11n_cesa_lite.yaml`
- Source class: `CESALite` from `custom_attention/common/attention_modules.py`
- Placement: P3/P4/P5 before `Segment`

## Multiseed Scope

This folder only reruns the existing CESA-Lite Segment Head module across the
configured seeds. It does not change the module, split, class names, or baseline
training protocol.

## Notebook

Open:

```text
cesa_lite_segment_head_multiseed.ipynb
```

The notebook is standalone: it contains the copied CESA-Lite setup and runs the
three seeds directly in notebook cells. It does not call `run_top4_multiseed.py`
or other shared pipeline scripts.

## Expected Trade-Off

CESA-Lite was close to Triplet in single-seed test mAP. The key question is
whether its higher disease miss rate persists across seeds.
