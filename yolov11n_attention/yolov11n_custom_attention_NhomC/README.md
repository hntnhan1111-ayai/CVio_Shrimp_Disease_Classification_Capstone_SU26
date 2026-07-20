# YOLO11n Custom Attention - Nhom C

This folder implements Group C from `custom_attention_improvement_transfer_plan.md`.
The focus is mask-head and boundary quality ablations, not adding new attention blocks.

## Notebooks

- `yolo11_cus_att_nm64.ipynb`: C1 changes only the `Segment` mask prototype count from `nm=32` to `nm=64`.
- `yolo11_cus_att_boundary_aware_loss.ipynb`: C2 applies a process-local boundary-aware mask loss patch.
- `yolo11_cus_att_bilinear_upsample.ipynb`: C3 changes YOLO neck/head `nn.Upsample` layers from `nearest` to `bilinear`.

## Notes

- Existing baseline, Group A, and Group B notebooks are not modified.
- Group C reuses Group A registry/evaluation and Group B train protocol utilities so metrics and CSV layouts stay aligned.
- Variant YAMLs are generated under `/kaggle/working/yolov11n_custom_attention_NhomC/<experiment>/generated_yamls`.
- C3 includes `baseline_clean` through `_shared/yolo11n_seg_resolved_baseline.yaml`, a resolved YOLO11n-seg YAML equivalent to the n-scale model with `npr=64`.
- For `nm64`, the helper changes only the second `Segment` argument and preserves the source YAML `npr` value. Existing custom YAMLs use `Segment [nc, 32, 64]`, so generated variants become `Segment [nc, 64, 64]`.
- Boundary loss is patched at runtime only; vendored Ultralytics files are left unchanged.
- Each notebook has a smoke/preflight section for build and dummy-forward checks before full training.
