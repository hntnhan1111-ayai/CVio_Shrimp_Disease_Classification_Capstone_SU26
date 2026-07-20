# YOLO11n Custom Attention - Nhom D

This folder implements Group D from `custom_attention_improvement_transfer_plan.md`.
The focus is controlled architecture attention, with one main architectural change per notebook.

## Notebooks

- `yolo11_cus_att_simam_backbone_p3p4.ipynb`: D1 inserts SimAM after original backbone P3/P4 C3k2 layers.
- `yolo11_cus_att_ca_backbone_dual.ipynb`: D2 inserts CoordAtt after original backbone P3/P4 C3k2 layers for Triplet and CoTE.
- `yolo11_cus_att_lka_head_refine.ipynb`: D3 inserts lightweight LKA residual refiners on Segment inputs for Triplet and CoTE.

## Notes

- Existing Group A/B/C notebooks and vendored Ultralytics files are not modified.
- Backbone insertion automatically remaps positive layer indices, so head concat links point to the post-attention P3/P4 outputs.
- `LKAResidualAttention` and `LPSCSoftGate` are registered process-locally by `_shared/nhomD_arch_utils.py`.
- Variant YAMLs are generated under `/kaggle/working/yolov11n_custom_attention_NhomD/<experiment>/generated_yamls`.
- Each notebook has a build/dummy-forward sanity section before training.

