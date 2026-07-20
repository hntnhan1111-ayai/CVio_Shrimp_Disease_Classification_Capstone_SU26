# YOLO11n Custom Attention - Nhom B

This folder implements Group B from `custom_attention_improvement_transfer_plan.md`.
The focus is training protocol, not new attention architecture.

## Notebooks

- `yolo11_cus_att_warmup_cosine_patience.ipynb`: B1 warmup, cosine LR, shorter patience.
- `yolo11_cus_att_label_smoothing.ipynb`: B2 runtime label smoothing patch for YOLO segmentation classification targets.
- `yolo11_cus_att_underwater_aug_matched.ipynb`: B3 matched underwater/strong augmentation.
- `yolo11_cus_att_swa_after_training.ipynb`: B4 save periodic checkpoints and average them after training.

## Notes

- Existing baseline and custom attention notebooks are not modified.
- Model/evaluation registration is reused from Group A to keep metric logic identical.
- `label_smoothing` is a removed Ultralytics key in the vendored version, so B2 uses a process-local loss patch instead of passing `label_smoothing=0.05` to `model.train`.
- Every notebook writes training/evaluation CSVs under `/kaggle/working/yolov11n_custom_attention_NhomB/<experiment>/reports`.
- Custom YAMLs should pass shape/build sanity before running full training.

