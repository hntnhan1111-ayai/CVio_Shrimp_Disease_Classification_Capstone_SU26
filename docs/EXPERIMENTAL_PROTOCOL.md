# Experimental protocol

## Training

- YOLO11s, 200 epochs, seed 42, deterministic mode.
- AdamW, `lr0=0.0005`, `lrf=0.01`, cosine schedule.
- Train image size 1280; batch size 2.
- `patience=0`; no early stopping.
- Mosaic, MixUp, Copy-Paste, CutMix, and erasing disabled.

## Evaluation

- Validation/test image size 1536.
- IoU 0.55; confidence 0.0005; max detections 600.
- Batch size 1; FP32.

## Fair baseline comparison

Both checkpoints are immutable and SHA-256 verified. They use the same test images, corruption realizations, evaluation parameters, and metric extraction.
