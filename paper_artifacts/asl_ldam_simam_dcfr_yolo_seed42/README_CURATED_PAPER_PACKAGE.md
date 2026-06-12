# Curated Final Paper Artifacts

This package is a curated paper-oriented artifact bundle intended for final manuscript writing.

Included only high-value items:
- 4 no-background sample images (one per class) + one combined sample panel
- best-method XAI panels + one combined XAI figure
- baseline model comparison tables (YOLO / TIMM)
- improvement-vs-baseline tables (YOLO / TIMM)
- Kaggle reference configuration table (from notebook71df4a9a4b)
- best-method training curves (train/loss, top-1 accuracy, val/loss only)
- key confusion matrices:
  - baseline CE clean
  - best method clean
  - best method impulse_noise severity 3
- official top-5 noise tables and figures:
  - impulse_noise
  - gaussian_noise
  - contrast_reduction
  - defocus_blur
  - low_light

Excluded intentionally:
- raw corrupted test image trees
- excessive confusion matrices
- meaningless or redundant figures
- model weights/checkpoints
- huge logs and raw predictions not needed for the paper body

The package is optimized to support a concise paper (approximately <= 12 pages).
