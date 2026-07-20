# New research attention modules

Each notebook is Kaggle-standalone and trains one independent, newly designed attention module twice: light baseline augmentation, then strong SimAM+CA-matched augmentation. Before either training run, it validates standalone tensor shapes at multiple channel/spatial sizes and performs a YOLO11n-seg build and 640×640 forward-pass sanity check.

`baseline_strong_augmentation/yolo11n_seg_baseline_strong_200e.ipynb` is the no-attention matched control for the strong runs. It uses the baseline-200e protocol and changes only the augmentation policy.

| Notebook | Module |
|---|---|
| `dpca_dual_polarity_contrast/` | DPCA — Dual-Polarity Contrast Attention |
| `laca_local_agreement/` | LACA — Local Agreement Contrast Attention |
| `odaa_orientation_diversity/` | ODAA — Orientation-Diversity Attention |
| `srea_scale_residual_evidence/` | SREA — Scale-Residual Evidence Attention |

Regenerate all notebooks locally with:

```powershell
python yolov11n_attention/new_research_attention/create_notebooks.py
```
