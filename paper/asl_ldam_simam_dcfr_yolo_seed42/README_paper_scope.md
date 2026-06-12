# Paper Scope

Recommended title:

**ASL-LDAM with SimAM-DCFR Attention for Robust YOLO-Based Shrimp Disease Image Classification Under Noisy Imaging Conditions**

Branch:

`paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp`

Scope:
- Fixed seed-42 Stage-1 split only.
- YOLO and TIMM baseline comparison.
- Main improved method: YOLO26m-cls + ASL-LDAM + SimAM-DCFR.
- Top-5 noise robustness based on the weakest CE baseline corruptions.
- XAI generated for the best method.
- No mean ± std over multiple seeds should be claimed.
- No model weights/checkpoints are included.
- No raw corrupted image datasets are included.

Main paper artifacts are stored in:

`paper_artifacts/asl_ldam_simam_dcfr_yolo_seed42/`

Experiment code is stored in:

`experiments/asl_ldam_simam_dcfr_yolo_seed42/`
