# Model Card

## Model

YOLO26m-cls + ASL-LDAM + SimAM-DCFR attention.

## Intended Use

- Research on class-imbalanced shrimp disease image classification
- Reproduction of the fixed seed-42 paper experiment
- Controlled clean-test and synthetic corruption evaluation
- Explainability visualization for qualitative review

## Non-Intended Use

- Clinical or veterinary diagnosis
- Autonomous treatment decisions
- Claims of performance outside the documented dataset
- Multi-seed statistical inference

## Dataset

Kaggle `uynnhy/processed-images`, with four classes: Healthy, BG, WSSV, and
WSSV_BG. The data are already background-removed with U2Net/rembg.

The fixed split contains 804 training, 172 validation, and 173 test images.

## Metrics

Seed-42 clean test:

- Macro-F1: 0.910137
- Accuracy: 0.913295
- Cohen's Kappa: 0.881593

The comparison CE baseline has Macro-F1 0.890200.

## Risks

- Misclassification can lead to inappropriate interpretation of animal health.
- Dataset imbalance and limited acquisition diversity may produce subgroup or
  domain-specific failure.
- Background removal can alter visual cues.
- Synthetic corruption performance does not guarantee field robustness.

## Limitations

- Single seed and fixed split
- No statistical significance test across seeds
- No distributed weights
- No field or mobile deployment validation
- Potential nondeterminism from GPU and library behavior

## Ethical Notes

Outputs require expert review. The model must not replace laboratory testing,
veterinary assessment, or biosecurity procedures.

## Reproducibility

Reference runtime: Kaggle T4x2 GPU, Python 3.12.3, 30 epochs, image size 224,
seed 42. See `configs/train/yolo26m_asl_ldam_simam_dcfr.yaml` and
`docs/REPRODUCE.md`.
