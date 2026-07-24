# Methodology

## Method-identity scope

This study preserves two executable method definitions. The CE baseline uses
`CrossEntropyLoss` with no custom attention. The proposed method uses ASL-LDAM with
`gamma_pos=0.0`, `gamma_neg=4.0`, `label_smoothing=0.1`, `ldam_max_m=0.5`,
`ldam_scale=30.0`, and SimAM-DCFR attention with `e_lambda=0.0001`. The merged package's
display label is not a substitute for `actual_source_method`; the selected Combined-4
metrics are from CE Baseline, while the selected ShrimpDB-3 metrics are from ASL-LDAM +
SimAM-DCFR. See `artifacts/metadata/merged_result_registry.json`.

This document describes the model architecture, loss function, attention mechanism, and training
configuration used in the **shrimpdb_combined_multisource** study.

## 1. Model

**Architecture**: YOLO26m-cls (medium classification variant of the YOLO architecture)

YOLO26m-cls is an image classification model derived from the YOLO family. It was selected
for this study due to its balance of parameter efficiency and representational capacity, and
its availability with ImageNet-pretrained weights through the Ultralytics framework.

- **Pretrained weights**: ImageNet-pretrained `yolo26m-cls.pt` from Ultralytics
- **Input resolution**: 224 x 224 pixels, RGB
- **Task**: Multi-class image classification

## 2. Loss Function: ASL-LDAM

The study uses a custom composite loss designated **ASL-LDAM**, which combines Asymmetric Loss
(ASL) with Label-Distribution-Aware Margin (LDAM) regularisation.

### 2.1 Asymmetric Loss (ASL)

ASL addresses class imbalance by applying asymmetric focusing factors to positive and negative
gradients:

```
L_ASL = (1 - y)^gamma_pos * log(1 - p) * x_ent_pos + y^gamma_neg * log(p) * x_ent_neg
```

where `p` is the model's predicted probability for the target class, `y` is the binary label,
`gamma_pos` controls down-weighting of easy positive samples, and `gamma_neg` controls
down-weighting of easy negative samples.

### 2.2 LDAM Regularisation

LDAM introduces class-dependent decision margins to improve classification boundaries for
tail classes. During training, the margin for class `c` is:

```
m_c = max_m / (N_c)^(1/4)
```

where `N_c` is the number of training samples in class `c` and `max_m` is a scalar upper bound.
The logit for class `c` is reduced by `m_c` before softmax.

### 2.3 ASL-LDAM Hyperparameters

| Parameter | Value | Description |
|---|---:|---|
| `gamma_pos` | 0.0 | Positive focusing factor (no down-weighting of positives) |
| `gamma_neg` | 4.0 | Negative focusing factor (down-weights easy negatives) |
| `label_smoothing` | 0.1 | Label smoothing epsilon |
| `ldam_max_m` | 0.5 | Maximum LDAM margin |
| `ldam_scale` | 30.0 | LDAM logit scaling factor |

## 3. Attention Mechanism: SimAM-DCFR

The study incorporates **SimAM-DCFR** (Simulated Attention with Gated Residual and
Dual-Channel Feature Refinement) as a custom attention module integrated into the YOLO26m-cls
backbone.

### 3.1 SimAM (Simulated Attention)

SimAM derives attention weights from the spatial energy function of feature tensors without
additional learnable parameters. For a given feature tensor, the attention weight at spatial
position `(x, y)` is computed based on the local mean and variance, enabling the model to
adaptively emphasise informative spatial regions.

### 3.2 DCFR (Dual-Channel Feature Refinement)

DCFR applies a gated residual pathway to refine texture-sensitive features. The gated
mechanism modulates the residual contribution to preserve fine-grained texture cues that are
relevant for shrimp disease classification under noisy imaging conditions.

### 3.3 SimAM-DCFR Hyperparameters

| Parameter | Value | Description |
|---|---:|---|
| `e_lambda` | 0.0001 | Regularisation coefficient for energy-based attention |

## 4. Training Configuration Summary

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate (initial) | 0.00125 |
| Learning rate final | 0.01 (cosine decay from initial) |
| Learning rate schedule | Cosine annealing |
| Epochs | 30 |
| Patience (early stopping) | 15 |
| Global batch size | 32 |
| Data loading workers | 4 |
| Automatic mixed precision | Enabled |
| Image size | 224 |
| Auto-augmentation | RandAugment |
| Random erasing probability | 0.4 |
| Checkpoint save period | Disabled (saves best and last only) |
| Seed | 42 |
| Deterministic mode | Enabled |

## 5. Optimiser Details

AdamW with decoupled weight decay was selected over SGD for its adaptive learning-rate
properties, which are beneficial when fine-tuning on small, imbalanced datasets. The initial
learning rate of 0.00125 was chosen as a moderate starting point appropriate for fine-tuning
pretrained convolutional features.

## 6. Regularisation

In addition to the LDAM margin regularisation and label smoothing, the training configuration
employs RandAugment and random erasing (probability 0.4) as data-level regularisers to
improve generalisation under imaging variability.
