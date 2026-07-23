# RECSRA method

**Robust Energy-Guided Circular–Spatial Residual Attention (RECSRA)** is the academic name of the frozen `SD001_A_RCCM` winner.

- Reporting name: `YOLO11s-RECSRA`
- Backbone insertion layers: `4`, `6`, `8`
- Parameters in the frozen training log: `8,343,534`
- Compute in the frozen training log: `20.2 GFLOPs`

```text
x → RMS channel energy → circular Conv1D
  ↘ grouped horizontal/vertical spatial evidence
  ↘ boundary-density evidence
  → fused evidence → tanh-bounded residual modulation → output
```

The frozen forward form is `output = x × (1 + α × tanh(evidence / T))`, with `α=0.15` and `T=1.0`.

The checkpoint was serialized with `SLDRA` and `C3k2SLDRA`; compatibility aliases are intentionally preserved.
