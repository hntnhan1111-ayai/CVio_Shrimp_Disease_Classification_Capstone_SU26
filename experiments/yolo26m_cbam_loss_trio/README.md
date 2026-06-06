# YOLOv26m-cls CBAM loss trio

This folder is for the three-run experiment:

1. CE + CBAM
2. ASL + CBAM
3. ASL-LDAM + CBAM

Fixed references:

- Stage1 YOLOv26m-cls CE: Macro-F1 0.8902, Accuracy 0.8902, Kappa 0.8505.
- ASL-LDAM no attention: Macro-F1 0.8991, Accuracy 0.9017, Kappa 0.8661.
- Prior ASL-LDAM + sparse_learnable_fusion_cbam: Macro-F1 0.9006, Accuracy 0.9017, Kappa 0.8658.

Important: this runner requires the full CVio project code in the repository root, especially shrimp_scripts/ and experiments/asl_custom_loss_screening/. If those folders are missing on the target machine, pull/copy the full project code before running.
