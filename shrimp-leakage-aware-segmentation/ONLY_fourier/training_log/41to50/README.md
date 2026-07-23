# Experiments 41-50

This group completes the N=2 Fourier-order screening and evaluates robustness and deployment cost.

Main ideas:

- Test the remaining N=2 augmentation pairs: Translate+FlipLR, Scale+FlipUD, Scale+FlipLR, and Translate+Degrees.
- Build a shared noisy test set for W1, Erasing, FlipUD, and Scale+Erasing.
- Measure Gaussian, salt-and-pepper, color-cast, low-contrast, and motion-blur robustness.
- Measure single-image Fourier preprocessing and YOLO prediction latency.

Key conclusion: Translate+FlipLR without Fourier is the strongest N=2 augmentation-only result. W1 is best for color cast and motion blur, while Erasing has the strongest mean noisy mAP50 and retention. Single-image W1 latency is substantially higher, but its YOLO-only component needs repeated randomized timing before being treated as an intrinsic model-speed difference.

The noise delta fields require clean-reference correction; raw noisy mAP50 values remain valid. The latest timing result is the batch-1 run in experiment 46.
