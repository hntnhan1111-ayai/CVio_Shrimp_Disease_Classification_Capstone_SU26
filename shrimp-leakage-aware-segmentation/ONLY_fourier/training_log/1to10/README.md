# Experiments 01-10

This group establishes the strict Fourier-only baseline and tests early image-level Fourier preprocessing variants.

Main ideas:

- Establish a no-Fourier, no-augmentation, hook-off control.
- Test fixed high-pass enhancement at different strengths.
- Compare band-pass enhancement, high-frequency damping, low-frequency flattening, and homomorphic filtering.
- Test train-only, mixed original/Fourier, and random Fourier-copy strategies.

Key conclusion: high-pass `sigma=50, alpha=0.10` was the strongest early image-Fourier setting, improving the bare baseline's disease mAP but increasing healthy false positives. Stronger high-pass and alternative filters did not provide a better balanced result.

Primary evidence is summarized in the consolidated training metrics and Fourier preprocessing plan. Some historical output artifacts may also remain in Downloads.
