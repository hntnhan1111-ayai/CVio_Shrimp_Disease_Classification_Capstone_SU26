# Experiments 31-40

This group formalizes offline Fourier-order experiments for N=1 augmentation and begins N=2 composition studies.

Main ideas:

- Use offline augmentation and offline Fourier processing for a fairer comparison.
- Test two values per N=1 family and both processing orders.
- Rank augmentation families by raw augmentation-only performance.
- Test N=2 combinations such as Translate+Scale, Translate+Erasing, Translate+FlipUD, and Scale+Erasing.

Key conclusion: there is no universal order. Augmentation before Fourier is generally safer for Scale and FlipUD, while Fourier before augmentation is stronger for selected Erasing and composed pairs. Positive Fourier deltas do not necessarily produce the best absolute model; absolute mAP and healthy-aware metrics must be reported together.

First-run partial CSVs and second-run paper-row CSVs are both retained as provenance.
