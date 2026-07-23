# Experiments 11-20

This group moves from basic Fourier filtering to controlled configuration selection, augmentation interaction, attention, and feature-Fourier experiments.

Main ideas:

- Sweep high-pass alpha around the original W1 setting.
- Run the Factorial A-H Fourier on/off comparison across clean-light augmentation and hook states.
- Tune Fourier configuration in the clean-light Mode G regime and perform a local sigma/alpha sweep.
- Test W1 with strong augmentation and SimAM+Coordinate Attention.
- Explore FDDem and other learnable or feature-level Fourier refinements.

Key conclusion: W1/G4 (`sigma=50, alpha=0.10`) remained the best image-level Fourier configuration. Fourier helped the clean-light hook-on regime and the reproduced attention branch, but the strongest baseline-architecture strong-augmentation result was later better without Fourier. FDDem is a separate feature-Fourier mechanism and should not be merged with image-level W1 results.

Experiment 16 and some FDDem tuning outputs require cross-checking against Downloads during final provenance cleanup.
