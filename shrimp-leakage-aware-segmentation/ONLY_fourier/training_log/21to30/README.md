# Experiments 21-30

This group explores newer Fourier ideas and the first systematic N=1 augmentation interactions.

Main ideas:

- Confirm and refine FDDem feature-Fourier designs.
- Sweep hybrid image/feature Fourier ideas and cutoff policies.
- Compare W1 with single augmentation families: HSV, Translate, Scale, FlipLR, Degrees, Erasing, and FlipUD.
- Compare Fourier before versus after augmentation, including the special HSV order study.

Key conclusion: Fourier interaction is augmentation- and strength-dependent. Fourier is harmful for many Translate and Scale settings, but positive interactions appear for selected Degrees, FlipLR, Erasing, and FlipUD values. Notebook 23's W1-only order pair is partly confounded by cached versus online Fourier execution; nonzero HSV settings provide the cleaner order evidence.

Several N=1 outputs are partial or remain in Downloads, especially around experiments 25-29.
