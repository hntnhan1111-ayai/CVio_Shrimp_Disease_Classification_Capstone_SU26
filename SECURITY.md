# Security and model-file safety

PyTorch `.pt` checkpoints may contain Python pickles. Load only checkpoints whose SHA-256 matches `checkpoints/SHA256SUMS.txt`, and use a trusted environment. Do not execute checkpoints obtained from unverified sources.
