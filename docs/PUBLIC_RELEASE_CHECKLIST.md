# Public release checklist

- [ ] Confirm the dataset may be referenced under its original access terms.
- [ ] Choose a license for original project code.
- [ ] Confirm redistribution rights for trained checkpoints.
- [ ] Review the Ultralytics license applicable to version 8.4.75.
- [ ] Install Git LFS before adding `.pt` files.
- [ ] Run `PYTHONPATH=src python scripts/verify_repository.py`.
- [ ] Run `sha256sum -c checkpoints/SHA256SUMS.txt`.
- [ ] Do not relabel RTX 4090 artifacts as Kaggle T4×2 results.
- [ ] Replace one-seed claims with multi-seed statistics before a strong journal claim.
