# Verified checkpoint registry

This directory is a release registry, not a claim that every experiment has a
publishable binary. Only the two EXT-3-Original checkpoints passed the current
hash, class-order, architecture, and package-provenance gates. They are stored
through Git LFS.

The official SDI-4 metrics remain reported results, but neither corresponding
official-final binary was found. The similarly named checkpoint with SHA-256
`4305e491...38c1` is a historical CE architecture that scores below both
official targets. It is deliberately excluded from `weights/`.

Use [`manifest.json`](manifest.json) for the complete status and
[`SHA256SUMS.txt`](SHA256SUMS.txt) to verify the two released binaries. See
[`docs/CHECKPOINTS.md`](../docs/CHECKPOINTS.md) for the evidence standard and
load instructions.

## Published binaries

| Dataset | Method | File | Status |
|---|---|---|---|
| EXT-3-Original | CE | `ext3_original/yolo26m_cls_ce_seed42_best.pt` | `VERIFIED_EXT3_ORIGINAL_CE` |
| EXT-3-Original | ASL-LDAM + SimAM-DCFR | `ext3_original/yolo26m_cls_asl_ldam_simam_dcfr_seed42_best.pt` | `VERIFIED_EXT3_ORIGINAL_PROPOSED` |

These research checkpoints are not clinical or production diagnostic devices.
