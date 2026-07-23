# Locked checkpoints

| File | Role |
|---|---|
| `yolo11s_baseline_best.pt` | Matched YOLO11s baseline for clean and corruption comparison |
| `yolo11s_recsra_best.pt` | Frozen RECSRA winner checkpoint |
| `yolo11s_recsra_last.pt` | Final-epoch checkpoint retained for provenance |

Verify with `sha256sum -c checkpoints/SHA256SUMS.txt`. Git LFS is recommended.
