# Checkpoints

Selected checkpoints are external artifacts and are not tracked in Git. The paper-facing
summary is [`artifacts/tables/checkpoint_summary.csv`](../artifacts/tables/checkpoint_summary.csv).

| Dataset | Actual source method | Classes | SHA-256 |
|---|---|---:|---|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | 3 | `ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36` |
| Combined-4 | CE Baseline | 4 | `adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad` |

Both models consume RGB 224 × 224 images. Class order is `Healthy, BG, WSSV` for ShrimpDB-3
and `Healthy, BG, WSSV, WSSV_BG` for Combined-4. Retrieve a checkpoint from the reviewed
result package and verify its SHA-256 before use. The display label is not sufficient to
identify the model method.

This is research software, not a veterinary diagnostic system. It is not clinically
validated or production-ready.
