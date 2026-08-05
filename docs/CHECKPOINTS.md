# Checkpoint registry and release decisions

## Release outcome

Only the two exact-hash EXT-3-Original checkpoints are published. Both official
SDI-4 checkpoint slots remain unresolved. This is intentional: a reported
metric is not sufficient to identify a binary, and a filename is not sufficient
to identify architecture or dataset regime.

![Provenance diagram showing sequential hash, dataset, class, architecture, and metric gates](assets/diagrams/checkpoint_provenance.svg)

*Figure 1. Release gate. Failure at any stage holds or rejects the candidate; no
later label can repair an earlier mismatch.*

## Primary checkpoint matrix

| Dataset regime | Method | SHA-256 | Test | Accuracy | Macro-F1 | Decision |
|---|---|---|---:|---:|---:|---|
| SDI-4 | CE | not located | 173 | 0.890200 | 0.890200 | `UNRESOLVED` — result reference only |
| SDI-4 | ASL-LDAM + SimAM-DCFR | not located | 173 | 0.913295 | 0.910137 | `UNRESOLVED` — result reference only |
| EXT-3-Original | CE | `055e22edd699bea623a4b1b8c34ad7a395b56658036a4495e5b54159b786c8ea` | 47 | 0.702128 | 0.722990 | `VERIFIED_EXT3_ORIGINAL_CE` |
| EXT-3-Original | ASL-LDAM + SimAM-DCFR | `ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36` | 47 | 0.914894 | 0.912937 | `VERIFIED_EXT3_ORIGINAL_PROPOSED` |

The full machine-readable registry is
[`weights/manifest.json`](../weights/manifest.json); hashes for released files
are in [`weights/SHA256SUMS.txt`](../weights/SHA256SUMS.txt).

## Why `4305e491...38c1` was rejected

The uploaded SDI-4 archive contains a checkpoint named as if it were proposed,
but the checkpoint itself records the run name
`yolo26m_cls__baseline_ce__seed42`. Architecture inspection found no
`AttentionBeforeClassify`, no `SimAMDCFR`, and no stored attention audit.

On the fixed 173-image seed-42 test partition, both Ultralytics 8.4.72 and
8.4.103 produced the same result:

| Accuracy | Macro-F1 | Cohen's Kappa | Status |
|---:|---:|---:|---|
| 0.867052 | 0.863062 | 0.818443 | `HISTORICAL_NONFINAL` |

This fails both the official proposed target and the official CE target at the
predeclared absolute tolerance `1e-6`. The candidate is neither an official
proposed checkpoint nor a substitute official CE checkpoint. See its
[`candidate record`](../artifacts/release_audit/candidates/4305e49158129c4a6acaa8fdaf7b5982a7f4d93cc233f11d17479e04b0e438c1.json).

## Other in-scope findings

| SHA-256 | Regime/method | Evidence | Status |
|---|---|---|---|
| `4acb5924b7d1d845a775fdcd623edf61834821ca8cb13f5f3b7a10c9cc2a409d` | SDI-4 CE | package clean Macro-F1 0.874872 | `HISTORICAL_NONFINAL` |
| `8fd72ad5f1ffc9767f3ed24ce65c3ac3b51dc07e5e95045eac2943676cbcaafe` | SDI-4 proposed architecture | package clean Macro-F1 0.801939 | `HISTORICAL_NONFINAL` |
| `adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad` | SDI-4 + EXT-3 CE | package clean Macro-F1 0.865105 | `VERIFIED_ADDITIONAL_REGIME_CE` |
| `9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606` | SDI-4 + EXT-3 proposed | package clean Macro-F1 0.821801 | `VERIFIED_ADDITIONAL_REGIME_PROPOSED` |

The complete path-level inventory, including unrelated checkpoint candidates,
is retained in
[`all_checkpoint_paths.tsv`](../artifacts/release_audit/all_checkpoint_paths.tsv)
and [`all_checkpoint_hashes.tsv`](../artifacts/release_audit/all_checkpoint_hashes.tsv).

## Deployment exports

| File | SHA-256 | Decision |
|---|---|---|
| `export/yolo26m_asl_ldam_simam_dcfr_fp32.tflite` | `9834ecbaeee14878da2fd53a811005cfa2ed8192b94f9d8a7f5b732b19d1689d` | `HISTORICAL_NONFINAL_EXPORT`; traces to rejected CE-architecture family |
| `export/yolo26m_asl_ldam_simam_dcfr_fp16.tflite` | `c8d1f744871c32f79e495ee950c7a6fc53de114aaefab757e4da3ff83f1275d7` | `UNRESOLVED_EXPORT`; source-checkpoint binding absent |

Neither file is an approved export of the official SDI-4 proposed checkpoint.
The ONNX file found in the uploaded archive has SHA-256
`44141c8e8ecc400b4b696adbfe736fe12aa07da42b53c6e2e99cb600268cb235`
and belongs to the same historical family as FP32.

## Safe loading and verification

Project checkpoints are trusted local artifacts, but they are PyTorch pickle
containers. Do not load arbitrary third-party `.pt` files. The audit registers
the repository classes through `register_checkpoint_safe_globals()` before
loading trusted candidates.

Verify released binaries before use:

```powershell
Get-FileHash -Algorithm SHA256 weights\ext3_original\*.pt
git lfs pull --include="weights/ext3_original/*.pt"
git lfs fsck
```

These EXT-3 checkpoints output three classes in this exact order:
`Healthy`, `BG`, `WSSV`. They must never be substituted into four-class SDI-4
evaluation.
