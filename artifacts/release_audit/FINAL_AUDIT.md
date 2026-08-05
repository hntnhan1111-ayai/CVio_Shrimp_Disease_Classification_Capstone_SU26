# Final checkpoint release audit

## Decision summary

The audit publishes only the two exact-hash EXT-3-Original checkpoints. Neither
official SDI-4 checkpoint binary was found. The uploaded proposed-named SDI-4
candidate is rejected as `HISTORICAL_NONFINAL` because it has a CE architecture
and fails both official metric targets.

## Repository preflight

- Repository: `CVio_Shrimp_Disease_Classification_Capstone_SU26`
- Target branch: `paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp`
- Remote: `origin` at the project GitHub repository
- Preserved prior work: local branch
  `wip/pre-classification-audit-20260805_222946`, commit
  `669cef5fd0e0a915f1bac057f92e877637a1dc9a`
- Git LFS: installed; release files use LFS pointers

Full machine-readable state: [`preflight.json`](preflight.json).

## Inventory coverage

The read-only inventory recorded 293 filesystem/Git/LFS candidate paths, 138
checkpoint members across 49 archives, and 26,877 textual references. It
searched the repository, practical CVio roots, attached Windows/WSL locations,
Git history, LFS objects, and explicitly supplied archives. Two unrelated
archives were unreadable and are recorded as errors rather than silently
skipped.

Exhaustive inventories:

- [`all_checkpoint_paths.tsv`](all_checkpoint_paths.tsv)
- [`all_checkpoint_hashes.tsv`](all_checkpoint_hashes.tsv)
- [`archive_checkpoint_hits.tsv`](archive_checkpoint_hits.tsv)
- [`checkpoint_reference_hits.tsv`](checkpoint_reference_hits.tsv)

Many of the 293 paths are duplicate archive/Git/LFS occurrences or unrelated
detection/GWHD experiments. The following table contains every unique in-scope
CVio classification candidate selected for trusted architecture/provenance
inspection.

## In-scope candidate decisions

| SHA-256 | Identity/evidence | Decision | Official publication |
|---|---|---|---|
| `4305e49158129c4a6acaa8fdaf7b5982a7f4d93cc233f11d17479e04b0e438c1` | SDI-4 proposed-named `.pt`; CE module tree; fixed-test Accuracy 0.867052, Macro-F1 0.863062 | `HISTORICAL_NONFINAL` | no |
| `4acb5924b7d1d845a775fdcd623edf61834821ca8cb13f5f3b7a10c9cc2a409d` | historical SDI-4 CE; package Macro-F1 0.874872 | `HISTORICAL_NONFINAL` | no |
| `8fd72ad5f1ffc9767f3ed24ce65c3ac3b51dc07e5e95045eac2943676cbcaafe` | historical SDI-4 proposed architecture; package Macro-F1 0.801939 | `HISTORICAL_NONFINAL` | no |
| `055e22edd699bea623a4b1b8c34ad7a395b56658036a4495e5b54159b786c8ea` | EXT-3 CE, three classes, exact package reference | `VERIFIED_EXT3_ORIGINAL_CE` | yes, Git LFS |
| `ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36` | EXT-3 proposed, three classes, late SimAM-DCFR present, exact package reference | `VERIFIED_EXT3_ORIGINAL_PROPOSED` | yes, Git LFS |
| `adebc0a4e16fe45f2b1f12e375c5514d15a9eafb8be5939834d27208a744c5ad` | combined SDI-4 + EXT-3 CE; package Macro-F1 0.865105 | `VERIFIED_ADDITIONAL_REGIME_CE` | no; analysis only |
| `9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606` | combined SDI-4 + EXT-3 proposed; package Macro-F1 0.821801 | `VERIFIED_ADDITIONAL_REGIME_PROPOSED` | no; analysis only |
| `44141c8e8ecc400b4b696adbfe736fe12aa07da42b53c6e2e99cb600268cb235` | ONNX tied to rejected 4305 family | `HISTORICAL_NONFINAL_EXPORT` | no |
| `9834ecbaeee14878da2fd53a811005cfa2ed8192b94f9d8a7f5b732b19d1689d` | FP32 TFLite tied to rejected 4305 family | `HISTORICAL_NONFINAL_EXPORT` | no |
| `c8d1f744871c32f79e495ee950c7a6fc53de114aaefab757e4da3ff83f1275d7` | FP16 TFLite without source-checkpoint binding | `UNRESOLVED_EXPORT` | no |

Candidate-level JSON records are under [`candidates/`](candidates/).

## SDI-4 candidate architecture and metric audit

The `4305e491...38c1` checkpoint records task classification, Ultralytics
8.4.72, seed 42, image size 224, 30 epochs, batch 32, AdamW, four classes, and
a `prepared_seed42` data path. Its stored run name is
`yolo26m_cls__baseline_ce__seed42`.

Architecture inspection found 10,358,340 parameters, head input shape
`[1,512,7,7]`, output shape `[1,4]`, no `AttentionBeforeClassify`, no
`SimAMDCFR`, and no stored attention audit. It is indistinguishable from the CE
architecture.

The tolerance `1e-6` absolute was declared before evaluation. The fixed manifest
resolved 173 images in order Healthy/BG/WSSV/WSSV_BG. Ultralytics 8.4.72 and
8.4.103 produced identical predictions:

- Accuracy: `0.8670520231213873`
- Macro-F1: `0.8630617393197718`
- Cohen's Kappa: `0.8184431465595912`

It matches neither the official proposed `0.913295/0.910137` nor CE
`0.890200/0.890200` result. Evaluation outputs, both confusion matrices,
predictions, environment, command, and hash are under
[`evaluation/4305e491.../`](evaluation/4305e49158129c4a6acaa8fdaf7b5982a7f4d93cc233f11d17479e04b0e438c1/).

## SDI-4 official-final gaps

No binary matching the official CE or official selected proposed result was
located. The historical `0.905448` class-wise report and matrices do not match
the official-final Macro-F1 `0.910137` and are not presented as such. The
missing `shrimp_results_summary_final_v3.zip` and its referenced historical run
directory remain unresolved.

## EXT-3 verification and publication

The exact hashes named by the results ZIP were found in a separate checkpoint
bundle. Both store three classes in order Healthy/BG/WSSV and produce output
shape `[1,3]`. The CE model has no late attention wrapper; the proposed model
contains `AttentionBeforeClassify`, `SimAMDCFR`, the expected texture/channel
branches, and residual path.

Package metrics are:

| Method | Accuracy | Macro-F1 | Kappa |
|---|---:|---:|---:|
| CE | 0.702128 | 0.722990 | 0.550239 |
| Proposed | 0.914894 | 0.912937 | 0.869444 |

The source images for the 47-image partition were unavailable for a second
inference run. Verification therefore binds exact binaries to package
predictions/reports/matrices and architecture, not to an independent rerun.

Published files:

- `weights/ext3_original/yolo26m_cls_ce_seed42_best.pt`
- `weights/ext3_original/yolo26m_cls_asl_ldam_simam_dcfr_seed42_best.pt`

Both are Git LFS objects with OIDs equal to their checkpoint SHA-256 values.

## Additional-regime reversal

For the 220-image combined package, CE Macro-F1 `0.865105` exceeds proposed
`0.821801`. A legacy display label incorrectly named the selected CE artifact
as proposed; audited `actual_source_method` identifies CE. Documentation and
figures use the source method.

## Documentation and visualization outcome

README and supporting documents now separate evidence levels, define the loss
and attention computations, distinguish published work from project additions,
provide reproduction/evaluation/export commands, and document non-clinical use.
Four quantitative charts are generated from retained CSV evidence, four vector
diagrams are deterministic, and the XAI panel carries a qualitative-only
warning.

## QA outcome

- Ruff formatting and lint on all changed Python files: pass.
- Full pytest: 14 passed, 1 skipped (TFLite interpreter dependency absent).
- Repository smoke script: pass after removing a machine-specific path.
- Python compile/import checks: pass.
- EXT-3 trusted checkpoint loads and `[1,3]` output checks: pass.
- Figure existence and PNG/SVG parse checks: pass for nine README figures.
- Manifest/hash/LFS pointer verification: pass.
- Local Markdown link check: pass.
- `git lfs fsck`: pass.
- `git diff --check`: pass.

Detailed receipts:

- [`test_report.txt`](test_report.txt)
- [`hash_verification.txt`](hash_verification.txt)
- [`readme_link_check.txt`](readme_link_check.txt)

## Unresolved evidence gaps

1. Exact official-final SDI-4 CE binary.
2. Exact official-final SDI-4 ASL-LDAM + SimAM-DCFR binary.
3. A verified FP32/FP16 export bound to the official-final proposed source hash.
4. EXT-3 source images for an independent 47-image inference rerun.
5. Multi-seed, specimen-independent, field, and deployment validation.

These gaps block only the affected scientific labels and artifacts; they do not
invalidate the two exact-hash EXT-3 checkpoint releases.

## Remote publication

The final gate confirmed the local branch was 11 commits ahead and 0 behind its
upstream, found redacted credentials through Git Credential Manager, uploaded
both Git LFS objects (43 MB total), and advanced the remote paper branch from
`137f54f` through the audited release series without force. The local WIP branch
and its preservation commit were not pushed.
