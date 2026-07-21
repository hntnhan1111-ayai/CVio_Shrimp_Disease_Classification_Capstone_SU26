# Closure Audit — shrimpdb_combined_multisource

- **Audit date (UTC):** 2026-07-21T12:31:50.501441+00:00
- **Branch:** `paper/shrimpdb-combined-asl-ldam-simam-dcfr`
- **Starting commit (this continuation):** `c70eecd5cee8ded9cf76a43a36510bc28c5571eb`
- **Final local hash:** `9f82add8859b3bfe17143d19eb7c6fa9928f93ae`
- **Final remote hash:** `9f82add8859b3bfe17143d19eb7c6fa9928f93ae`
- **Remote branch URL:** <https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26/tree/paper/shrimpdb-combined-asl-ldam-simam-dcfr>

## Test and Tool Results

- **Test command:** `uv run pytest -q`
- **Exit code:** `0`
- **Pytest summary:** 

| Tool | Result |
|---|---|
| `tools/validate_artifact_tree.py` | PASS |
| `tools/verify_checksums.py` | PASS |
| `tools/compare_source_snapshot.py` | PASS |
| `tools/closure_audit.py` | PASS |

## Source Snapshot Provenance

- **Status:** `match_with_explicit_diff`
- **Snapshot tree SHA-256:** `17a9f497afe86c5cc750acd3b2460c86fb710053006f65f2f110e6cec9ba3917`
- **Repository tree SHA-256:** `37837e005c95d458dcf211272115fdc19dfa543ab11350ab6431946111a41037`
- **Files missing in repository:** 0
- **Files missing in package:** 16
- **Content mismatches:** 1 (allowlisted only; see `allowlisted_diffs` in `artifacts/metadata/source_comparison.json`)

## Figure and Report Registries

- **Figure registry:** `artifacts/metadata/figure_registry.json` — 14 verified entries
  - Figure 1, Figure 10, Figure 13, Figure 14, Figure 15, Figure 16, Figure 2, Figure 3, Figure 4, Figure 5, Figure 6, Figure 7, Figure 8, Figure 9
- **HTML report:** `artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html`
  - SHA-256: `8d13f1ff3d774d48473807210801108b9635b5eb00b3745bae559bd2fc7e275f`
  - Size: 2009106 bytes; embedded images: 17
  - Sections required: True

## Final Application Checkpoint

- **Path (not tracked):** `final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt`
- **SHA-256:** `9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606`
- **Effective epochs (one-based indexing):**
  - ShrimpDB-3: executed 21 of 30 budget epochs; best validation checkpoint at epoch 6; early-stopped after `patience=15` triggered.
  - Combined-4: executed 30 of 30 budget epochs; best validation checkpoint at epoch 26; `patience=15` not triggered.

## Repository Hygiene

- **Status:** `clean`
- **Tracked files:** 218 (total 6367157 bytes)
- **Forbidden tracked files:** 0
- **Unapproved .pt files tracked:** 0
- **Files > 50 MB tracked:** 0

## CI Workflow

- **Path:** `.github/workflows/shrimpdb-combined-ci.yml`
- The workflow runs: `uv sync --frozen`, environment check, YAML/JSON validation, the full pytest suite, source comparison, artifact-tree validation, checksum verification, the closure audit, and a whitespace `git diff --check`.

## Commits Added in This Continuation

| Hash | Subject |
|---|---|
| `ef370e8` | chore: harden study validators and add source snapshot comparator |
| `d436602` | feat: add source snapshot provenance and comparison artifact |
| `4d2b245` | feat: add paper figures, figure registry, and split-distribution chart |
| `a1cd4c6` | feat: add report registry, effective epochs, hygiene, and closure audit artifacts |
| `6d1d7c4` | test: expand scientific protocol and metric regression coverage |
| `9f82add` | docs: fix effective epoch wording, integrate HTML report, expand figures, and harden CI |

## Remaining Scientific Limitations

- Single seed (42) only; no mean ± standard deviation across seeds.
- No statistical significance claim; bootstrap CI provided for test metrics.
- Image-level split; specimen or animal identity not established.
- Exact-duplicate detection does not prove removal of all near duplicates.
- ShrimpDB-3 and Combined-4 use different class spaces (3 vs 4); direct overall comparison is not controlled.
- ShrimpDB-3 executed through epoch 21 (best at epoch 6, patience=15 triggered). Combined-4 completed full 30 epochs.
- Calibration remains weak (ECE 19.30% ShrimpDB-3 best.pt, 33.46% Combined-4 best.pt).
- Validation-selected Combined-4 best.pt is the designated application candidate; last.pt exhibits higher test metrics but is post-hoc.
- No raw dataset, virtual environment, prepared image tree, result ZIP, or unapproved checkpoint is tracked.
- Research code only; not a veterinary diagnostic system.

## Non-Tracked Artifacts

- Raw ShrimpDB and ShrimpDiseaseDB images (download from Kaggle; not tracked).
- The 110.4 MB CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS.zip (not tracked; contents reproduced under artifacts/).
- Trained .pt checkpoints (excluded by .gitignore; documented in model_registry/checkpoints.json with SHA-256).
- Virtual environment (.venv/), __pycache__, .pytest_cache, .ipynb_checkpoints (gitignored).

## Final Git State

```
## paper/shrimpdb-combined-asl-ldam-simam-dcfr
 M artifacts/metadata/repository_hygiene.json
 M artifacts/metadata/source_comparison.json
?? _b.py
```

## Status

**`closure_ok`**
