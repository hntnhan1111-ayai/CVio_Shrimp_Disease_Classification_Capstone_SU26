# Reproduction workflow

1. Choose exactly one dataset config: `legacy_129_test` or `mrtu_v1`.
2. Preserve or create a split manifest before training. For `mrtu_v1`, the
   split must be grouped and near-duplicate leakage-safe.
3. Run the strong baseline and a candidate with the same seed, image size,
   augmentation, epoch budget, threshold and checkpoint-selection rule.
4. Store one manifest for every run using `scripts/02_create_run_manifest.py`.
5. Collect full mask mAP50, disease-only mask mAP50, disease mAP50-95, healthy
   FP, mask-count MAE and disease miss rate. Calculate HA with the package
   helper.
6. Repeat seeds 42, 43 and 44 before reporting mean ± std.

`scripts/04_run_ultralytics_benchmark.py` invokes `YOLO.train()` only for the
three stock Ultralytics models declared runnable in `config.yaml`. Run it first
with `--dry-run`, then `--smoke`. The report-selected custom layers remain
notebook implementations and are blocked until each has a parity regression
test and a versioned model YAML.

## Importing completed expanded-dataset runs

When new result ZIPs are placed in the repository-level outputs/ directory, run:

```powershell
python scripts/03_import_mrtu_outputs.py
```

The importer brings compact CSV histories, arguments and selected figures into
artifacts/. It never moves or deletes the original ZIPs, checkpoints or
notebooks.
