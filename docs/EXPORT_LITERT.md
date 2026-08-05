# LiteRT/TFLite export

The export workflow is separate from training and must receive an already
verified source checkpoint. Exporting a file does not verify its scientific
identity.

## Current artifact status

The tracked FP32 file is historical and traces to the rejected CE-architecture
SDI-4 candidate family. The FP16 file has no retained source-checkpoint hash
binding. Neither is an official proposed deployment artifact. Exact hashes and
statuses are listed in [CHECKPOINTS.md](CHECKPOINTS.md).

## Known export environment

Use the pinned packages in `requirements-export-litert.txt`; the retained
environment uses Python 3.12.2, Ultralytics 8.4.75, TensorFlow 2.19.1, and
AI Edge LiteRT 2.1.5. Treat `export/export_environment.json` as the local
machine-readable record.

```bash
python -m venv .venv-export
source .venv-export/bin/activate        # PowerShell: .venv-export\Scripts\Activate.ps1
python -m pip install -r requirements-export-litert.txt
```

## Export a verified source

```bash
python scripts/06_export_litert_fp32_fp16.py \
  --weights runs/training/<verified-run>/weights/best.pt \
  --out-dir export --imgsz 224
```

Before publication, bind each export to the source checkpoint SHA-256 and rerun
the evaluator. Record format, dtype, input/output shape, classes, metrics, and
the export hash in `weights/manifest.json`.

The script never trains or downloads the dataset. It produces FP32/FP16 TFLite
files and performs an interpreter shape/dtype sanity check; that format check is
not a metric reproduction test.
