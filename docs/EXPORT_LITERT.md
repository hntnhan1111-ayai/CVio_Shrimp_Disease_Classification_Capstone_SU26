# LiteRT/TFLite Export

This document covers exporting the trained main-method model to LiteRT/TFLite.
The export workflow is **separate from training** and never triggers training.

## Prerequisites

Use a clean **Python 3.12.2** virtual environment. Python 3.13/base
environments caused TensorFlow/tf-keras version problems during the original
export. Long Bash heredocs were error-prone when pasted into a terminal;
prefer the provided scripts and short CLI commands.

```bash
python -m venv .venv-export
source .venv-export/bin/activate        # Windows: .venv-export\Scripts\activate
python -m pip install -r requirements-export-litert.txt
```

`requirements-export-litert.txt` pins the known-good versions:

| Package | Version |
|---|---|
| Python | 3.12.2 |
| ultralytics | 8.4.75 |
| tensorflow | 2.19.1 |
| tf-keras | 2.19.0 |
| numpy | 2.1.3 |
| protobuf | 5.29.6 |
| onnx | 1.22.0 |
| onnxslim | 0.1.94 |
| onnxruntime | 1.27.0 |
| onnx2tf | 1.28.8 |
| sng4onnx | 2.0.1 |
| onnx-graphsurgeon | 0.6.1 |
| ai-edge-litert | 2.1.5 |

## Export command

```bash
python scripts/06_export_litert_fp32_fp16.py \
  --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt \
  --out-dir export \
  --imgsz 224
```

The script:

- accepts `--weights`, `--out-dir`, `--imgsz`, `--format tflite`
- never trains, never imports a training module
- produces `export/yolo26m_asl_ldam_simam_dcfr_fp32.tflite`
- produces `export/yolo26m_asl_ldam_simam_dcfr_fp16.tflite`
- writes `export/tflite_sanity_check.json` and `export/export_environment.json`
- runs a LiteRT/TFLite interpreter sanity check on both files
- fails loudly if either TFLite file is missing

Use `--skip-if-present` to avoid re-exporting when both files already exist.

## Provided files

The two TFLite files are committed to this repository (~40 MB FP32, ~20 MB FP16).
Both are under GitHub's 100 MiB block, so no Git LFS is used. See
[export/README.md](../export/README.md) for input/output shapes and class order.

## FP16 quantization note

FP16 quantization converts internal weights to float16. The input and output
tensors may remain float32 depending on the converter and runtime. The FP32
file is the safest baseline for deployment testing.
