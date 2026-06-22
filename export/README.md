# Exported LiteRT/TFLite Models

This folder contains the reviewer-facing LiteRT/TFLite exports of the main
method (YOLO26m-cls + ASL-LDAM + SimAM-DCFR), trained on the fixed seed-42 split.

## Files

| File | Approx. size | Description |
|---|---:|---|
| `yolo26m_asl_ldam_simam_dcfr_fp32.tflite` | ~40 MB | FP32 baseline; safest for deployment testing |
| `yolo26m_asl_ldam_simam_dcfr_fp16.tflite` | ~20 MB | FP16 weights; smaller and often faster on compatible GPU/mobile delegates |
| `tflite_sanity_check.json` | ? | Recorded interpreter sanity-check shapes/dtypes |
| `export_environment.json` | ? | Known-good export environment pins |

> FP16 quantization converts internal weights to float16; input/output tensors
> may remain float32 depending on the converter/runtime.

## Input

- RGB image, resized to **224?224**
- Normalized consistently with Ultralytics classification preprocessing
- Layout: **NHWC** `[1, 224, 224, 3]`
- dtype: `float32`

## Output

- Shape: `[1, 4]` (four class logits/probabilities ? verify with the sanity script)
- dtype: `float32`
- Apply `softmax` to obtain probabilities if the runtime returns logits

## Class order

| Index | Class |
|---:|---|
| 0 | Healthy |
| 1 | BG |
| 2 | WSSV |
| 3 | WSSV_BG |

## Run the sanity check

```bash
python - <<'PY'
import sys; sys.path.insert(0, "src")
from cvio_asl_ldam.export import tflite_sanity_check
print(tflite_sanity_check("export/yolo26m_asl_ldam_simam_dcfr_fp32.tflite"))
print(tflite_sanity_check("export/yolo26m_asl_ldam_simam_dcfr_fp16.tflite"))
PY
```

Requires `ai-edge-litert` or `tensorflow` (see `requirements-export-litert.txt`).

## Regenerate the files

Use a clean **Python 3.12.2** venv (3.13/base caused TensorFlow/tf-keras problems):

```bash
python -m venv .venv-export
source .venv-export/bin/activate        # Windows: .venv-export\Scripts\activate
python -m pip install -r requirements-export-litert.txt

python scripts/06_export_litert_fp32_fp16.py \
  --weights runs/training/yolo26m_cls__asl_ldam_simam_dcfr__seed42/weights/best.pt \
  --out-dir export \
  --imgsz 224
```

The export script never trains and never downloads the dataset.

## GitHub size note

- GitHub warns for files larger than 50 MiB and blocks files larger than 100 MiB.
- The FP32 file (~40 MB) is under the 50 MiB warning; the FP16 file (~20 MB) is smaller.
- Both files are committed to this repository without Git LFS.
- If a file is missing from your clone (e.g., stripped by a fork), regenerate it
  with the command above.
