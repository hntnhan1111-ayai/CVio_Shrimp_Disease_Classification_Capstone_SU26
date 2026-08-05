# Historical LiteRT/TFLite exports

## Release warning

These files are retained for format inspection and provenance. They are **not**
verified exports of the official SDI-4 ASL-LDAM + SimAM-DCFR checkpoint.

| File | SHA-256 | Status |
|---|---|---|
| `yolo26m_asl_ldam_simam_dcfr_fp32.tflite` | `9834ecbaeee14878da2fd53a811005cfa2ed8192b94f9d8a7f5b732b19d1689d` | `HISTORICAL_NONFINAL_EXPORT`; family traces to rejected CE architecture |
| `yolo26m_asl_ldam_simam_dcfr_fp16.tflite` | `c8d1f744871c32f79e495ee950c7a6fc53de114aaefab757e4da3ff83f1275d7` | `UNRESOLVED_EXPORT`; source binding absent |

The filenames predate the release audit and do not prove model identity.

## Recorded tensor interface

- Input: float32 NHWC `[1, 224, 224, 3]`.
- Output: float32 `[1, 4]`.
- Recorded class order: Healthy, BG, WSSV, WSSV_BG.
- FP16 refers to internal weight conversion; input/output may remain float32.

Run the shape/dtype sanity check:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "src")
from cvio_asl_ldam.export import tflite_sanity_check

for name in ("fp32", "fp16"):
    path = f"export/yolo26m_asl_ldam_simam_dcfr_{name}.tflite"
    print(path, tflite_sanity_check(path))
PY
```

Do not use these exports for an official-result or deployment claim. Regenerate
exports only from a source checkpoint that has passed the hash, class-order,
architecture, and metric gates in [CHECKPOINTS.md](../docs/CHECKPOINTS.md).
