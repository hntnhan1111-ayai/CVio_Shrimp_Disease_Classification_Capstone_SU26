# Environment

## Actual frozen-artifact environment

The artifacts in this repository were generated or evaluated on:

| Item | Value |
|---|---|
| OS | Linux |
| Python | 3.12.13 |
| PyTorch | 2.12.1+cu126 |
| CUDA runtime | 12.6 |
| cuDNN | bundled with CUDA 12.6 |
| Ultralytics | 8.4.75 |
| NumPy | 2.5.0 |
| Pandas | 3.0.4 |
| OpenCV | 4.13.0.92 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.11.0 |
| SciPy | 1.15.0 |

**GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)

Source files:
- `environment/runtime_actual_rtx4090.json`
- `environment/pip_freeze_actual.txt`
- `environment/nvidia_summary_actual.csv`
- `environment/system_runtime_actual.txt`

## Target Kaggle reproduction environment

This is a **target profile** for future Kaggle T4×2 reproduction. Existing RTX 4090 artifacts must not be relabeled as T4×2 results.

| Item | Target |
|---|---|
| Accelerator | 2 × NVIDIA Tesla T4 |
| VRAM | 16 GB per GPU, 32 GB total |
| Architecture | Turing |
| Host CPU | ~4 vCPU |
| RAM | ~30 GB |
| Temporary disk | ~100–150 GB |
| Allocation | ~30 GPU hours/week |

Source file: `environment/kaggle_t4x2_target.yaml`

## Notes

- Kaggle allocations may vary. Verify actual T4 hardware before reporting results.
- The T4×2 target is provided for reproducibility planning, not as a claim that results were obtained on T4×2.
