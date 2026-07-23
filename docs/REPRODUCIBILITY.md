# Reproducibility guide

1. Install Git LFS and verify checkpoints.
2. Create the Python 3.12 environment.
3. Install CUDA-enabled PyTorch 2.12.1 and pinned packages.
4. Install the local package with `pip install -e .`.
5. Patch Ultralytics.
6. Configure the dataset path.
7. Reproduce clean evaluation before the 50 corruption conditions.

Do not replace the fair loading flow with only a raw `load_state_dict` call. The trainer must receive the prepared model after direct transfer and wrapper remapping.
