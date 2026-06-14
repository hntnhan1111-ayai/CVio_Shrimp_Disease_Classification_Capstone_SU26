# Data

## Source

- Kaggle dataset: `uynnhy/processed-images`
- Classes: `Healthy`, `BG`, `WSSV`, `WSSV_BG`
- Total images: 1,149

```python
import kagglehub

path = kagglehub.dataset_download("uynnhy/processed-images")
print("Path to dataset files:", path)
```

## Fixed Seed-42 Stage-1 Split

| Split | Healthy | BG | WSSV | WSSV_BG | Total |
|---|---:|---:|---:|---:|---:|
| train | 282 | 139 | 229 | 154 | 804 |
| val | 60 | 30 | 49 | 33 | 172 |
| test | 61 | 29 | 50 | 33 | 173 |

The portable split assignment is stored in
`artifacts/manifests/split_manifest_seed42.csv`. The split is stratified and
image-level. It is not claimed to be group-safe by shrimp identity.

## Background Removal

The Kaggle dataset already contains background-removed images generated with
U2Net/rembg. Reproduction skips background removal by default because it is
slow and unnecessary for the reported experiments.

Optional installation:

```bash
pip install "rembg[cpu]"
```

## Repository Exclusions

No raw images, generated corrupted-image folders, Kaggle caches, or copied
training split trees are committed. They belong under ignored `datasets/` or
`runs/` paths.
