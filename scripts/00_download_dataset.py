#!/usr/bin/env python3
"""Download the processed Kaggle dataset and print likely dataset roots."""

from pathlib import Path

import kagglehub

path = kagglehub.dataset_download("uynnhy/processed-images")
print("Path to dataset files:", path)

base = Path(path)
candidates = [base]
if base.exists():
    candidates.extend(folder for folder in base.rglob("*") if folder.is_dir())
print("Candidate dataset folders:")
for candidate in candidates[:100]:
    children = {child.name for child in candidate.iterdir()} if candidate.is_dir() else set()
    if children.intersection({"Healthy", "1. Healthy", "processed_images"}):
        print(" -", candidate)
