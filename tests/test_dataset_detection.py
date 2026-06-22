"""Test robust class-root detection with a tiny synthetic folder structure."""

from pathlib import Path

import pytest

from cvio_asl_ldam.data.audit_dataset import resolve_dataset_root, looks_like_dataset_root


CONFIG = {
    "classes": [
        {"name": "Healthy", "aliases": ["Healthy", "1. Healthy"]},
        {"name": "BG", "aliases": ["BG", "2. BG"]},
        {"name": "WSSV", "aliases": ["WSSV", "3. WSSV"]},
        {"name": "WSSV_BG", "aliases": ["WSSV_BG", "4. WSSV_BG"]},
    ]
}


def test_detects_numbered_class_folders(tmp_path: Path):
    # Simulate the Kaggle structure: .../versions/1/processed_images/{1. Healthy, ...}
    root = tmp_path / "versions" / "1" / "processed_images"
    for alias in ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]:
        (root / alias).mkdir(parents=True)
    found, candidates = resolve_dataset_root(tmp_path, CONFIG)
    assert found == root.resolve()
    assert looks_like_dataset_root(found, CONFIG)


def test_detects_plain_class_folders(tmp_path: Path):
    root = tmp_path / "processed_images"
    for alias in ["Healthy", "BG", "WSSV", "WSSV_BG"]:
        (root / alias).mkdir(parents=True)
    found, _ = resolve_dataset_root(tmp_path, CONFIG)
    assert found == root.resolve()


def test_raises_when_no_class_root(tmp_path: Path):
    (tmp_path / "unrelated").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_dataset_root(tmp_path, CONFIG)
