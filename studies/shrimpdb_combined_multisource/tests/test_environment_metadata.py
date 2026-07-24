from pathlib import Path


def test_verified_runtime_metadata_is_documented():
    text = Path("README.md").read_text(encoding="utf-8")
    for value in ("Python | 3.12.13", "PyTorch | 2.10.0+cu128", "TorchVision | 0.25.0+cu128", "CUDA reported by PyTorch | 12.8", "Ultralytics | 8.4.75", "2 × Tesla T4"):
        assert value in text
