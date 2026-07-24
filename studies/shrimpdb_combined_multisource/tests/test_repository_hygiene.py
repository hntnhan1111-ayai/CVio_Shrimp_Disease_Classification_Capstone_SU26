from pathlib import Path


def test_merged_import_does_not_contain_checkpoints_or_zip_files():
    root = Path("artifacts/merged_best_by_dataset")
    assert not list(root.rglob("*.pt"))
    assert not list(root.rglob("*.zip"))
