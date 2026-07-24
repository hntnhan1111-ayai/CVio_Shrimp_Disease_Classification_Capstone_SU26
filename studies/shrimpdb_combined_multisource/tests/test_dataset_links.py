from pathlib import Path


def test_required_dataset_links_are_documented():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "https://www.kaggle.com/datasets/vohoangtu/shrimpdb" in text
    assert "https://www.kaggle.com/datasets/uynnhy/processed-images" in text
