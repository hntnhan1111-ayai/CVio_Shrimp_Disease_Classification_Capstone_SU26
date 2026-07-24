import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_merged_registry_preserves_display_and_source_methods():
    registry = json.loads((ROOT / "artifacts/metadata/merged_result_registry.json").read_text(encoding="utf-8"))
    results = {item["dataset"]: item for item in registry["selected_results"]}
    assert registry["status"] == "verified"
    assert results["shrimpdb3"]["display_label"] == "ASL-LDAM + SimAM-DCFR"
    assert results["shrimpdb3"]["actual_source_method"] == "ASL-LDAM + SimAM-DCFR"
    assert results["shrimpdb3"]["label_matches_source_method"] is True
    assert results["combined4"]["display_label"] == "ASL-LDAM + SimAM-DCFR"
    assert results["combined4"]["actual_source_method"] == "CE Baseline"
    assert results["combined4"]["label_matches_source_method"] is False


def test_merged_registry_has_required_provenance_fields():
    registry = json.loads((ROOT / "artifacts/metadata/merged_result_registry.json").read_text(encoding="utf-8"))
    required = {
        "dataset", "display_label", "actual_source_method", "label_matches_source_method",
        "selection_metric", "tie_breaker", "accuracy", "balanced_accuracy", "macro_f1",
        "weighted_f1", "test_images", "source_checkpoint", "checkpoint_sha256",
        "provenance_file", "status",
    }
    for item in registry["selected_results"]:
        assert required <= item.keys()
        assert item["status"] == "verified"
