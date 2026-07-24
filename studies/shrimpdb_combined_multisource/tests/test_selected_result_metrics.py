import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_selected_metrics_match_imported_raw_metrics():
    registry = json.loads((ROOT / "artifacts/metadata/merged_result_registry.json").read_text(encoding="utf-8"))
    for item in registry["selected_results"]:
        raw = json.loads((ROOT / "artifacts/merged_best_by_dataset/evaluation" / item["dataset"] / "metrics_raw.json").read_text(encoding="utf-8"))
        assert item["test_images"] == raw["n"]
        assert item["accuracy"] == raw["accuracy"]
        assert item["macro_f1"] == raw["macro_f1"]
        assert abs(item["balanced_accuracy"] - raw["balanced_accuracy"]) < 1e-15
        assert abs(item["weighted_f1"] - raw["weighted_f1"]) < 1e-15


def test_selected_values_are_the_expected_verified_values():
    registry = json.loads((ROOT / "artifacts/metadata/merged_result_registry.json").read_text(encoding="utf-8"))
    results = {item["dataset"]: item for item in registry["selected_results"]}
    assert round(results["shrimpdb3"]["accuracy"] * 100, 2) == 91.49
    assert round(results["shrimpdb3"]["macro_f1"] * 100, 2) == 91.29
    assert round(results["combined4"]["accuracy"] * 100, 2) == 87.73
    assert round(results["combined4"]["macro_f1"] * 100, 2) == 86.51
