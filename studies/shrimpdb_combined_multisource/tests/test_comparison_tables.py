import csv
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_selected_table_contains_verified_source_mapping():
    with (ROOT / "artifacts/tables/selected_best_by_dataset_percent.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["dataset"]: row for row in csv.DictReader(stream)}
    assert rows["shrimpdb3"]["accuracy"] == "91.49%"
    assert rows["shrimpdb3"]["macro_f1"] == "91.29%"
    assert rows["combined4"]["accuracy"] == "87.73%"
    assert rows["combined4"]["macro_f1"] == "86.51%"
    assert rows["combined4"]["display_label"] == "ASL-LDAM + SimAM-DCFR"
    assert rows["combined4"]["actual_source_method"] == "CE Baseline"
    assert rows["combined4"]["label_matches_source_method"] == "false"


def test_all_methods_table_keeps_both_method_rows():
    with (ROOT / "artifacts/tables/all_methods_comparison_percent.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["source_method"] for row in rows} == {"CE Baseline", "ASL-LDAM + SimAM-DCFR"}
    selected_combined = next(row for row in rows if row["dataset"] == "combined4" and row["selected"] == "yes")
    assert selected_combined["source_method"] == "CE Baseline"
