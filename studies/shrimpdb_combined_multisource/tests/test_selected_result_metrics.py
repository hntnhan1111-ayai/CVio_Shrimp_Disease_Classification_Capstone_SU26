import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
START_COMMIT = "8b37a176680d6efb5b07f9b4435801621563e63d"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_selected_metrics_match_imported_raw_metrics():
    final = json.loads((ROOT / "artifacts/results/FINAL_BEST_RESULTS.json").read_text(encoding="utf-8"))
    for dataset in ("shrimpdb3", "combined4"):
        selected = final["datasets"][dataset]
        raw = json.loads((ROOT / "artifacts/results" / dataset / "metrics_raw.json").read_text(encoding="utf-8"))
        assert selected["n"] == raw["n"]
        assert selected["accuracy"] == raw["accuracy"]
        assert selected["macro_f1"] == raw["macro_f1"]
        assert abs(selected["balanced_accuracy_percent"] / 100 - raw["balanced_accuracy"]) < 1e-15
        assert abs(selected["weighted_f1_percent"] / 100 - raw["weighted_f1"]) < 1e-15


def test_selected_values_are_the_expected_verified_values():
    final = json.loads((ROOT / "artifacts/results/FINAL_BEST_RESULTS.json").read_text(encoding="utf-8"))
    assert round(final["datasets"]["shrimpdb3"]["accuracy"] * 100, 2) == 91.49
    assert round(final["datasets"]["shrimpdb3"]["macro_f1"] * 100, 2) == 91.29
    assert round(final["datasets"]["combined4"]["accuracy"] * 100, 2) == 87.73
    assert round(final["datasets"]["combined4"]["macro_f1"] * 100, 2) == 86.51


def test_legacy_raw_metric_files_are_byte_identical_to_starting_commit():
    paths = (
        "studies/shrimpdb_combined_multisource/artifacts/evaluation/shrimpdb3/best_pt/metrics_raw.json",
        "studies/shrimpdb_combined_multisource/artifacts/evaluation/combined4/best_pt/metrics_raw.json",
    )
    repo = ROOT.parents[1]
    for relative in paths:
        before = subprocess.check_output(["git", "cat-file", "blob", f"{START_COMMIT}:{relative}"], cwd=repo)
        after = (repo / relative).read_bytes()
        assert digest_bytes(before) == digest_bytes(after), relative
