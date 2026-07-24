import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_registry_is_external_and_hashes_are_preserved():
    registry = json.loads((ROOT / "model_registry/selected_best_by_dataset.json").read_text(encoding="utf-8"))
    for item in registry["selected_results"]:
        assert item["tracked_in_git"] is False
        assert len(item["checkpoint_sha256"]) == 64
        assert item["number_of_classes"] == len(item["class_order"])
        allowed_legacy = ROOT / "artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt"
        assert all(path == allowed_legacy for path in ROOT.rglob("*.pt")), "Unapproved checkpoints are present"


def test_imported_raw_metric_hashes_match_package_manifest():
    manifest = {}
    with (ROOT / "artifacts/merged_best_by_dataset/metadata/MANIFEST_SHA256.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            manifest[row["path"].replace("\\", "/")] = row["sha256"]
    for dataset in ("shrimpdb3", "combined4"):
        relative = f"selected_best_by_dataset/{dataset}/evaluation/metrics_raw.json"
        imported = ROOT / "artifacts/merged_best_by_dataset/evaluation" / dataset / "metrics_raw.json"
        assert sha256(imported) == manifest[relative]
