import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_registry_is_external_and_hashes_are_preserved():
    with (ROOT / "artifacts/tables/checkpoint_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for item in rows:
        assert item["tracked_in_git"] == "false"
        assert len(item["checkpoint_sha256"]) == 64
        allowed_legacy = ROOT / "artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt"
        assert all(path == allowed_legacy for path in ROOT.rglob("*.pt")), "Unapproved checkpoints are present"


def test_imported_raw_metric_hashes_are_valid():
    for dataset in ("shrimpdb3", "combined4"):
        imported = ROOT / "artifacts/results" / dataset / "metrics_raw.json"
        assert len(sha256(imported)) == 64
