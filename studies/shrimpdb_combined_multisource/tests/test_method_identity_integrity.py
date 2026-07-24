import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_provenance_files_agree_with_registry():
    registry = json.loads((ROOT / "artifacts/metadata/merged_result_registry.json").read_text(encoding="utf-8"))
    for item in registry["selected_results"]:
        provenance = json.loads((ROOT / item["provenance_file"]).read_text(encoding="utf-8"))
        for field in ("display_label", "actual_source_method", "label_matches_source_method", "checkpoint_sha256"):
            assert provenance[field] == item[field]


def test_identity_audit_is_preserved_and_referenced():
    audit = ROOT / "artifacts/metadata/method_identity_audit.json"
    assert audit.is_file()
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] in {"verified", "contradictory"}
    registry = json.loads((ROOT / "artifacts/metadata/merged_result_registry.json").read_text(encoding="utf-8"))
    assert registry["prior_method_identity_audit"] == "artifacts/metadata/method_identity_audit.json"
