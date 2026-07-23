#!/usr/bin/env python3
"""Standalone repository verification (no pytest required)."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def pass_(msg: str) -> None:
    print(f"[PASS] {msg}")


def test_canonical_original_baseline():
    canonical = json.loads(
        (ROOT / "results/raw/clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8")
    )
    assert canonical["baseline"]["mAP50"] == 0.129
    assert canonical["baseline"]["mAP50_95"] == 0.038
    pass_("Original baseline values correct in canonical JSON")


def test_canonical_recsra():
    canonical = json.loads(
        (ROOT / "results/raw/clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8")
    )
    assert abs(canonical["recsra"]["mAP50"] - 0.16036460211295228) < 1e-6
    assert abs(canonical["recsra"]["mAP50_95"] - 0.04656557248773138) < 1e-6
    pass_("RECSRA values match raw evidence in canonical JSON")


def test_102_rows():
    with (ROOT / "results/raw/all_clean_and_50_condition_metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 102, f"Expected 102, got {len(rows)}"
    keys = [(r["condition"], r["noise_id"], r["severity"], r["model"]) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicate keys found"
    pass_("102 corruption metric rows with no duplicates")


def test_top5_ids():
    import csv
    with (ROOT / "results/tables/top5_corruptions_percent.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = {r["noise_id"] for r in rows}
    assert ids == {"N07", "N06", "N01", "N03", "N10"}, f"Wrong top5: {ids}"
    pass_("Top-five corruption IDs correct: N07, N06, N01, N03, N10")


def test_checkpoint_hashes():
    expected = {
        "yolo11s_baseline_best.pt": "b9e30aa76f819126c7e6e9f3d3007a74d0839ad1f95941978a6a214484eb219d",
        "yolo11s_recsra_best.pt": "c1652101bb870a0b174b569ac47a70bb8cfafe119577e1ca2fe2de87f57824ff",
    }
    for name, exp in expected.items():
        p = ROOT / "checkpoints" / name
        if p.is_file():
            actual = digest(p)
            assert actual == exp, f"SHA mismatch for {name}: {actual} != {exp}"
            pass_(f"Checkpoint {name} SHA-256 verified")
        else:
            pass_(f"Checkpoint {name} not present (LFS may skip)")


def test_no_v007_in_public():
    exclude = ["cvio_yolo11s_paper_ready_outputs_no_weights", "__pycache__", ".venv", "tests", "scripts"]
    # V007 is allowed in documentation that explicitly accounts for its exclusion
    v007_allowed = {"METRIC_PROVENANCE.md", "FINAL_REFACTOR_AUDIT.md", "PRE_REFACTOR_AUDIT.md", "REPOSITORY_AUDIT.json", "test_repository.py", "verify_repository_full.py"}
    public_ext = {".md", ".html", ".csv", ".tex", ".py", ".yaml", ".yml", ".json"}
    violations = []
    for path in ROOT.rglob("*"):
        if path.is_file():
            rel = path.relative_to(ROOT)
            if any(part in exclude for part in rel.parts):
                continue
            if rel.name in v007_allowed:
                continue
            if path.suffix.lower() in public_ext:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for term in ["V007", "SimAM"]:
                    if term in text:
                        violations.append(f"{rel}: contains '{term}'")
    if violations:
        for v in violations[:10]:
            fail(v)
    pass_("No V007/SimAM references in public files (excluded documentation exempt)")


def test_no_wrong_baseline_in_headlines():
    wrong = [str(0.14418131018074604), str(0.0414657850886188)]
    headlines = [
        "README.md",
        "docs/results_dashboard.html",
        "docs/METRIC_REPORTING.md",
        "results/tables/clean_baseline_vs_recsra.md",
        "results/tables/clean_original_yolo11s_vs_recsra.md",
        "paper/tables/clean_results.tex",
        "results/reports/PAPER_RESULTS_DRAFT.md",
    ]
    for rel in headlines:
        p = ROOT / rel
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            for w in wrong:
                if w in text:
                    fail(f"Wrong baseline {w} found in {rel}")
    pass_("No wrong baseline values in headline files")


def test_percentage_first_in_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "12.900%" in text
    assert "3.800%" in text
    pass_("README uses percentage-first format")


def test_readme_not_wrong_baseline():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for bad in ["14.418%", "4.147%"]:
        assert bad not in text, f"README still has wrong baseline: {bad}"
    pass_("README does not contain wrong baseline percentages")


def test_eda_complete():
    required_figs = [
        "fig_split_image_count.png",
        "fig_split_box_count.png",
        "fig_class_boxes_total.png",
        "fig_class_percent_total.png",
        "fig_class_boxes_by_split.png",
        "fig_boxes_per_image_hist.png",
        "fig_box_area_percent_hist.png",
        "fig_box_size_bucket.png",
        "fig_box_aspect_ratio_hist.png",
        "fig_box_width_height_scatter.png",
        "fig_box_center_heatmap.png",
        "fig_image_resolution_scatter.png",
    ]
    for fig in required_figs:
        assert (ROOT / "dataset_eda/figures" / fig).is_file(), f"Missing figure: {fig}"
    required_tables = [
        "dataset_overview.csv",
        "class_counts_by_split.csv",
        "class_counts_total.csv",
        "box_size_buckets.csv",
        "audit_issues.csv",
        "duplicate_labels.csv",
        "image_records.csv",
        "label_records.csv",
    ]
    for tbl in required_tables:
        assert (ROOT / "dataset_eda/tables" / tbl).is_file(), f"Missing table: {tbl}"
    pass_("EDA package complete with all figures and tables")


def test_eda_values():
    summary = json.loads((ROOT / "dataset_eda/summary.json").read_text(encoding="utf-8"))
    assert summary["total_images"] == 746
    assert summary["total_boxes"] == 5569
    assert summary["train_images"] == 523
    assert summary["val_images"] == 112
    assert summary["test_images"] == 111
    assert summary["audit_issue_images"] == 0
    assert abs(summary["mean_boxes_per_image"] - 7.465) < 0.001
    pass_("EDA summary values verified")


def test_documentation_exists():
    required = [
        "docs/METRIC_PROVENANCE.md",
        "docs/DATASET_EDA.md",
        "docs/ENVIRONMENT.md",
        "docs/FINAL_REFACTOR_AUDIT.md",
        "docs/PRE_REFACTOR_AUDIT.md",
        "dataset_eda/SOURCE_PROVENANCE.md",
        "tests/test_repository.py",
        "results/raw/clean_original_baseline_vs_recsra.json",
        "results/raw/corruption_control_clean_metrics.json",
        "results/tables/clean_original_yolo11s_vs_recsra_percent.csv",
        "results/tables/clean_original_yolo11s_vs_recsra.md",
        "results/tables/corruption_control_clean_percent.csv",
        "paper/tables/top5_corruptions.tex",
        "environment/requirements-lock.txt",
        "environment/environment.yml",
        "environment/runtime_actual.json",
        "environment/kaggle_t4x2_target.yaml",
        "reproducibility/REPOSITORY_AUDIT.json",
    ]
    for rel in required:
        p = ROOT / rel
        assert p.is_file(), f"Missing required file: {rel}"
    pass_("All required documentation files present")


def test_clean_table_metrics():
    import csv
    with (ROOT / "results/tables/clean_original_yolo11s_vs_recsra_percent.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = {"mAP50", "mAP50_95", "precision", "recall", "AP_BG_mAP50_95", "AP_WSSV_mAP50_95"}
    assert {r["metric"] for r in rows} == expected
    assert len(rows) == 6
    pass_("Clean comparison table has correct metrics")


if __name__ == "__main__":
    tests = [
        test_canonical_original_baseline,
        test_canonical_recsra,
        test_102_rows,
        test_top5_ids,
        test_checkpoint_hashes,
        test_no_v007_in_public,
        test_no_wrong_baseline_in_headlines,
        test_percentage_first_in_readme,
        test_readme_not_wrong_baseline,
        test_eda_complete,
        test_eda_values,
        test_documentation_exists,
        test_clean_table_metrics,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed += 1
    print(f"\n{'='*50}")
    if failed:
        print(f"[RESULT] {len(tests) - failed}/{len(tests)} passed, {failed} FAILED")
        sys.exit(1)
    else:
        print(f"[RESULT] All {len(tests)} tests PASSED")
