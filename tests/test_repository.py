#!/usr/bin/env python3
"""Automated repository verification tests."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TestCanonicalBaseline:
    def test_original_baseline_map50(self):
        canonical = json.loads(
            (ROOT / "results/raw/clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8")
        )
        assert canonical["baseline"]["mAP50"] == 0.129

    def test_original_baseline_map50_95(self):
        canonical = json.loads(
            (ROOT / "results/raw/clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8")
        )
        assert canonical["baseline"]["mAP50_95"] == 0.038

    def test_recsra_map50_approx(self):
        canonical = json.loads(
            (ROOT / "results/raw/clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8")
        )
        assert abs(canonical["recsra"]["mAP50"] - 0.16036460211295228) < 1e-6

    def test_recsra_map50_95_approx(self):
        canonical = json.loads(
            (ROOT / "results/raw/clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8")
        )
        assert abs(canonical["recsra"]["mAP50_95"] - 0.04656557248773138) < 1e-6


class TestCorruptionData:
    def test_102_rows_exist(self):
        import pandas as pd
        metrics = pd.read_csv(ROOT / "results/raw/all_clean_and_50_condition_metrics.csv")
        assert len(metrics) == 102

    def test_no_duplicate_keys(self):
        import pandas as pd
        metrics = pd.read_csv(ROOT / "results/raw/all_clean_and_50_condition_metrics.csv")
        assert not metrics.duplicated(["condition", "noise_id", "severity", "model"]).any()

    def test_top5_ids(self):
        import pandas as pd
        top5 = pd.read_csv(ROOT / "results/tables/top5_corruptions_percent.csv")
        assert set(top5["noise_id"]) == {"N07", "N06", "N01", "N03", "N10"}

    def test_top5_five_rows(self):
        import pandas as pd
        top5 = pd.read_csv(ROOT / "results/tables/top5_corruptions_percent.csv")
        assert len(top5) == 5


class TestCheckpointHashes:
    EXPECTED = {
        "yolo11s_baseline_best.pt": "b9e30aa76f819126c7e6e9f3d3007a74d0839ad1f95941978a6a214484eb219d",
        "yolo11s_recsra_best.pt": "c1652101bb870a0b174b569ac47a70bb8cfafe119577e1ca2fe2de87f57824ff",
    }

    @pytest.mark.parametrize("name,expected", list(EXPECTED.items()))
    def test_checkpoint_sha256(self, name, expected):
        path = ROOT / "checkpoints" / name
        if path.is_file():
            assert _digest(path) == expected


class TestForbiddenReferences:
    PROHIBITED = ["V007", "SimAM"]

    @pytest.mark.parametrize("term", PROHIBITED)
    def test_no_v007_in_public_files(self, term):
        public_extensions = {".md", ".html", ".csv", ".tex", ".py", ".yaml", ".yml", ".json"}
        # Exclude the legacy archive directory
        exclude_dirs = {"cvio_yolo11s_paper_ready_outputs_no_weights", "__pycache__"}
        for path in ROOT.rglob("*"):
            if path.is_file():
                relative = path.relative_to(ROOT)
                # Skip excluded directories
                if any(part in exclude_dirs for part in relative.parts):
                    continue
                if path.suffix.lower() in public_extensions:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    # V007 is allowed ONLY in explicitly named corruption-control provenance files
                    if term == "V007" and "corruption_control_clean_metrics.json" in str(relative):
                        continue
                    assert term not in text, f"Prohibited term '{term}' found in {relative}"

    def test_no_wrong_baseline_in_headline_files(self):
        """Headline public files must not use 14.418% or 4.147% as the original baseline."""
        wrong_dec = [0.14418131018074604, 0.0414657850886188]
        headline_files = [
            "README.md",
            "docs/results_dashboard.html",
            "docs/METRIC_REPORTING.md",
            "results/tables/clean_baseline_vs_recsra.md",
            "results/tables/clean_original_yolo11s_vs_recsra.md",
            "paper/tables/clean_results.tex",
            "results/reports/PAPER_RESULTS_DRAFT.md",
        ]
        for relpath in headline_files:
            path = ROOT / relpath
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                for wrong in wrong_dec:
                    assert str(wrong) not in text, f"Wrong baseline {wrong} found in {relpath}"


class TestTableConsistency:
    def test_clean_table_six_metrics(self):
        import pandas as pd
        df = pd.read_csv(ROOT / "results/tables/clean_original_yolo11s_vs_recsra_percent.csv")
        assert len(df) == 6
        expected_metrics = {"mAP50", "mAP50_95", "precision", "recall", "AP_BG_mAP50_95", "AP_WSSV_mAP50_95"}
        assert set(df["metric"]) == expected_metrics


class TestPercentageFirst:
    def test_readme_uses_percentages(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "12.900%" in text
        assert "3.800%" in text
        assert "16.036%" in text
        assert "4.657%" in text


class TestEnvironmentFiles:
    def test_requirements_lock_exists(self):
        assert (ROOT / "environment/requirements-lock.txt").is_file()

    def test_environment_yml_exists(self):
        assert (ROOT / "environment/environment.yml").is_file()

    def test_runtime_actual_exists(self):
        assert (ROOT / "environment/runtime_actual.json").is_file()

    def test_kaggle_target_exists(self):
        assert (ROOT / "environment/kaggle_t4x2_target.yaml").is_file()


class TestEDAPackage:
    def test_summary_json_exists(self):
        assert (ROOT / "dataset_eda/summary.json").is_file()

    def test_all_figures_exist(self):
        required = [
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
        for fig in required:
            assert (ROOT / "dataset_eda/figures" / fig).is_file(), f"Missing figure: {fig}"

    def test_all_tables_exist(self):
        required = [
            "dataset_overview.csv",
            "class_counts_by_split.csv",
            "class_counts_total.csv",
            "box_size_buckets.csv",
            "class_cooccurrence_summary.csv",
            "class_cooccurrence_by_image.csv",
            "image_count_with_class.csv",
            "audit_issues.csv",
            "duplicate_labels.csv",
            "image_records.csv",
            "label_records.csv",
        ]
        for table in required:
            assert (ROOT / "dataset_eda/tables" / table).is_file(), f"Missing table: {table}"

    def test_eda_summary_values(self):
        import pandas as pd
        summary = json.loads((ROOT / "dataset_eda/summary.json").read_text(encoding="utf-8"))
        assert summary["total_images"] == 746
        assert summary["total_boxes"] == 5569
        assert summary["train_images"] == 523
        assert summary["val_images"] == 112
        assert summary["test_images"] == 111
        assert summary["audit_issue_images"] == 0
        assert abs(summary["mean_boxes_per_image"] - 7.465) < 0.001


class TestDocumentation:
    def test_metric_provenance_exists(self):
        assert (ROOT / "docs/METRIC_PROVENANCE.md").is_file()

    def test_dataset_eda_doc_exists(self):
        assert (ROOT / "docs/DATASET_EDA.md").is_file()

    def test_environment_doc_exists(self):
        assert (ROOT / "docs/ENVIRONMENT.md").is_file()
