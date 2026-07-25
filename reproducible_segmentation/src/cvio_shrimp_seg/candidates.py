"""Catalog for the baseline and five report-selected candidates.

This module records provenance instead of silently reimplementing custom YOLO
layers from notebooks. Port a candidate only after preserving its listed source
behavior and adding a regression test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    identifier: str
    display_name: str
    family: str
    selection_reason: str
    historical_metrics: dict[str, float | None]
    legacy_sources: tuple[str, ...]
    new_dataset_notebooks: tuple[str, ...]
    port_status: str


BASELINE = Candidate(
    identifier="baseline_strong",
    display_name="YOLO11n-Seg baseline strong",
    family="matched control",
    selection_reason="Đối chứng bắt buộc để so sánh các cấu hình strong augmentation.",
    historical_metrics={
        "full_mask_map50": None,
        "disease_mask_map50": None,
        "disease_mask_map50_95": None,
        "healthy_fp_rate": None,
        "healthy_aware": None,
    },
    legacy_sources=(),
    new_dataset_notebooks=(
        "yolov11n_attention/new_combined_data/yolo11n_seg_baseline_strong_mrtu_grouped_near_duplicate_split.ipynb",
        "yolov11n_attention/new_data_leakage_safe_modules/yolo11n_seg_baseline_strong_new_data_leakage_safe.ipynb",
    ),
    port_status="catalog_only",
)


TOP_FIVE: tuple[Candidate, ...] = (
    Candidate(
        identifier="simam_ca_strong",
        display_name="SimAM + CA strong",
        family="attention + strong augmentation",
        selection_reason="Dẫn đầu disease mAP50, disease mAP50-95 và HA trong báo cáo.",
        historical_metrics={
            "full_mask_map50": 0.4971,
            "disease_mask_map50": 0.5507,
            "disease_mask_map50_95": 0.1964,
            "healthy_fp_rate": 0.341,
            "healthy_aware": 0.4808,
        },
        legacy_sources=(
            "yolov11n_attention/yolov11n_grouped_attention/yolo11n_seg_simam_ca_experiment (1).ipynb",
            "yolov11n_attention/yolov11n_simam_NhomA/augmentation/yolo11n_simam_augmentation_run_all_results_20260615_080026/reports/summary_all_models.csv",
        ),
        new_dataset_notebooks=(
            "yolov11n_attention/new_combined_data/yolo11n_seg_simam_ca_mrtu_grouped_near_duplicate_split.ipynb",
            "yolov11n_attention/new_data_leakage_safe_modules/yolo11n_seg_simam_ca_new_data_mixed_split.ipynb",
        ),
        port_status="not_ported",
    ),
    Candidate(
        identifier="lka_simam_head",
        display_name="LKA → SimAM head",
        family="mask-head attention",
        selection_reason="Dẫn đầu full-test mask mAP50 trong báo cáo.",
        historical_metrics={
            "full_mask_map50": 0.5003,
            "disease_mask_map50": 0.5249,
            "disease_mask_map50_95": 0.1812,
            "healthy_fp_rate": 0.366,
            "healthy_aware": 0.4518,
        },
        legacy_sources=(
            "yolov11n_attention/yolov11n_simam_NhomC/lka_head/yolov11n-simam-lka-head.ipynb",
            "yolov11n_attention/yolov11n_simam_NhomC/group_c_vs_baseline_comparison.csv",
        ),
        new_dataset_notebooks=(
            "yolov11n_attention/new_combined_data/yolo11n_seg_lka_simam_head_strong_mrtu_grouped_near_duplicate_split.ipynb",
            "yolov11n_attention/new_data_leakage_safe_modules/yolo11n_seg_lka_simam_head_strong_new_data_leakage_safe.ipynb",
        ),
        port_status="not_ported",
    ),
    Candidate(
        identifier="dpca_strong",
        display_name="DPCA strong",
        family="dual-polarity contrast attention",
        selection_reason="Ứng viên attention mới có cân bằng mAP/healthy-FP tốt trong báo cáo.",
        historical_metrics={
            "full_mask_map50": 0.4880,
            "disease_mask_map50": 0.5225,
            "disease_mask_map50_95": 0.1580,
            "healthy_fp_rate": 0.268,
            "healthy_aware": 0.4590,
        },
        legacy_sources=(
            "yolov11n_attention/new_research_attention/dpca_dual_polarity_contrast/dpca_dual_polarity_contrast.ipynb",
            "yolov11n_attention/new_research_attention/result/dpca-dual-polarity-contrast.ipynb",
        ),
        new_dataset_notebooks=(
            "yolov11n_attention/new_combined_data/yolo11n_seg_dpca_p4_strong_mrtu_grouped_near_duplicate_split.ipynb",
            "yolov11n_attention/new_combined_data/yolo11n_seg_dpca_p3p4p5_strong_mrtu_grouped_near_duplicate_split.ipynb",
            "yolov11n_attention/new_data_leakage_safe_modules/yolo11n_seg_dpca_p4_strong_new_data_leakage_safe.ipynb",
            "yolov11n_attention/new_data_leakage_safe_modules/yolo11n_seg_dpca_p3p4p5_strong_new_data_leakage_safe.ipynb",
        ),
        port_status="not_ported",
    ),
    Candidate(
        identifier="cote_boundarylite_p4_strong",
        display_name="CoTE + BoundaryLite P4 strong",
        family="P4 refinement attention",
        selection_reason="Cấu hình CoTE refinement mạnh nhất trong báo cáo.",
        historical_metrics={
            "full_mask_map50": 0.4921,
            "disease_mask_map50": 0.5174,
            "disease_mask_map50_95": 0.1660,
            "healthy_fp_rate": 0.341,
            "healthy_aware": 0.4540,
        },
        legacy_sources=(
            "yolov11n_attention/improving_labeled_test/cote_boundarylite_p4_strong.ipynb",
            "yolov11n_attention/custom_attention_research_modules/cote_bl_gate/generated_yamls/cote_bl_p4_strong.yaml",
        ),
        new_dataset_notebooks=(
            "yolov11n_attention/new_combined_data/yolo11n_seg_cote_boundarylite_p4_strong_mrtu_grouped_near_duplicate_split.ipynb",
            "yolov11n_attention/new_data_leakage_safe_modules/yolo11n_seg_cote_boundarylite_p4_strong_new_data_leakage_safe.ipynb",
        ),
        port_status="not_ported",
    ),
    Candidate(
        identifier="simam_ca_wiou_v3",
        display_name="SimAM + CA + WIoU v3",
        family="low-false-positive loss variant",
        selection_reason="Lựa chọn khi chi phí healthy false positive đặc biệt cao.",
        historical_metrics={
            "full_mask_map50": 0.4314,
            "disease_mask_map50": 0.4539,
            "disease_mask_map50_95": 0.1473,
            "healthy_fp_rate": 0.171,
            "healthy_aware": 0.3954,
        },
        legacy_sources=(
            "yolov11n_attention/yolov11n_simam_NhomB/yolov11n_simam_WIoUv3.ipynb",
            "yolov11n_attention/yolov11n_simam_NhomB/result/group_b_vs_baseline_comparison.csv",
        ),
        new_dataset_notebooks=(),
        port_status="missing_new_dataset_notebook",
    ),
)


def candidate_by_id(identifier: str) -> Candidate:
    for candidate in (BASELINE, *TOP_FIVE):
        if candidate.identifier == identifier:
            return candidate
    available = ", ".join(item.identifier for item in (BASELINE, *TOP_FIVE))
    raise KeyError(f"Unknown candidate '{identifier}'. Available: {available}")
