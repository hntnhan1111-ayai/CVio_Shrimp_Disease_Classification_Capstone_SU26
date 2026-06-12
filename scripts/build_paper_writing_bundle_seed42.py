#!/usr/bin/env python3
# scripts/build_paper_writing_bundle_seed42.py
# Copies core paper-writing materials into paper_writing_bundle/asl_ldam_simam_dcfr_yolo_seed42/.
# Deterministic, no training, no generated metrics. Filters out weights/raw datasets.

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEED = "asl_ldam_simam_dcfr_yolo_seed42"
BUNDLE = REPO / "paper_writing_bundle" / SEED
ARTIFACTS = REPO / "paper_artifacts" / SEED
EXPERIMENTS = REPO / "experiments" / SEED
PAPER = REPO / "paper" / SEED

FORBIDDEN_EXTS = {".pt", ".pth", ".onnx", ".engine", ".tflite", ".ckpt", ".safetensors", ".pkl", ".pickle", ".zip", ".rar", ".7z"}
FORBIDDEN_DIRS = {"weights", "checkpoints", "raw_dataset", "processed_images", "corrupted_test_sets", "top9_corrupted_test_sets"}


def is_forbidden(path: Path) -> bool:
    if any(part.lower() in FORBIDDEN_DIRS for part in path.parts):
        return True
    if any(path.suffix.lower() in FORBIDDEN_EXTS for _ in [path.suffix]):
        return False
    return False


def copy_file(src: Path, dst: Path):
    if src.exists() and not is_forbidden(src):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    dirs = [
        "title", "tables/dataset", "tables/baselines", "tables/improvements", "tables/noise", "tables/classwise",
        "tables/confusion_matrices", "tables/config", "tables/efficiency", "figures/samples", "figures/baselines",
        "figures/improvements", "figures/noise", "figures/training_curves", "figures/confusion_matrices",
        "xai/panels", "xai/metadata", "images/sample_no_bg", "source_evidence", "code_reference", "validation"
    ]
    for d in dirs:
        (BUNDLE / d).mkdir(parents=True, exist_ok=True)

    # title
    copy_file(PAPER / "paper_title_latex.tex", BUNDLE / "title/paper_title_latex.tex")
    copy_file(PAPER / "README_paper_scope.md", BUNDLE / "title/README_paper_scope.md")
    if not (BUNDLE / "title/paper_title_latex.tex").exists():
        (BUNDLE / "title/paper_title_latex.tex").write_text(
            "\\title{ASL-LDAM with SimAM-DCFR Attention for Robust YOLO-Based Shrimp Disease Image Classification Under Noisy Imaging Conditions}\n\\titlerunning{Robust YOLO-Based Shrimp Disease Classification}\n"
        )
    if not (BUNDLE / "title/README_paper_scope.md").exists():
        (BUNDLE / "title/README_paper_scope.md").write_text(
            "# Paper Scope\n\nRecommended title: ...\nBranch: paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp\nScope: Fixed seed-42 Stage-1 split only.\n\nMain artifacts: paper_artifacts/asl_ldam_simam_dcfr_yolo_seed42/\nExperiment code: experiments/asl_ldam_simam_dcfr_yolo_seed42/\n"
        )

    # tables
    tables = [
        "tables/dataset/dataset_split_distribution_table_seed42.csv",
        "tables/dataset/dataset_class_distribution_total_seed42.csv",
        "tables/baselines/yolo_baseline_comparison_seed42.csv",
        "tables/baselines/timm_baseline_comparison_seed42.csv",
        "tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv",
        "tables/improvements/yolo_best_method_vs_baseline_seed42.csv",
        "tables/improvements/timm_best_method_vs_baseline_seed42.csv",
        "tables/noise/top5_noise_selection_from_ce_baseline.csv",
        "tables/noise/top5_noise_baseline_vs_best_mean_summary.csv",
        "tables/noise/best_method_top5_noise_by_severity.csv",
        "tables/classwise/best_method_classwise_metrics_seed42.csv",
        "tables/classwise/best_method_classwise_overall_metrics_seed42.json",
        "tables/confusion_matrices/baseline_ce_clean_confusion_matrix.csv",
        "tables/confusion_matrices/best_method_clean_confusion_matrix.csv",
        "tables/confusion_matrices/best_method_impulse_noise_s3_confusion_matrix.csv",
        "tables/config/kaggle_reference_configuration_table.csv",
        "tables/efficiency/model_params_size_latency_fps_seed42.csv",
        "tables/efficiency/model_efficiency_summary_seed42.csv",
    ]
    for t in tables:
        copy_file(ARTIFACTS / t, BUNDLE / ("tables/" + t.split("/", 1)[1]))

    # figures
    figures = [
        "figures/samples/sample_no_bg_4class_panel.png",
        "figures/baselines/yolo_baseline_macro_f1_seed42.png",
        "figures/baselines/timm_baseline_macro_f1_seed42.png",
        "figures/improvements/yolo26m_key_result_macro_f1_seed42.png",
        "figures/improvements/yolo_delta_macro_f1_seed42.png",
        "figures/improvements/timm_delta_macro_f1_seed42.png",
        "figures/noise/top5_noise_baseline_vs_best_mean_macro_f1.png",
        "figures/noise/best_method_top5_noise_macro_f1_by_severity.png",
        "figures/training_curves/best_method_training_curves_top1_only.png",
        "figures/confusion_matrices/best_method_clean_confusion_matrix.png",
        "figures/confusion_matrices/baseline_ce_clean_confusion_matrix.png",
        "figures/confusion_matrices/best_method_impulse_noise_s3_confusion_matrix.png",
        "figures/xai/xai_best_method_4class_grid.png",
    ]
    for f in figures:
        rel = f.split("/", 1)[1]
        copy_file(ARTIFACTS / f, BUNDLE / ("figures/" + rel))
        if f == "figures/xai/xai_best_method_4class_grid.png":
            copy_file(ARTIFACTS / f, BUNDLE / "xai/xai_best_method_4class_grid.png")

    # xai
    for src_dir in ["panels", "metadata"]:
        for p in (ARTIFACTS / "xai" / src_dir).iterdir():
            copy_file(p, BUNDLE / f"xai/{src_dir}/{p.name}")

    # images
    for p in (ARTIFACTS / "images" / "sample_no_bg").iterdir():
        copy_file(p, BUNDLE / f"images/sample_no_bg/{p.name}")

    # source_evidence
    for p in (ARTIFACTS / "source_evidence").iterdir():
        copy_file(p, BUNDLE / f"source_evidence/{p.name}")
    copy_file(ARTIFACTS / "paper_ready_summary" / "CORRECTIONS_AND_SOURCES.md", BUNDLE / "source_evidence/CORRECTIONS_AND_SOURCES.md")

    # code_reference
    code_refs = [
        EXPERIMENTS / "final_loss_cbam_top5_noise_v5_stage1split_native_yolo_pack/README.md",
        EXPERIMENTS / "final_loss_cbam_top5_noise_v5_stage1split_native_yolo_pack/final_loss_cbam_top5_noise_stage1split_native_yolo_runner_v5.py",
        EXPERIMENTS / "final_loss_cbam_top5_noise_stage1split_timm_runner_v5.py",
        EXPERIMENTS / "collect_v5_yolo_timm_all_results.sh",
    ]
    for p in code_refs:
        copy_file(p, BUNDLE / f"code_reference/{p.name}")

    # validation
    validation = [
        ARTIFACTS / "PACKAGE_VALIDATION.json",
        ARTIFACTS / "PACKAGE_MANIFEST.csv",
        ARTIFACTS / "TITLE_REMOVAL_VALIDATION_REVIEWED.json",
        ARTIFACTS / "TITLE_REMOVAL_VALIDATION_FINAL.json",
        ARTIFACTS / "tables/baselines/coverage_yolo_seed42.csv",
        ARTIFACTS / "tables/baselines/coverage_timm_seed42.csv",
    ]
    for p in validation:
        copy_file(p, BUNDLE / f"validation/{p.name}")

    print(f"Bundle created at {BUNDLE}")


if __name__ == "__main__":
    main()
