"""Final aggregation, paper tables, figures, reports, and zip packaging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .evaluate import collect_run_outputs
from .progress import log_event
from .utils import ensure_dir, environment_versions, read_json, save_csv, write_json, zip_directory
from .xai import generate_selected_xai


def expected_all_run_ids(output_dir: str | Path | None = None) -> set[str]:
    from .models_torch import list_core_torch_runs, list_lightweight_diagnostic_runs, list_lightweight_runs
    from .models_yolo import list_core_yolo_runs, list_yolo_family_runs

    rows = list_core_torch_runs() + list_core_yolo_runs() + list_yolo_family_runs() + list_lightweight_runs()
    expected = {row["run_id"] for row in rows}
    if output_dir is not None:
        runs_dir = Path(output_dir) / "runs"
        for row in list_lightweight_diagnostic_runs():
            if (runs_dir / row["run_id"]).exists():
                expected.add(row["run_id"])
    return expected


def make_tables(output_dir: str | Path, metrics_frame: pd.DataFrame, xai_frame: pd.DataFrame) -> dict[str, Path]:
    output_dir = Path(output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    paths: dict[str, Path] = {}
    split_path = output_dir / "fixed_split_manifest_seed42_with_md5.csv"
    if split_path.exists():
        split = pd.read_csv(split_path)
        table1 = split.groupby(["class_name", "split"]).size().unstack(fill_value=0).reset_index()
        paths["dataset_split"] = save_csv(table1, reports_dir / "table1_dataset_split.csv")
    core_models = {config.CONVNEXT_CORE_MODEL_NAME, config.YOLO_CORE_MODEL}
    if not metrics_frame.empty:
        paths["core_ablation"] = save_csv(metrics_frame[metrics_frame["model"].isin(core_models)].copy(), reports_dir / "table2_core_ablation.csv")
        paths["yolo_family"] = save_csv(metrics_frame[metrics_frame["backend"] == "ultralytics"].copy(), reports_dir / "table3_yolo_family.csv")
        paths["lightweight"] = save_csv(metrics_frame[metrics_frame["backend"].isin(["timm", "torchvision"])].copy(), reports_dir / "table4_lightweight_models.csv")
        paths["deployment_candidates"] = save_csv(metrics_frame.sort_values(["test_macro_f1", "latency_ms_image", "model_size_mb"], ascending=[False, True, True]).copy(), reports_dir / "table5_selected_candidates.csv")
    paths["xai"] = save_csv(xai_frame, reports_dir / "table6_xai_status.csv")
    return paths


def make_figures(output_dir: str | Path, metrics_frame: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = ensure_dir(Path(output_dir) / "figures")
    outputs: list[Path] = []
    if metrics_frame.empty:
        return outputs
    macro = metrics_frame.sort_values("test_macro_f1", ascending=False).head(30)
    plt.figure(figsize=(12, 5))
    plt.bar(macro["run_id"].astype(str), macro["test_macro_f1"].astype(float))
    plt.xticks(rotation=80, ha="right", fontsize=6)
    plt.ylabel("Test Macro-F1")
    plt.tight_layout()
    out = figures_dir / "figure1_test_macro_f1.png"
    plt.savefig(out, dpi=300)
    plt.close()
    outputs.append(out)
    if {"latency_ms_image", "test_macro_f1"}.issubset(metrics_frame.columns):
        plt.figure(figsize=(7, 5))
        plt.scatter(metrics_frame["latency_ms_image"].astype(float), metrics_frame["test_macro_f1"].astype(float))
        plt.xlabel("Latency ms/image")
        plt.ylabel("Test Macro-F1")
        plt.tight_layout()
        out = figures_dir / "figure2_macro_f1_latency.png"
        plt.savefig(out, dpi=300)
        plt.close()
        outputs.append(out)
    error_cols = [col for col in ["BG->WSSV_BG", "WSSV->WSSV_BG", "WSSV_BG->BG", "WSSV_BG->WSSV"] if col in metrics_frame.columns]
    if error_cols:
        plt.figure(figsize=(12, 5))
        metrics_frame.set_index("run_id")[error_cols].astype(float).head(30).plot(kind="bar", ax=plt.gca())
        plt.ylabel("Error count")
        plt.xticks(rotation=80, ha="right", fontsize=6)
        plt.tight_layout()
        out = figures_dir / "figure3_coinfection_errors.png"
        plt.savefig(out, dpi=300)
        plt.close()
        outputs.append(out)
    return outputs


def write_excel_summary(output_dir: str | Path, table_paths: dict[str, Path], metrics_frame: pd.DataFrame) -> Path:
    output_dir = Path(output_dir)
    out = output_dir / "final_summary.xlsx"
    with pd.ExcelWriter(out) as writer:
        metrics_frame.to_excel(writer, sheet_name="final_summary", index=False)
        for name, path in table_paths.items():
            frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return out


def write_reproducibility_report(output_dir: str | Path, metrics_frame: pd.DataFrame, xai_frame: pd.DataFrame) -> Path:
    output_dir = Path(output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    completed = int((metrics_frame["status"] == "completed").sum()) if "status" in metrics_frame.columns else 0
    lines = [
        "# Reproducibility Report",
        "",
        f"Project: {config.PROJECT_TITLE}",
        f"Dataset: {config.DATASET_ID}",
        f"Runtime target: {config.RUNTIME_TARGET}",
        f"Python target: {config.PYTHON_TARGET}",
        f"Seed: {config.SEED}",
        f"Split seed: {config.SPLIT_SEED}",
        f"Completed runs in this output directory: {completed}",
        "",
        "ASLSingleLabel is an existing ASL baseline, not a custom contribution.",
        "PairwiseCoInfectionRankingASL is treated as a dataset-specific co-infection-aware ASL variant.",
        "YOLO runs use native Ultralytics classification training and Ultralytics auto_augment when available.",
        "These scripts generate classification experiment artifacts only.",
        "",
        f"XAI rows recorded: {len(xai_frame)}",
    ]
    path = reports_dir / "reproducibility_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_reports(output_dir: str | Path, dry_run: bool = False, progress_enabled: bool = True) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "reports")
    ensure_dir(output_dir / "figures")
    if dry_run:
        if progress_enabled:
            log_event("Report generation dry-run requested.", output_dir=output_dir)
        return {
            "dry_run": True,
            "expected_inputs": [
                "fixed_split_manifest_seed42_with_md5.csv",
                "yolo_split_manifest_seed42_with_md5.csv",
                "runs/*/metrics.json",
            ],
            "planned_outputs": [
                "final_summary.csv",
                "final_summary.xlsx",
                "reports/reproducibility_report.md",
                "paper_outputs.zip",
            ],
        }
    if progress_enabled:
        log_event("Writing environment versions.", output_dir=output_dir)
    write_json(output_dir / "environment_versions.json", environment_versions({
        "project": config.PROJECT_TITLE,
        "dataset_id": config.DATASET_ID,
        "seed": config.SEED,
    }))
    if progress_enabled:
        log_event("Collecting completed run outputs.", output_dir=output_dir)
    metrics_frame, predictions, confusions, failed = collect_run_outputs(output_dir, expected_run_ids=expected_all_run_ids(output_dir))
    if progress_enabled:
        log_event("Run output collection completed.", output_dir=output_dir, extra={"metrics_rows": len(metrics_frame), "prediction_rows": len(predictions), "confusion_rows": len(confusions), "failed_rows": len(failed)})
    split_manifest = pd.read_csv(output_dir / "fixed_split_manifest_seed42_with_md5.csv") if (output_dir / "fixed_split_manifest_seed42_with_md5.csv").exists() else None
    yolo_manifest = pd.read_csv(output_dir / "yolo_split_manifest_seed42_with_md5.csv") if (output_dir / "yolo_split_manifest_seed42_with_md5.csv").exists() else None
    xai_frame = generate_selected_xai(output_dir, metrics_frame, split_manifest=split_manifest, yolo_manifest=yolo_manifest, progress_enabled=progress_enabled)
    if progress_enabled:
        log_event("Building paper tables.", output_dir=output_dir)
    table_paths = make_tables(output_dir, metrics_frame, xai_frame)
    if progress_enabled:
        log_event("Building paper figures.", output_dir=output_dir)
    figure_paths = make_figures(output_dir, metrics_frame)
    if progress_enabled:
        log_event("Writing Excel summary.", output_dir=output_dir)
    excel_path = write_excel_summary(output_dir, table_paths, metrics_frame)
    if progress_enabled:
        log_event("Writing reproducibility report.", output_dir=output_dir)
    report_path = write_reproducibility_report(output_dir, metrics_frame, xai_frame)
    if progress_enabled:
        log_event("Creating downloadable zip package.", output_dir=output_dir)
    zip_path = zip_directory(output_dir, output_dir / "paper_outputs.zip", exclude_names={"paper_outputs.zip"})
    write_json(output_dir / "report_generation_summary.json", {
        "metrics_rows": len(metrics_frame),
        "prediction_rows": len(predictions),
        "confusion_rows": len(confusions),
        "failed_rows": len(failed),
        "xai_rows": len(xai_frame),
        "tables": {k: str(v) for k, v in table_paths.items()},
        "figures": [str(path) for path in figure_paths],
        "excel": str(excel_path),
        "report": str(report_path),
        "zip": str(zip_path),
    })
    if progress_enabled:
        log_event("Report generation completed.", output_dir=output_dir, extra={"final_summary_rows": len(metrics_frame), "zip": str(zip_path), "report": str(report_path)})
    return {"final_summary_rows": len(metrics_frame), "zip": str(zip_path), "report": str(report_path)}
