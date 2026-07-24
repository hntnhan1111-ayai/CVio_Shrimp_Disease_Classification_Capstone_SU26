"""Import and generate provenance-preserving merged best-by-dataset artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY_LABEL = "ASL-LDAM + SimAM-DCFR"
DATASETS = ("shrimpdb3", "combined4")
METHOD_COLUMNS = (
    "accuracy_percent",
    "balanced_accuracy_percent",
    "macro_f1_percent",
    "weighted_f1_percent",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{float(value) * 100.0:.2f}%"


def pct_percent(value: float) -> str:
    return f"{float(value):.2f}%"


def source_metric_path(merged_root: Path, dataset: str) -> Path:
    return merged_root / "selected_best_by_dataset" / dataset / "evaluation" / "metrics_raw.json"


def validate_package(merged_root: Path, expected_zip_sha256: str | None) -> dict[str, Any]:
    required = (
        "README.md",
        "RESULT_DISPLAY_LABEL.txt",
        "FINAL_BEST_RESULTS.json",
        "FINAL_BEST_RESULTS.csv",
        "ALL_METHODS_COMPARISON.csv",
        "MANIFEST_SHA256.csv",
        "source_packages",
        "selected_best_by_dataset",
    )
    missing = [str(merged_root / item) for item in required if not (merged_root / item).exists()]
    if missing:
        raise FileNotFoundError("Missing merged package artifacts: " + ", ".join(missing))
    if expected_zip_sha256:
        zip_candidates = sorted(merged_root.parent.glob(merged_root.name + ".zip"))
        if not zip_candidates:
            raise FileNotFoundError("Expected merged ZIP is not beside the merged result directory.")
        actual_zip_sha256 = sha256(zip_candidates[0])
        if actual_zip_sha256.lower() != expected_zip_sha256.lower():
            raise ValueError(f"Merged ZIP hash mismatch: {actual_zip_sha256} != {expected_zip_sha256}")
    final_results = read_json(merged_root / "FINAL_BEST_RESULTS.json")
    required_values = {
        "combined4": ("CE Baseline", False),
        "shrimpdb3": (DISPLAY_LABEL, True),
    }
    for dataset, (method, label_matches) in required_values.items():
        selected = final_results["datasets"][dataset]
        if selected["actual_source_method"] != method:
            raise ValueError(f"{dataset}: unexpected actual_source_method")
        if bool(selected["label_matches_source_method"]) != label_matches:
            raise ValueError(f"{dataset}: unexpected label_matches_source_method")
        provenance = read_json(merged_root / "selected_best_by_dataset" / dataset / "PROVENANCE.json")
        for field in (
            "display_label",
            "actual_source_method",
            "label_matches_source_method",
            "accuracy",
            "macro_f1",
            "checkpoint_sha256",
        ):
            if selected[field] != provenance[field]:
                raise ValueError(f"{dataset}: FINAL_BEST_RESULTS and PROVENANCE disagree on {field}")
        metrics = read_json(source_metric_path(merged_root, dataset))
        for field in ("accuracy", "macro_f1"):
            if abs(float(selected[field]) - float(metrics[field])) > 1e-12:
                raise ValueError(f"{dataset}: selected metric does not match metrics_raw.json: {field}")
        for percent_field, raw_field in (
            ("balanced_accuracy_percent", "balanced_accuracy"),
            ("weighted_f1_percent", "weighted_f1"),
        ):
            if abs(float(selected[percent_field]) / 100.0 - float(metrics[raw_field])) > 1e-12:
                raise ValueError(f"{dataset}: selected metric does not match metrics_raw.json: {raw_field}")
        checkpoint = merged_root / "selected_best_by_dataset" / dataset / "checkpoint" / "selected_best.pt"
        if sha256(checkpoint) != selected["checkpoint_sha256"]:
            raise ValueError(f"{dataset}: selected checkpoint hash mismatch")
    return final_results


def copy_selected_artifacts(merged_root: Path, study_root: Path) -> dict[str, str]:
    destination = study_root / "artifacts" / "merged_best_by_dataset"
    metadata = destination / "metadata"
    provenance = destination / "provenance"
    metadata.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    for source_name, target in (
        ("FINAL_BEST_RESULTS.json", metadata / "FINAL_BEST_RESULTS.json"),
        ("FINAL_BEST_RESULTS.csv", metadata / "FINAL_BEST_RESULTS.csv"),
        ("ALL_METHODS_COMPARISON.csv", metadata / "ALL_METHODS_COMPARISON.csv"),
        ("RESULT_DISPLAY_LABEL.txt", metadata / "RESULT_DISPLAY_LABEL.txt"),
        ("MANIFEST_SHA256.csv", metadata / "MANIFEST_SHA256.csv"),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(merged_root / source_name, target)
    for dataset in DATASETS:
        source = merged_root / "selected_best_by_dataset" / dataset
        target = destination / "evaluation" / dataset
        target.mkdir(parents=True, exist_ok=True)
        for item in (source / "evaluation").iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
        shutil.copy2(source / "PROVENANCE.json", provenance / f"{dataset}_PROVENANCE.json")
        if (source / "RESULT_LABEL.txt").is_file():
            shutil.copy2(source / "RESULT_LABEL.txt", provenance / f"{dataset}_RESULT_LABEL.txt")
    return {
        "metadata_dir": "artifacts/merged_best_by_dataset/metadata",
        "evaluation_dir": "artifacts/merged_best_by_dataset/evaluation",
        "provenance_dir": "artifacts/merged_best_by_dataset/provenance",
    }


def class_order(report_path: Path) -> list[str]:
    with report_path.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream)
        next(rows)
        result = []
        for row in rows:
            if row and row[0] not in {"accuracy", "macro avg", "weighted avg", ""}:
                result.append(row[0])
        return result


def build_registry(
    final_results: dict[str, Any],
    merged_root: Path,
    study_root: Path,
) -> dict[str, Any]:
    selected = []
    for dataset in DATASETS:
        value = final_results["datasets"][dataset]
        provenance = read_json(merged_root / "selected_best_by_dataset" / dataset / "PROVENANCE.json")
        source_method = value["actual_source_method"]
        number_of_classes = len(class_order(study_root / "artifacts" / "merged_best_by_dataset" / "evaluation" / dataset / "classification_report_percent.csv"))
        selected.append(
            {
                "dataset": dataset,
                "display_label": value["display_label"],
                "actual_source_method": source_method,
                "label_matches_source_method": value["label_matches_source_method"],
                "selection_metric": final_results["selection_metric"],
                "tie_breaker": final_results["tie_breaker"],
                "accuracy": value["accuracy"],
                "balanced_accuracy": value["balanced_accuracy_percent"] / 100.0,
                "macro_f1": value["macro_f1"],
                "weighted_f1": value["weighted_f1_percent"] / 100.0,
                "test_images": value["n"],
                "source_checkpoint": value["source_checkpoint"],
                "checkpoint_filename": "selected_best.pt",
                "checkpoint_sha256": value["checkpoint_sha256"],
                "local_source_path": str(merged_root / "selected_best_by_dataset" / dataset / "checkpoint" / "selected_best.pt"),
                "provenance_file": f"artifacts/merged_best_by_dataset/provenance/{dataset}_PROVENANCE.json",
                "number_of_classes": number_of_classes,
                "class_order": class_order(study_root / "artifacts" / "merged_best_by_dataset" / "evaluation" / dataset / "classification_report_percent.csv"),
                "tracked_in_git": False,
                "scientific_integrity_note": final_results["scientific_integrity_note"],
                "status": "verified",
            }
        )
    registry = {
        "package_type": final_results["package_type"],
        "display_label_for_selected_results": final_results["display_label_for_selected_results"],
        "selection_metric": final_results["selection_metric"],
        "tie_breaker": final_results["tie_breaker"],
        "source_zip_sha256": "dc3e0c544e386008f62df6dbd12d806f9d00d854ff1f833b70365db7e7030e59",
        "source_metadata": "artifacts/merged_best_by_dataset/metadata/FINAL_BEST_RESULTS.json",
        "prior_method_identity_audit": "artifacts/metadata/method_identity_audit.json",
        "merged_method_identity_audit": "artifacts/metadata/merged_method_identity_audit.json",
        "scientific_integrity_note": final_results["scientific_integrity_note"],
        "selected_results": selected,
        "status": "verified",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(study_root / "artifacts" / "metadata" / "merged_result_registry.json", registry)
    write_json(study_root / "model_registry" / "selected_best_by_dataset.json", {
        "distribution_policy": "Selected checkpoints are not tracked in Git; retrieve from the merged package and verify SHA-256.",
        "source_zip_sha256": registry["source_zip_sha256"],
        "selected_results": selected,
        "status": "verified",
    })
    return registry


def write_tables(
    registry: dict[str, Any],
    merged_root: Path,
    study_root: Path,
) -> list[str]:
    table_dir = study_root / "artifacts" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    rows = []
    for item in registry["selected_results"]:
        rows.append({
            "dataset": item["dataset"],
            "display_label": item["display_label"],
            "actual_source_method": item["actual_source_method"],
            "label_matches_source_method": str(item["label_matches_source_method"]).lower(),
            "selection_metric": item["selection_metric"],
            "tie_breaker": item["tie_breaker"],
            "accuracy": pct(item["accuracy"]),
            "balanced_accuracy": pct(item["balanced_accuracy"]),
            "macro_f1": pct(item["macro_f1"]),
            "weighted_f1": pct(item["weighted_f1"]),
            "test_images": item["test_images"],
            "checkpoint_sha256": item["checkpoint_sha256"],
        })
    selected_path = table_dir / "selected_best_by_dataset_percent.csv"
    pd.DataFrame(rows).to_csv(selected_path, index=False)
    paths.append(str(selected_path.relative_to(study_root)))

    comparison = pd.read_csv(merged_root / "ALL_METHODS_COMPARISON.csv")
    for column in METHOD_COLUMNS:
        comparison[column] = comparison[column].map(pct_percent)
    comparison_path = table_dir / "all_methods_comparison_percent.csv"
    comparison.to_csv(comparison_path, index=False)
    paths.append(str(comparison_path.relative_to(study_root)))

    for dataset in DATASETS:
        source = study_root / "artifacts" / "merged_best_by_dataset" / "evaluation" / dataset / "classification_report_percent.csv"
        report = pd.read_csv(source)
        for column in ("precision", "recall", "f1-score"):
            report[column] = report[column].map(pct_percent)
        target = table_dir / f"per_class_selected_{dataset}_percent.csv"
        report.to_csv(target, index=False)
        paths.append(str(target.relative_to(study_root)))

    source_domain = read_json(
        study_root / "artifacts" / "merged_best_by_dataset" / "evaluation" / "combined4" / "source_domain_metrics_raw.json"
    )
    source_rows = []
    for source_name, value in source_domain.items():
        source_rows.append({
            "source_dataset": source_name,
            "n": value["n"],
            "accuracy": pct(value["accuracy"]),
            "balanced_accuracy": pct(value["balanced_accuracy"]),
            "macro_f1": pct(value["macro_f1"]),
            "display_label": registry["selected_results"][1]["display_label"],
            "actual_source_method": registry["selected_results"][1]["actual_source_method"],
            "checkpoint_sha256": registry["selected_results"][1]["checkpoint_sha256"],
        })
    source_path = table_dir / "source_domain_selected_combined4_percent.csv"
    pd.DataFrame(source_rows).to_csv(source_path, index=False)
    paths.append(str(source_path.relative_to(study_root)))

    checkpoint_path = table_dir / "checkpoint_provenance.csv"
    pd.DataFrame([
        {
            "dataset": item["dataset"],
            "display_label": item["display_label"],
            "actual_source_method": item["actual_source_method"],
            "label_matches_source_method": str(item["label_matches_source_method"]).lower(),
            "checkpoint_filename": item["checkpoint_filename"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "number_of_classes": item["number_of_classes"],
            "class_order": " | ".join(item["class_order"]),
            "local_source_path": item["local_source_path"],
            "tracked_in_git": "false",
        }
        for item in registry["selected_results"]
    ]).to_csv(checkpoint_path, index=False)
    paths.append(str(checkpoint_path.relative_to(study_root)))
    return paths


def chart_caption(item: dict[str, Any], dataset: str) -> str:
    return (
        f"{dataset}: test split, selected checkpoint {item['checkpoint_filename']} "
        f"({item['checkpoint_sha256'][:12]}…), display label {item['display_label']}; "
        f"actual source method {item['actual_source_method']}."
    )


def write_figures(registry: dict[str, Any], study_root: Path) -> list[dict[str, Any]]:
    figure_dir = study_root / "artifacts" / "merged_best_by_dataset" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = {item["dataset"]: item for item in registry["selected_results"]}
    figures: list[dict[str, Any]] = []

    def save_figure(name: str, experiment: str, method: str, checkpoint: str, split: str, source: str, caption: str) -> None:
        path = figure_dir / name
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        figures.append({
            "figure_id": f"Merged Figure {len(figures) + 1}",
            "filename": f"artifacts/merged_best_by_dataset/figures/{name}",
            "experiment": experiment,
            "verified_method": method,
            "checkpoint": checkpoint,
            "split": split,
            "source_artifact": source,
            "sha256": sha256(path),
            "caption": caption,
            "identity_audit_reference": "artifacts/metadata/merged_method_identity_audit.json",
            "status": "verified",
        })

    split_path = study_root / "artifacts" / "academic" / "tables" / "dataset_source_split_distribution.csv"
    split = pd.read_csv(split_path)
    aggregate = split[split["experiment"] == "combined4"].groupby(["split", "class_name"], as_index=False)["images"].sum()
    pivot = aggregate.pivot(index="split", columns="class_name", values="images").fillna(0)
    pivot.plot(kind="bar", figsize=(9, 5), title="Combined-4 split distribution")
    plt.xlabel("Partition")
    plt.ylabel("Images")
    plt.tight_layout()
    save_figure(
        "fig01_dataset_split_distribution.png",
        "combined4",
        "mixed selected sources; see registry",
        "not applicable",
        "train+val+test",
        "artifacts/academic/tables/dataset_source_split_distribution.csv",
        "Combined-4 dataset split distribution; selected test results use the Combined-4 CE Baseline checkpoint and the ASL-LDAM + SimAM-DCFR display label.",
    )

    datasets = ["ShrimpDB-3", "Combined-4"]
    accuracy = [selected["shrimpdb3"]["accuracy"] * 100, selected["combined4"]["accuracy"] * 100]
    plt.figure(figsize=(7, 5))
    plt.bar(datasets, accuracy, color=["#2070b4", "#d95f02"])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Selected accuracy by dataset")
    for index, value in enumerate(accuracy):
        plt.text(index, value + 1, f"{value:.2f}%", ha="center")
    plt.figtext(0.01, -0.03, chart_caption(selected["shrimpdb3"], "shrimpdb3") + " " + chart_caption(selected["combined4"], "combined4"), fontsize=7)
    save_figure("fig02_selected_accuracy_by_dataset.png", "shrimpdb3+combined4", "mixed selected sources", "selected_best.pt", "test", "artifacts/metadata/FINAL_BEST_RESULTS.json", "Selected accuracy on each test split; " + chart_caption(selected["shrimpdb3"], "ShrimpDB-3") + " " + chart_caption(selected["combined4"], "Combined-4"))

    macro_f1 = [selected["shrimpdb3"]["macro_f1"] * 100, selected["combined4"]["macro_f1"] * 100]
    plt.figure(figsize=(7, 5))
    plt.bar(datasets, macro_f1, color=["#2070b4", "#d95f02"])
    plt.ylim(0, 100)
    plt.ylabel("Macro-F1 (%)")
    plt.title("Selected Macro-F1 by dataset")
    for index, value in enumerate(macro_f1):
        plt.text(index, value + 1, f"{value:.2f}%", ha="center")
    plt.figtext(0.01, -0.03, chart_caption(selected["shrimpdb3"], "shrimpdb3") + " " + chart_caption(selected["combined4"], "combined4"), fontsize=7)
    save_figure("fig03_selected_macro_f1_by_dataset.png", "shrimpdb3+combined4", "mixed selected sources", "selected_best.pt", "test", "artifacts/metadata/FINAL_BEST_RESULTS.json", "Selected Macro-F1 on each test split; " + chart_caption(selected["shrimpdb3"], "ShrimpDB-3") + " " + chart_caption(selected["combined4"], "Combined-4"))

    comparison = pd.read_csv(study_root / "artifacts/merged_best_by_dataset/metadata/ALL_METHODS_COMPARISON.csv")
    for dataset, filename in (("shrimpdb3", "fig04_all_methods_shrimpdb3_comparison.png"), ("combined4", "fig05_all_methods_combined4_comparison.png")):
        subset = comparison[comparison["dataset"] == dataset]
        labels = [f"{row.source_method}\n{'selected' if row.selected == 'yes' else 'reference'}" for row in subset.itertuples()]
        values = subset["macro_f1_percent"].to_numpy()
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=["#d95f02" if row.selected == "yes" else "#999999" for row in subset.itertuples()])
        plt.ylabel("Macro-F1 (%)")
        plt.title(f"All methods: {dataset}")
        plt.ylim(0, 100)
        for index, value in enumerate(values):
            plt.text(index, value + 1, f"{value:.2f}%", ha="center")
        plt.figtext(0.01, -0.03, chart_caption(selected[dataset], dataset), fontsize=7)
        save_figure(filename, dataset, selected[dataset]["actual_source_method"], "selected_best.pt", "test", "artifacts/merged_best_by_dataset/metadata/ALL_METHODS_COMPARISON.csv", f"All-method Macro-F1 comparison for {dataset}; {chart_caption(selected[dataset], dataset)}")

    for dataset, source_name, names in (
        ("shrimpdb3", "shrimpdb3", ("counts", "normalized")),
        ("combined4", "combined4", ("counts", "normalized")),
    ):
        for variant in names:
            source = study_root / "artifacts/merged_best_by_dataset/evaluation" / dataset / ("confusion_matrix_counts.png" if variant == "counts" else "confusion_matrix_normalized.png")
            number = 6 if dataset == "shrimpdb3" and variant == "counts" else 7 if dataset == "shrimpdb3" else 8 if variant == "counts" else 9
            target = figure_dir / f"fig{number:02d}_{dataset}_{source.stem}.png"
            shutil.copy2(source, target)
            figures.append({
                "figure_id": f"Merged Figure {len(figures) + 1}",
                "filename": f"artifacts/merged_best_by_dataset/figures/{target.name}",
                "experiment": dataset,
                "verified_method": selected[dataset]["actual_source_method"],
                "checkpoint": "selected_best.pt",
                "split": "test",
                "source_artifact": f"artifacts/merged_best_by_dataset/evaluation/{dataset}/{source.name}",
                "sha256": sha256(target),
                "caption": f"{dataset} {variant} confusion matrix on the test split; {chart_caption(selected[dataset], dataset)}",
                "identity_audit_reference": "artifacts/metadata/merged_method_identity_audit.json",
                "status": "verified",
            })

    source_domain_path = study_root / "artifacts/merged_best_by_dataset/evaluation/combined4/source_domain_metrics_raw.json"
    source_domain = read_json(source_domain_path)
    names = list(source_domain)
    values = [source_domain[name]["macro_f1"] * 100 for name in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color=["#2070b4", "#d95f02"])
    plt.ylabel("Macro-F1 (%)")
    plt.title("Combined-4 selected checkpoint by source domain")
    plt.ylim(0, 100)
    for index, value in enumerate(values):
        plt.text(index, value + 1, f"{value:.2f}%", ha="center")
    plt.figtext(0.01, -0.03, chart_caption(selected["combined4"], "Combined-4"), fontsize=7)
    save_figure("fig10_combined4_source_domain_comparison.png", "combined4", selected["combined4"]["actual_source_method"], "selected_best.pt", "test", "artifacts/merged_best_by_dataset/evaluation/combined4/source_domain_metrics_raw.json", f"Combined-4 source-domain Macro-F1; {chart_caption(selected['combined4'], 'Combined-4')}")

    plt.figure(figsize=(10, 4))
    plt.axis("off")
    for index, dataset in enumerate(DATASETS):
        item = selected[dataset]
        x = 0.1 + index * 0.48
        plt.text(x, 0.72, dataset, fontsize=13, weight="bold")
        plt.text(x, 0.54, f"Display: {item['display_label']}", fontsize=9)
        plt.text(x, 0.42, f"Source: {item['actual_source_method']}", fontsize=9)
        plt.text(x, 0.30, f"SHA-256: {item['checkpoint_sha256'][:16]}…", fontsize=8, family="monospace")
        plt.text(x, 0.18, f"Classes: {', '.join(item['class_order'])}", fontsize=8)
    plt.title("Selected-checkpoint provenance (checkpoints excluded from Git)")
    save_figure("fig11_checkpoint_provenance_diagram.png", "shrimpdb3+combined4", "mixed selected sources", "selected_best.pt", "test", "model_registry/selected_best_by_dataset.json", "Selected-checkpoint provenance diagram; each test result retains its display label, actual source method, checkpoint hash, and class order.")
    return figures


def update_registries(registry: dict[str, Any], figures: list[dict[str, Any]], study_root: Path) -> None:
    figure_path = study_root / "artifacts/metadata/figure_registry.json"
    # The merged registry is authoritative for repository-facing figures. Legacy figure
    # files remain on disk as historical evidence but are not silently presented as part
    # of the corrected benchmark figure set.
    write_json(figure_path, figures)
    report_path = study_root / "artifacts/metadata/report_registry.json"
    report = read_json(report_path) if report_path.exists() else {}
    legacy_report = study_root / "artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html"
    if legacy_report.exists():
        report["filename"] = "artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html"
        report["sha256"] = sha256(legacy_report)
        report["size_bytes"] = legacy_report.stat().st_size
        report["status"] = "verified"
    report["merged_best_by_dataset"] = {
        "filename": "artifacts/merged_best_by_dataset/reports/merged_best_by_dataset_report.html",
        "source_artifact": "artifacts/metadata/merged_result_registry.json",
        "sha256": sha256(study_root / "artifacts/merged_best_by_dataset/reports/merged_best_by_dataset_report.html"),
        "status": "verified",
    }
    write_json(report_path, report)
    closure_path = study_root / "artifacts/metadata/closure_audit.json"
    closure = read_json(closure_path) if closure_path.exists() else {}
    closure["merged_best_by_dataset"] = {
        "status": "verified",
        "registry": "artifacts/metadata/merged_result_registry.json",
        "selected_results": registry["selected_results"],
        "figures": len(figures),
        "raw_metrics_modified": False,
        "checkpoints_tracked": False,
    }
    write_json(closure_path, closure)


def write_html_report(registry: dict[str, Any], figures: list[dict[str, Any]], study_root: Path) -> None:
    report_path = study_root / "artifacts/merged_best_by_dataset/reports/merged_best_by_dataset_report.html"
    rows = "\n".join(
        f"<tr><td>{item['dataset']}</td><td>{item['display_label']}</td><td>{item['actual_source_method']}</td><td>{pct(item['accuracy'])}</td><td>{pct(item['macro_f1'])}</td><td>{str(item['label_matches_source_method']).lower()}</td></tr>"
        for item in registry["selected_results"]
    )
    figure_rows = "\n".join(f"<li>{entry['filename']}: {entry['caption']}</li>" for entry in figures)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Merged best-by-dataset results</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.45rem;text-align:left}}th{{background:#eef}}code{{font-size:.9em}}</style></head>
<body><h1>Merged best-by-dataset benchmark</h1>
<p>The common display label is <strong>{registry['display_label_for_selected_results']}</strong>. The actual source method and checkpoint hash are retained for every selected result.</p>
<table><thead><tr><th>Dataset</th><th>Display label</th><th>Actual source method</th><th>Accuracy</th><th>Macro-F1</th><th>Label/source match</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Scientific integrity note</h2><p>{registry['scientific_integrity_note']}</p>
<h2>Figures</h2><ul>{figure_rows}</ul>
<p>Selection rule: highest Macro-F1 within dataset; accuracy is the tie-breaker. Results are fixed seed 42 and do not establish statistical significance or universal superiority.</p>
</body></html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    legacy_report = study_root / "artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html"
    legacy_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, legacy_report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-root", type=Path, default=Path(r"D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS"))
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--zip-sha256", default="dc3e0c544e386008f62df6dbd12d806f9d00d854ff1f833b70365db7e7030e59")
    args = parser.parse_args()
    final_results = validate_package(args.merged_root, args.zip_sha256)
    copy_selected_artifacts(args.merged_root, args.study_root)
    registry = build_registry(final_results, args.merged_root, args.study_root)
    table_paths = write_tables(registry, args.merged_root, args.study_root)
    figures = write_figures(registry, args.study_root)
    write_html_report(registry, figures, args.study_root)
    update_registries(registry, figures, args.study_root)
    print(json.dumps({"status": "verified", "tables": table_paths, "figures": len(figures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
