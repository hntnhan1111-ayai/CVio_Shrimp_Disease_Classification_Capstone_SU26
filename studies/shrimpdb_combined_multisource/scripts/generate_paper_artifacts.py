"""Generate publication-facing tables and figures from the imported raw results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
TABLES = ROOT / "artifacts" / "tables"
FIGURES = ROOT / "artifacts" / "figures"
DATASETS = ("shrimpdb3", "combined4")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def write_tables() -> None:
    final = load_json(RESULTS / "FINAL_BEST_RESULTS.json")
    selected = []
    per_class = []
    for dataset in DATASETS:
        result = final["datasets"][dataset]
        raw = load_json(RESULTS / dataset / "metrics_raw.json")
        selected.append({
            "dataset": "ShrimpDB-3" if dataset == "shrimpdb3" else "Combined-4",
            "display_label": result["display_label"],
            "actual_source_method": result["actual_source_method"],
            "label_matches_source_method": str(result["label_matches_source_method"]).lower(),
            "accuracy": percent(raw["accuracy"]),
            "balanced_accuracy": percent(raw["balanced_accuracy"]),
            "macro_f1": percent(raw["macro_f1"]),
            "weighted_f1": percent(raw["weighted_f1"]),
            "ece": percent(raw["ece_15_bins"]),
            "test_images": raw["n"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        })
        for class_name, metrics in raw["classification_report"].items():
            if class_name in {"accuracy", "macro avg", "weighted avg"}:
                continue
            per_class.append({
                "dataset": "ShrimpDB-3" if dataset == "shrimpdb3" else "Combined-4",
                "class": class_name,
                "precision": percent(metrics["precision"]),
                "recall": percent(metrics["recall"]),
                "f1": percent(metrics["f1-score"]),
                "support": int(metrics["support"]),
            })
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(TABLES / "main_results_percent.csv", index=False)
    pd.DataFrame(per_class).to_csv(TABLES / "per_class_results_percent.csv", index=False)

    split = pd.read_csv(ROOT / "artifacts" / "academic" / "tables" / "dataset_source_split_distribution.csv")
    split["dataset"] = split["experiment"].map({"shrimpdb3": "ShrimpDB-3", "combined4": "Combined-4"})
    split = split.groupby(["dataset", "split", "class_name"], as_index=False)["images"].sum()
    split = split.pivot_table(index=["dataset", "split"], columns="class_name", values="images", fill_value=0).reset_index()
    class_columns = [column for column in split.columns if column not in {"dataset", "split"}]
    split[class_columns] = split[class_columns].astype(int)
    split["total"] = split[class_columns].sum(axis=1).astype(int)
    split.to_csv(TABLES / "dataset_split_counts.csv", index=False)

    comparison = pd.read_csv(RESULTS / "ALL_METHODS_COMPARISON.csv")
    for column in ("accuracy_percent", "balanced_accuracy_percent", "macro_f1_percent", "weighted_f1_percent"):
        comparison[column] = comparison[column].map(lambda value: f"{float(value):.2f}%")
    comparison.to_csv(TABLES / "method_comparison_percent.csv", index=False)

    checkpoints = []
    for dataset in DATASETS:
        result = final["datasets"][dataset]
        checkpoints.append({
            "dataset": "ShrimpDB-3" if dataset == "shrimpdb3" else "Combined-4",
            "display_label": result["display_label"],
            "actual_source_method": result["actual_source_method"],
            "checkpoint_filename": "selected_best.pt",
            "checkpoint_sha256": result["checkpoint_sha256"],
            "class_order": "Healthy | BG | WSSV" if dataset == "shrimpdb3" else "Healthy | BG | WSSV | WSSV_BG",
            "tracked_in_git": "false",
        })
    pd.DataFrame(checkpoints).to_csv(TABLES / "checkpoint_summary.csv", index=False)


def save(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def write_figures() -> None:
    final = load_json(RESULTS / "FINAL_BEST_RESULTS.json")
    selected = {dataset: load_json(RESULTS / dataset / "metrics_raw.json") for dataset in DATASETS}

    split = pd.read_csv(TABLES / "dataset_split_counts.csv")
    split.set_index(["dataset", "split"]).drop(columns=["total"]).plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Images")
    plt.title("Dataset split distribution")
    plt.tight_layout()
    save("fig01_dataset_split_distribution.png")

    labels = ["ShrimpDB-3", "Combined-4"]
    values = [selected["shrimpdb3"]["accuracy"] * 100, selected["combined4"]["accuracy"] * 100]
    plt.figure(figsize=(7, 5)); plt.bar(labels, values, color=["#2070b4", "#d95f02"])
    plt.ylabel("Accuracy (%)"); plt.ylim(0, 100); plt.title("Selected accuracy")
    save("fig02_selected_accuracy_by_dataset.png")

    values = [selected["shrimpdb3"]["macro_f1"] * 100, selected["combined4"]["macro_f1"] * 100]
    plt.figure(figsize=(7, 5)); plt.bar(labels, values, color=["#2070b4", "#d95f02"])
    plt.ylabel("Macro-F1 (%)"); plt.ylim(0, 100); plt.title("Selected Macro-F1")
    save("fig03_selected_macro_f1_by_dataset.png")

    comparison = pd.read_csv(RESULTS / "ALL_METHODS_COMPARISON.csv")
    for dataset, name in (("shrimpdb3", "fig04_all_methods_shrimpdb3_comparison.png"), ("combined4", "fig05_all_methods_combined4_comparison.png")):
        subset = comparison[comparison["dataset"] == dataset]
        plt.figure(figsize=(8, 5)); plt.bar(subset["source_method"], subset["macro_f1_percent"])
        plt.ylabel("Macro-F1 (%)"); plt.ylim(0, 100); plt.title(f"All methods: {dataset}")
        plt.xticks(rotation=15); plt.tight_layout(); save(name)

    for dataset, number in (("shrimpdb3", 6), ("combined4", 8)):
        path = RESULTS / dataset / "confusion_matrix_counts.csv"
        matrix = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
        classes = ["Healthy", "BG", "WSSV"] if dataset == "shrimpdb3" else ["Healthy", "BG", "WSSV", "WSSV_BG"]
        for normalized, suffix in ((False, "counts"), (True, "normalized")):
            values = matrix / matrix.sum(axis=1, keepdims=True) if normalized else matrix
            plt.figure(figsize=(6, 5)); plt.imshow(values, cmap="Blues")
            plt.xticks(range(len(classes)), classes, rotation=30); plt.yticks(range(len(classes)), classes)
            plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{dataset} confusion matrix ({suffix})")
            for row in range(len(classes)):
                for col in range(len(classes)):
                    plt.text(col, row, f"{values[row, col]:.2f}" if normalized else f"{int(values[row, col])}", ha="center", va="center")
            plt.colorbar(); plt.tight_layout(); save(f"fig{number if not normalized else number + 1:02d}_{dataset}_confusion_matrix_{suffix}.png")

    domain = load_json(RESULTS / "combined4" / "source_domain_metrics_raw.json")
    plt.figure(figsize=(8, 5)); plt.bar(list(domain), [value["macro_f1"] * 100 for value in domain.values()])
    plt.ylabel("Macro-F1 (%)"); plt.ylim(0, 100); plt.title("Combined-4 source-domain Macro-F1")
    plt.xticks(rotation=15); plt.tight_layout(); save("fig10_combined4_source_domain_comparison.png")

    plt.figure(figsize=(9, 4)); plt.axis("off")
    for index, dataset in enumerate(DATASETS):
        result = final["datasets"][dataset]
        plt.text(index / 2, 0.75, dataset, fontsize=12, weight="bold")
        plt.text(index / 2, 0.52, f"Display: {result['display_label']}")
        plt.text(index / 2, 0.35, f"Source: {result['actual_source_method']}")
        plt.text(index / 2, 0.18, result["checkpoint_sha256"][:16] + "…", family="monospace")
    plt.title("Selected checkpoint provenance")
    save("fig11_checkpoint_provenance_diagram.png")


if __name__ == "__main__":
    write_tables()
    write_figures()
    print("Generated paper tables and figures")
