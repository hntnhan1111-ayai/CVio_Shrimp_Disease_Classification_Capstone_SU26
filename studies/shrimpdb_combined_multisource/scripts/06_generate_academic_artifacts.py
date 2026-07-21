#!/usr/bin/env python3
"""Generate paper-ready figures and tables from evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_confusion_png(matrix: list[list[int]], path: Path, class_names: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(values, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_confusion_normalized(matrix: list[list[int]], path: Path, class_names: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    values = np.asarray(matrix, dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normed = values / row_sums
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(normed, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis, vmin=0.0, vmax=1.0)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_overview_table(eval_root: Path, out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for exp in ("shrimpdb3", "combined4"):
        exp_dir = eval_root / exp
        if not exp_dir.is_dir():
            continue
        for ckpt_dir in sorted(exp_dir.iterdir()):
            metrics_path = ckpt_dir / "metrics_raw.json"
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {
                "experiment": exp,
                "checkpoint": ckpt_dir.name,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "cohen_kappa": metrics.get("cohen_kappa"),
                "mcc": metrics.get("mcc"),
                "top2": metrics.get("top2"),
                "ece_15": metrics.get("ece_15"),
            }
            rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_csv(out_path, rows)


def generate_confusion_matrices(eval_root: Path, out_dir: Path) -> None:
    class_names_3 = ("Healthy", "BG", "WSSV")
    class_names_4 = ("Healthy", "BG", "WSSV", "WSSV_BG")
    for exp, names in (("shrimpdb3", class_names_3), ("combined4", class_names_4)):
        ckpt_dir = eval_root / exp / "best"
        if not ckpt_dir.is_dir():
            continue
        metrics_path = ckpt_dir / "metrics_raw.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cm = metrics.get("confusion_matrix", [])
        if cm:
            _save_confusion_png(cm, out_dir / f"{exp}_best_confusion_count.png", names)
            _save_confusion_normalized(cm, out_dir / f"{exp}_best_confusion_normalized.png", names)


def generate_source_split_table(study_config: Path, out_path: Path) -> None:
    import yaml

    study = yaml.safe_load(study_config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for ds_key, ds_cfg in study.get("datasets", {}).items():
        for cls, count in ds_cfg.get("expected_counts", {}).items():
            rows.append({"dataset": ds_key, "class": cls, "total": count})
    _save_csv(out_path, rows)


def run(eval_root: Path = Path("evaluation"), figures_dir: Path = Path("artifacts/figures"), tables_dir: Path = Path("artifacts/tables"), study_config: Path = STUDY_CONFIG) -> int:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    generate_overview_table(eval_root, tables_dir / "experiment_overview_percent.csv")
    generate_confusion_matrices(eval_root, figures_dir)
    generate_source_split_table(study_config, tables_dir / "dataset_source_split_distribution.csv")
    LOGGER.info("Generated artifacts in %s and %s", figures_dir, tables_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate academic artifacts")
    parser.add_argument("--eval-root", default="evaluation")
    parser.add_argument("--figures-dir", default="artifacts/figures")
    parser.add_argument("--tables-dir", default="artifacts/tables")
    parser.add_argument("--config", default=str(STUDY_CONFIG))
    args = parser.parse_args()
    try:
        return run(Path(args.eval_root), Path(args.figures_dir), Path(args.tables_dir), Path(args.config))
    except Exception as exc:
        LOGGER.error("Artifact generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
