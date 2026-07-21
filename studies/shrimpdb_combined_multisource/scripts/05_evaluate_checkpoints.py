#!/usr/bin/env python3
"""Evaluate best.pt and last.pt for both experiments."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

from cvio_asl_ldam.attention.patch_yolo import register_checkpoint_safe_globals
from cvio_asl_ldam.data.audit_dataset import resolve_dataset_root
from cvio_asl_ldam.utils.io import load_yaml, write_json
from cvio_asl_ldam.utils.paths import resolve_device
from cvio_asl_ldam.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")


def _ece(y_true: list[int], y_prob: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = np.equal(predictions, y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if not np.any(mask):
            continue
        ece += np.abs(confidences[mask] - accuracies[mask]).mean() * np.mean(mask)
    return float(ece)


def evaluate_checkpoint(
    weights: Path,
    dataset_root: Path,
    manifest_path: Path,
    output_dir: Path,
    device: str,
    num_classes: int = 4,
    class_names: tuple[str, ...] = ("Healthy", "BG", "WSSV", "WSSV_BG"),
) -> dict[str, Any]:
    from ultralytics import YOLO

    register_checkpoint_safe_globals()
    model = YOLO(str(weights))
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]

    paths = [str(dataset_root / row["rel_path"]) for row in rows]
    predictions = model.predict(
        source=paths,
        imgsz=224,
        device=device.split(",")[0],
        verbose=False,
    )
    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(result.probs.top1) for row, result in zip(rows, predictions)]
    y_prob = np.array([result.probs.data.cpu().numpy() for result in predictions])

    labels = list(range(num_classes))
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_class = {
        str(class_names[i]): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)[i]),
        }
        for i in range(num_classes)
    }

    metrics: dict[str, Any] = {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean([per_class[c]["precision"] for c in class_names])),
        "macro_recall": float(np.mean([per_class[c]["recall"] for c in class_names])),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "top2": float(top_k_accuracy_score(y_true, y_prob, k=2, labels=labels)),
        "ece_15": _ece(y_true, y_prob, n_bins=15),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=list(class_names), output_dict=True, zero_division=0
        ),
        "per_class": per_class,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics_raw.json", metrics)
    with (output_dir / "overall_metrics_percent.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "cohen_kappa", "mcc", "top2", "ece_15"]:
            writer.writerow([key, metrics[key]])

    LOGGER.info("Evaluated %s: accuracy=%.4f macro_f1=%.4f", weights.name, metrics["accuracy"], metrics["macro_f1"])
    return metrics


def run(config_path: str | Path = STUDY_CONFIG) -> int:
    study = load_yaml(config_path)
    experiments = study.get("experiments", {})
    device = resolve_device("auto")

    for exp_name in ("shrimpdb3", "combined4"):
        exp_cfg = experiments.get(exp_name, {})
        run_name = exp_cfg.get("run_name", f"yolo26m_cls__asl_ldam_simam_dcfr__{exp_name}__seed42")
        run_dir = Path("runs") / run_name
        weights_dir = run_dir / "weights"
        dataset_key = exp_cfg.get("dataset", exp_name)
        if dataset_key == "shrimpdb":
            ds_yaml = Path("configs/datasets/shrimpdb.yaml")
            class_names = tuple(exp_cfg.get("class_names", ["Healthy", "BG", "WSSV"]))
        else:
            ds_yaml = Path("configs/datasets/combined4.yaml")
            class_names = tuple(exp_cfg.get("class_names", ["Healthy", "BG", "WSSV", "WSSV_BG"]))

        ds_cfg = load_yaml(ds_yaml)
        manifest = Path(ds_cfg.get("manifest", f"experiments/{exp_name}/split_manifest_seed42_generated.csv"))
        data_root = Path(study.get("datasets", {}).get(dataset_key, {}).get("root", "datasets/processed-images"))
        dataset_root, _ = resolve_dataset_root(data_root, ds_yaml)

        for ckpt in ("best.pt", "last.pt"):
            weights = weights_dir / ckpt
            if not weights.is_file():
                LOGGER.warning("Missing checkpoint: %s", weights)
                continue
            out_dir = Path("evaluation") / exp_name / ckpt.replace(".pt", "")
            evaluate_checkpoint(weights, dataset_root, manifest, out_dir, device, len(class_names), class_names)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate experiment checkpoints")
    parser.add_argument("--config", default=str(STUDY_CONFIG))
    args = parser.parse_args()
    try:
        return run(args.config)
    except Exception as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
