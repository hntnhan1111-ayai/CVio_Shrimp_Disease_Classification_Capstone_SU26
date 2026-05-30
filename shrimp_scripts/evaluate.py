"""Metrics, prediction artifacts, and run-output collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .utils import read_json, save_csv, write_json


def confusion_matrix_np(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def per_class_stats(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) != 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    support = cm.sum(axis=1).astype(int)
    return precision, recall, f1, support


def cohen_kappa_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.trace(cm) / total
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total)
    if pe == 1:
        return 0.0
    return float((po - pe) / (1 - pe))


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    cm = confusion_matrix_np(y_true, y_pred)
    per_p, per_r, per_f1, per_support = per_class_stats(cm)
    precision = float(per_p.mean())
    recall = float(per_r.mean())
    macro_f1 = float(per_f1.mean())
    accuracy = float(np.trace(cm) / max(1, cm.sum()))
    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "cohen_kappa": cohen_kappa_from_cm(cm),
        "confusion_matrix": cm.tolist(),
        "BG->WSSV_BG": int(cm[1, 3]),
        "WSSV->WSSV_BG": int(cm[2, 3]),
        "WSSV_BG->BG": int(cm[3, 1]),
        "WSSV_BG->WSSV": int(cm[3, 2]),
        "per_class": {},
    }
    for index, name in enumerate(config.CLASS_NAMES):
        metrics["per_class"][name] = {
            "precision": float(per_p[index]),
            "recall": float(per_r[index]),
            "f1": float(per_f1[index]),
            "support": int(per_support[index]),
        }
    return metrics


def prediction_frame(
    frame: pd.DataFrame,
    run_id: str,
    model_name: str,
    backend: str,
    loss_name: str,
    condition_key: str,
    randaugment: bool,
    split_name: str,
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, (_idx, item) in enumerate(frame.reset_index(drop=True).iterrows()):
        probs = probabilities[i]
        rows.append({
            "run_id": run_id,
            "model": model_name,
            "backend": backend,
            "loss": loss_name,
            "condition": condition_key,
            "randaugment": bool(randaugment),
            "split": split_name,
            "image_path": item.get("yolo_path", item.get("processed_path", item.get("source_path", ""))),
            "rel_path": item["rel_path"],
            "true_label": int(y_true[i]),
            "true_class": config.CLASS_NAMES[int(y_true[i])],
            "pred_label": int(y_pred[i]),
            "pred_class": config.CLASS_NAMES[int(y_pred[i])],
            "confidence": float(max(probs)),
            "correct": int(int(y_true[i]) == int(y_pred[i])),
            "class_prob_Healthy": float(probs[0]),
            "class_prob_BG": float(probs[1]),
            "class_prob_WSSV": float(probs[2]),
            "class_prob_WSSV_BG": float(probs[3]),
        })
    return pd.DataFrame(rows)


def confusion_count_frame(metrics: dict[str, Any], run_id: str, model_name: str, backend: str, condition_key: str, loss_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cm = np.array(metrics["confusion_matrix"], dtype=int)
    for true_index in range(config.NUM_CLASSES):
        for pred_index in range(config.NUM_CLASSES):
            rows.append({
                "run_id": run_id,
                "model": model_name,
                "backend": backend,
                "condition": condition_key,
                "loss_key": loss_key,
                "true_label": true_index,
                "true_class": config.CLASS_NAMES[true_index],
                "pred_label": pred_index,
                "pred_class": config.CLASS_NAMES[pred_index],
                "count": int(cm[true_index, pred_index]),
            })
    return pd.DataFrame(rows)


def classification_report_frame(y_true: list[int], y_pred: list[int]) -> pd.DataFrame:
    cm = confusion_matrix_np(y_true, y_pred)
    per_p, per_r, per_f1, per_support = per_class_stats(cm)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(config.CLASS_NAMES):
        rows.append({
            "class_or_average": name,
            "precision": float(per_p[index]),
            "recall": float(per_r[index]),
            "f1-score": float(per_f1[index]),
            "support": int(per_support[index]),
        })
    rows.append({
        "class_or_average": "macro avg",
        "precision": float(per_p.mean()),
        "recall": float(per_r.mean()),
        "f1-score": float(per_f1.mean()),
        "support": int(per_support.sum()),
    })
    rows.append({
        "class_or_average": "accuracy",
        "precision": "",
        "recall": "",
        "f1-score": float(np.trace(cm) / max(1, cm.sum())),
        "support": int(per_support.sum()),
    })
    return pd.DataFrame(rows)


def confusion_matrix_frame(y_true: list[int], y_pred: list[int]) -> pd.DataFrame:
    cm = confusion_matrix_np(y_true, y_pred)
    return pd.DataFrame(cm, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).reset_index().rename(columns={"index": "true_class"})


def save_prediction_artifacts(
    run_dir: str | Path,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    y_true: list[int],
    y_pred: list[int],
    split_name: str = "test",
) -> None:
    run_dir = Path(run_dir)
    save_csv(predictions, run_dir / f"{split_name}_predictions.csv")
    if split_name == "test":
        save_csv(predictions, run_dir / "test_predictions.csv")
        save_csv(classification_report_frame(y_true, y_pred), run_dir / "classification_report.csv")
        cm_frame = confusion_matrix_frame(y_true, y_pred)
        save_csv(cm_frame, run_dir / "confusion_matrix.csv")
        write_json(run_dir / "confusion_matrix.json", metrics["confusion_matrix"])


def collect_run_outputs(output_dir: str | Path, expected_run_ids: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    failed_rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        empty = pd.DataFrame()
        save_csv(empty, output_dir / "final_summary.csv")
        save_csv(empty, output_dir / "missing_or_failed_runs.csv")
        return empty, empty, empty, empty
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if expected_run_ids is not None and run_dir.name not in expected_run_ids:
            continue
        status = read_json(run_dir / "status.json", default={})
        metrics_path = run_dir / "metrics.json"
        if status.get("status") == "completed" and metrics_path.exists():
            metrics = read_json(metrics_path)
            test = metrics.get("test", {})
            val = metrics.get("val", {})
            metrics_rows.append({
                "run_id": metrics.get("run_id", run_dir.name),
                "status": "completed",
                "model": metrics.get("model"),
                "model_key": metrics.get("model_key"),
                "backend": metrics.get("backend"),
                "loss_key": metrics.get("loss_key"),
                "loss": metrics.get("loss"),
                "condition": metrics.get("condition"),
                "randaugment": metrics.get("randaugment"),
                "seed": metrics.get("seed"),
                "repeat": metrics.get("repeat"),
                "split_seed": metrics.get("split_seed"),
                "val_macro_f1": val.get("macro_f1"),
                "test_accuracy": test.get("accuracy"),
                "test_macro_precision": test.get("macro_precision"),
                "test_macro_recall": test.get("macro_recall"),
                "test_macro_f1": test.get("macro_f1"),
                "cohen_kappa": test.get("cohen_kappa"),
                "BG->WSSV_BG": test.get("BG->WSSV_BG"),
                "WSSV->WSSV_BG": test.get("WSSV->WSSV_BG"),
                "WSSV_BG->BG": test.get("WSSV_BG->BG"),
                "WSSV_BG->WSSV": test.get("WSSV_BG->WSSV"),
                "training_time_s": metrics.get("training_time_s"),
                "latency_ms_image": test.get("latency_ms_image"),
                "fps": test.get("fps"),
                "params_m": metrics.get("params_m"),
                "model_size_mb": metrics.get("model_size_mb"),
                "checkpoint_path": metrics.get("checkpoint_path"),
                "checkpoint_sha256": metrics.get("checkpoint_sha256"),
            })
            if (run_dir / "test_predictions.csv").exists():
                prediction_frames.append(pd.read_csv(run_dir / "test_predictions.csv"))
            if (run_dir / "confusion_counts.csv").exists():
                confusion_frames.append(pd.read_csv(run_dir / "confusion_counts.csv"))
        else:
            failed_rows.append({
                "run_id": run_dir.name,
                "status": status.get("status", "missing_status"),
                "error": status.get("error", status.get("errors", "")),
            })
    metrics_frame = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    confusions = pd.concat(confusion_frames, ignore_index=True) if confusion_frames else pd.DataFrame()
    failed = pd.DataFrame(failed_rows)
    save_csv(metrics_frame, output_dir / "per_run_metrics.csv")
    save_csv(predictions, output_dir / "per_seed_predictions.csv")
    save_csv(confusions, output_dir / "per_seed_confusion_counts.csv")
    save_csv(failed, output_dir / "missing_or_failed_runs.csv")
    save_csv(metrics_frame, output_dir / "final_summary.csv")
    return metrics_frame, predictions, confusions, failed
