"""Classification metric calculation."""

from __future__ import annotations

from typing import Any


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=["Healthy", "BG", "WSSV", "WSSV_BG"],
            output_dict=True,
            zero_division=0,
        ),
    }
