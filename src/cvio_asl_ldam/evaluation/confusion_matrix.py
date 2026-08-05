"""Save confusion matrices as CSV and PNG."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def save_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    output_prefix: str | Path,
    class_names: tuple[str, ...] = ("Healthy", "BG", "WSSV", "WSSV_BG"),
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    values = np.asarray(matrix, dtype=int)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for name, row in zip(class_names, values, strict=True):
            writer.writerow([name, *row.tolist()])

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=class_names)
    axis.set_yticks(range(len(class_names)), labels=class_names)
    threshold = float(values.max()) / 2.0 if values.size else 0.0
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                f"{int(value)}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, png_path


def save_normalized_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    output_prefix: str | Path,
    class_names: tuple[str, ...] = ("Healthy", "BG", "WSSV", "WSSV_BG"),
) -> tuple[Path, Path]:
    """Save a row-normalized confusion matrix as CSV and PNG."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    counts = np.asarray(matrix, dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)
    values = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for name, row in zip(class_names, values, strict=True):
            writer.writerow([name, *(f"{value:.9f}" for value in row)])

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=class_names)
    axis.set_yticks(range(len(class_names)), labels=class_names)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, png_path
