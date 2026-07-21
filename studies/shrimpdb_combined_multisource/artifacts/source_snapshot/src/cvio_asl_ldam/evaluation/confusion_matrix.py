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
    import seaborn as sns

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, png_path
