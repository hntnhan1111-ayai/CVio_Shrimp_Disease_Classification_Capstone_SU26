#!/usr/bin/env python3
"""Generate publication figures from canonical metric sources."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"

# Canonical data
canonical = json.loads((RAW / "clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8"))
bline = canonical["baseline"]
recsra = canonical["recsra"]

METRICS = ["mAP50", "mAP50_95", "precision", "recall", "AP_BG_mAP50_95", "AP_WSSV_mAP50_95"]
METRIC_LABELS = ["mAP50", "mAP50-95", "Precision", "Recall", "BG AP50-95", "WSSV AP50-95"]

b_vals = [bline[m] * 100 for m in METRICS]
r_vals = [recsra[m] * 100 for m in METRICS]
gains = [r - b for r, b in zip(r_vals, b_vals)]
rel_gains = [g / b * 100 if b else 0 for g, b in zip(gains, b_vals)]


def save(fig, name):
    fig.savefig(name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_clean_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(METRIC_LABELS))
    w = 0.35
    bars1 = ax.bar(x - w/2, b_vals, w, label="YOLO11s baseline", color="#4C72B0")
    bars2 = ax.bar(x + w/2, r_vals, w, label="YOLO11s-RECSRA", color="#DD8452")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Clean test comparison — Original YOLO11s baseline vs YOLO11s-RECSRA")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, rotation=15, ha="right")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save(fig, FIG / "clean" / "fig_clean_original_yolo11s_vs_recsra.png")


def fig_absolute_relative_gains():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(METRIC_LABELS))
    ax1.bar(x, gains, color="#55A868", label="Absolute gain (pp)")
    ax1.set_ylabel("Absolute gain (percentage points)", color="#55A868")
    ax1.tick_params(axis="y", labelcolor="#55A868")
    ax2 = ax1.twinx()
    ax2.plot(x, rel_gains, "o-", color="#C44E52", label="Relative gain (%)")
    ax2.set_ylabel("Relative improvement (%)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax1.set_xticks(x)
    ax1.set_xticklabels(METRIC_LABELS, rotation=15, ha="right")
    ax1.set_title("Clean-test gains over original YOLO11s baseline")
    ax1.spines["top"].set_visible(False)
    fig.tight_layout()
    save(fig, FIG / "clean" / "fig_clean_absolute_and_relative_gains.png")


def fig_efficiency():
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["YOLO11s\nbaseline", "YOLO11s-RECSRA"]
    params = [9.43, 8.344]
    sizes = [18.39, None]
    fps = [169.5, None]
    x = np.arange(len(labels))
    ax.bar(x, params, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Parameter count comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for i, v in enumerate(params):
        ax.text(i, v + 0.1, f"{v:.2f}M", ha="center", va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, FIG / "clean" / "fig_efficiency_comparison.png")


def fig_top5_gain_by_severity():
    import csv
    top5 = []
    with (ROOT / "results/tables/top5_corruptions_percent.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            top5.append(row)
    noise_ids = [r["noise_id"] for r in top5]
    noise_names = [r["noise_name"].replace("_", " ").title() for r in top5]

    sev_data = {nid: [] for nid in noise_ids}
    with (ROOT / "results/raw/all_clean_and_50_condition_metrics.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["noise_id"] in sev_data and row["model"] == "recsra" and row["condition"] == "corruption":
                sev_data[row["noise_id"]].append(float(row["mAP50_95"]))
            if row["noise_id"] in sev_data and row["model"] == "baseline" and row["condition"] == "corruption":
                sev_data[row["noise_id"]].append(float(row["mAP50_95"]))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    severities = ["S1", "S2", "S3", "S4", "S5"]
    for idx, (nid, name) in enumerate(zip(noise_ids, noise_names)):
        ax = axes[idx]
        b_vals_sev = []
        r_vals_sev = []
        with (ROOT / "results/raw/all_clean_and_50_condition_metrics.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["noise_id"] == nid and row["condition"] == "corruption":
                    if row["model"] == "baseline":
                        b_vals_sev.append(float(row["mAP50_95"]))
                    else:
                        r_vals_sev.append(float(row["mAP50_95"]))
        x = np.arange(5)
        ax.plot(x, [v*100 for v in b_vals_sev], "o--", label="Baseline", color="#4C72B0")
        ax.plot(x, [v*100 for v in r_vals_sev], "s-", label="RECSRA", color="#DD8452")
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(severities)
        ax.set_ylabel("mAP50-95 (%)")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[5].axis("off")
    fig.suptitle("Top-five corruptions: RECSRA vs baseline by severity", y=1.02)
    fig.tight_layout()
    save(fig, FIG / "robustness" / "fig_top5_gain_by_severity.png")


def fig_training_curves():
    try:
        import pandas as pd
        df = pd.read_csv(RAW / "training" / "results.csv")
    except ImportError:
        print("[SKIP] pandas not available for training curves")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs = df["epoch"]
    axes[0].plot(epochs, df["train/box_loss"], label="train", color="#4C72B0")
    axes[0].plot(epochs, df["val/box_loss"], label="val", color="#DD8452")
    axes[0].set_title("Box loss")
    axes[0].legend()
    axes[1].plot(epochs, df["train/cls_loss"], label="train", color="#4C72B0")
    axes[1].plot(epochs, df["val/cls_loss"], label="val", color="#DD8452")
    axes[1].set_title("Classification loss")
    axes[1].legend()
    axes[2].plot(epochs, df["metrics/mAP50(B)"], label="val mAP50", color="#55A868")
    axes[2].plot(epochs, df["metrics/mAP50-95(B)"], label="val mAP50-95", color="#C44E52")
    axes[2].set_title("Validation mAP")
    axes[2].legend()
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("RECSRA training curves (seed 42, 200 epochs)")
    fig.tight_layout()
    save(fig, FIG / "training" / "fig_training_validation_metrics.png")


if __name__ == "__main__":
    (FIG / "clean").mkdir(parents=True, exist_ok=True)
    (FIG / "robustness").mkdir(parents=True, exist_ok=True)
    (FIG / "training").mkdir(parents=True, exist_ok=True)
    (FIG / "dataset_eda").mkdir(parents=True, exist_ok=True)

    fig_clean_comparison()
    fig_absolute_relative_gains()
    fig_efficiency()
    fig_top5_gain_by_severity()
    fig_training_curves()
    print("[OK] All figures generated.")
