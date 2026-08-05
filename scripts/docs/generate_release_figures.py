"""Regenerate README figures from retained, audited machine-readable evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["svg.hashsalt"] = "cvio-release-audit-v1"

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "assets" / "results"
DIAGRAMS = ROOT / "docs" / "assets" / "diagrams"

INK = "#243447"
BLUE = "#2F6B9A"
BLUE_LIGHT = "#A9C8DE"
GOLD = "#D49A2A"
GOLD_LIGHT = "#F2D69A"
GREY = "#E8EDF2"
WHITE = "#FFFFFF"


def read_rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finish_chart(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        facecolor=WHITE,
        metadata={"Software": "CVio release figure generator"},
    )
    plt.close(fig)


def grouped_horizontal(
    labels: list[str],
    left: list[float],
    right: list[float],
    left_name: str,
    right_name: str,
    title: str,
    subtitle: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.75 * len(labels) + 2.4)))
    y = list(range(len(labels)))
    height = 0.32
    ax.barh(
        [i - height / 2 for i in y],
        left,
        height,
        color=BLUE_LIGHT,
        edgecolor=INK,
        label=left_name,
    )
    ax.barh(
        [i + height / 2 for i in y],
        right,
        height,
        color=GOLD,
        edgecolor=INK,
        label=right_name,
    )
    for i, value in enumerate(left):
        ax.text(
            value + 0.008,
            i - height / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    for i, value in enumerate(right):
        ax.text(
            value + 0.008,
            i + height / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("Macro-F1 (0 to 1)")
    ax.set_title(title, loc="left", fontsize=15, weight="bold", color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color="#526575")
    ax.grid(axis="x", color=GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    finish_chart(fig, path)


def class_distribution() -> None:
    rows = read_rows(
        "artifacts/tables/dataset/dataset_split_distribution_table_seed42.csv"
    )
    labels = ["Healthy", "BG", "WSSV", "WSSV_BG"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = list(range(len(labels)))
    height = 0.22
    colors = [BLUE, GOLD, "#788B9B"]
    offsets = [-height, 0, height]
    for split, color, offset in zip(
        ("train", "val", "test"), colors, offsets, strict=True
    ):
        row = next(item for item in rows if item["split"] == split)
        values = [int(row[label]) for label in labels]
        ax.barh(
            [i + offset for i in y],
            values,
            height,
            color=color,
            edgecolor=INK,
            label=split,
        )
        for i, value in enumerate(values):
            ax.text(
                value + 4, i + offset, str(value), va="center", fontsize=8.5, color=INK
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 320)
    ax.set_xlabel("Images")
    ax.set_title(
        "SDI-4 class distribution",
        loc="left",
        fontsize=15,
        weight="bold",
        color=INK,
        pad=24,
    )
    ax.text(
        0,
        1.02,
        "Fixed seed-42 image-level split: train 804, validation 172, test 173",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#526575",
    )
    ax.grid(axis="x", color=GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    finish_chart(fig, RESULTS / "class_distribution.png")


def clean_results() -> None:
    row = read_rows(
        "artifacts/tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv"
    )[0]
    labels = ["Accuracy", "Macro-F1"]
    baseline = [float(row["baseline_accuracy"]), float(row["baseline_macro_f1"])]
    proposed = [float(row["best_accuracy"]), float(row["best_macro_f1"])]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = [0, 1]
    width = 0.34
    ax.bar(
        [i - width / 2 for i in x],
        baseline,
        width,
        color=BLUE_LIGHT,
        edgecolor=INK,
        label="CE",
    )
    ax.bar(
        [i + width / 2 for i in x],
        proposed,
        width,
        color=GOLD,
        edgecolor=INK,
        label="ASL-LDAM + SimAM-DCFR",
    )
    for i, value in enumerate(baseline):
        ax.text(
            i - width / 2,
            value + 0.004,
            f"{value:.6f}",
            ha="center",
            fontsize=9,
            color=INK,
        )
    for i, value in enumerate(proposed):
        ax.text(
            i + width / 2,
            value + 0.004,
            f"{value:.6f}",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.84, 0.94)
    ax.set_ylabel("Score (focused scale)")
    ax.set_title(
        "Reported clean SDI-4 results",
        loc="left",
        fontsize=15,
        weight="bold",
        color=INK,
        pad=24,
    )
    ax.text(
        0,
        1.02,
        "Fixed test n=173; official-final checkpoint binaries remain unresolved",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#526575",
    )
    ax.grid(axis="y", color=GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    finish_chart(fig, RESULTS / "clean_results.png")


def corruption_results() -> None:
    rows = read_rows(
        "artifacts/tables/noise/top5_noise_baseline_vs_best_mean_summary.csv"
    )
    labels = [row["corruption"].replace("_", " ") for row in rows]
    baseline = [float(row["baseline_mean_macro_f1_s1_s3"]) for row in rows]
    proposed = [float(row["mean_macro_f1_s1_s3"]) for row in rows]
    grouped_horizontal(
        labels,
        baseline,
        proposed,
        "CE comparator",
        "Historical method package",
        "Controlled-corruption mean performance",
        "Severity 1-3 mean; historical 0.905448 clean package, not the official-final SDI-4 checkpoint",
        RESULTS / "corruption_results.png",
    )


def regime_comparison() -> None:
    rows = read_rows("artifacts/release_audit/verified_regime_results.csv")
    grouped_horizontal(
        [row["dataset_regime"] for row in rows],
        [float(row["ce_macro_f1"]) for row in rows],
        [float(row["proposed_macro_f1"]) for row in rows],
        "CE",
        "ASL-LDAM + SimAM-DCFR",
        "Macro-F1 by dataset regime",
        "SDI-4 is reported-only; EXT-3 and combined rows come from audited result packages",
        RESULTS / "regime_comparison.png",
    )


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    fill: str = WHITE,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        facecolor=fill,
        edgecolor=INK,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
        wrap=True,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=INK
        )
    )


def diagram(
    path: Path,
    title: str,
    subtitle: str,
    boxes: list[tuple[float, float, float, float, str, str]],
    arrows: list[tuple[tuple[float, float], tuple[float, float]]],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(0.2, 4.65, title, fontsize=17, weight="bold", color=INK)
    ax.text(0.2, 4.28, subtitle, fontsize=10, color="#526575")
    for x, y, w, h, label, fill in boxes:
        box(ax, (x, y), w, h, label, fill)
    for start, end in arrows:
        arrow(ax, start, end)
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        facecolor=WHITE,
        metadata={"Date": None, "Creator": "CVio release figure generator"},
    )
    plt.close(fig)
    svg = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8"
    )


def architecture_diagrams() -> None:
    diagram(
        DIAGRAMS / "architecture_overview.svg",
        "YOLO26m classification with a late feature-recalibration path",
        "The proposed architecture inserts SimAM-DCFR before the original Classify head; CE omits the highlighted block.",
        [
            (0.3, 1.6, 1.6, 1.0, "RGB image\n224 x 224", WHITE),
            (2.5, 1.6, 2.1, 1.0, "YOLO26m\nfeature extractor", BLUE_LIGHT),
            (5.2, 1.6, 2.1, 1.0, "Feature tensor\nB x 512 x 7 x 7", WHITE),
            (
                7.9,
                1.35,
                1.8,
                1.5,
                "SimAM-DCFR\nlate recalibration\n(proposed only)",
                GOLD_LIGHT,
            ),
            (10.3, 1.6, 1.4, 1.0, "Classify\nlogits", WHITE),
        ],
        [
            ((1.9, 2.1), (2.5, 2.1)),
            ((4.6, 2.1), (5.2, 2.1)),
            ((7.3, 2.1), (7.9, 2.1)),
            ((9.7, 2.1), (10.3, 2.1)),
        ],
    )
    diagram(
        DIAGRAMS / "simam_dcfr_block.svg",
        "SimAM-DCFR block",
        "Texture and channel gates modulate the late feature tensor, followed by residual fusion.",
        [
            (0.3, 1.7, 1.5, 0.9, "Input X", WHITE),
            (2.4, 2.7, 2.1, 0.9, "Depthwise 3 x 3\nPointwise 1 x 1", BLUE_LIGHT),
            (5.1, 2.7, 1.7, 0.9, "Sigmoid\ntexture mask", GOLD_LIGHT),
            (2.4, 0.7, 2.1, 0.9, "Global average pool\nPointwise 1 x 1", BLUE_LIGHT),
            (5.1, 0.7, 1.7, 0.9, "Sigmoid\nchannel gate", GOLD_LIGHT),
            (7.5, 1.7, 2.0, 0.9, "X * texture * gate", WHITE),
            (10.1, 1.7, 1.6, 0.9, "Residual\nX + ...", GOLD_LIGHT),
        ],
        [
            ((1.8, 2.15), (2.4, 3.15)),
            ((1.8, 2.15), (2.4, 1.15)),
            ((4.5, 3.15), (5.1, 3.15)),
            ((4.5, 1.15), (5.1, 1.15)),
            ((6.8, 3.15), (7.5, 2.35)),
            ((6.8, 1.15), (7.5, 1.95)),
            ((9.5, 2.15), (10.1, 2.15)),
            ((1.8, 2.15), (10.1, 2.55)),
        ],
    )
    diagram(
        DIAGRAMS / "asl_ldam_flow.svg",
        "ASL-LDAM computation used in this repository",
        "Class-count margins modify the true-class logit before the single-label asymmetric loss.",
        [
            (0.3, 1.7, 1.4, 0.9, "Logits z", WHITE),
            (2.2, 2.6, 1.9, 0.9, "Class counts n_j\nmargin m_j", BLUE_LIGHT),
            (2.2, 0.8, 1.9, 0.9, "True class y", WHITE),
            (4.8, 1.7, 2.0, 0.9, "Subtract m_y\nfrom z_y", GOLD_LIGHT),
            (7.5, 1.7, 1.5, 0.9, "Scale logits", WHITE),
            (
                9.6,
                1.35,
                2.0,
                1.6,
                "Single-label ASL\nfocus easy/hard\nnegatives differently",
                GOLD_LIGHT,
            ),
        ],
        [
            ((1.7, 2.15), (4.8, 2.15)),
            ((4.1, 3.05), (4.8, 2.45)),
            ((4.1, 1.25), (4.8, 1.85)),
            ((6.8, 2.15), (7.5, 2.15)),
            ((9.0, 2.15), (9.6, 2.15)),
        ],
    )
    diagram(
        DIAGRAMS / "checkpoint_provenance.svg",
        "Checkpoint provenance and publication gate",
        "A filename is never sufficient: hash, regime, classes, architecture, and result evidence must all agree.",
        [
            (0.3, 1.7, 1.6, 0.9, "Candidate\nbinary", WHITE),
            (2.5, 1.7, 1.6, 0.9, "SHA-256\nidentity", BLUE_LIGHT),
            (4.7, 1.7, 1.6, 0.9, "Dataset +\nclass order", BLUE_LIGHT),
            (6.9, 1.7, 1.6, 0.9, "Architecture\naudit", BLUE_LIGHT),
            (9.1, 1.7, 1.2, 0.9, "Metric\nevidence", BLUE_LIGHT),
            (10.9, 2.6, 0.9, 0.9, "Publish", GOLD_LIGHT),
            (10.9, 0.8, 0.9, 0.9, "Reject /\nhold", GREY),
        ],
        [
            ((1.9, 2.15), (2.5, 2.15)),
            ((4.1, 2.15), (4.7, 2.15)),
            ((6.3, 2.15), (6.9, 2.15)),
            ((8.5, 2.15), (9.1, 2.15)),
            ((10.3, 2.3), (10.9, 2.85)),
            ((10.3, 2.0), (10.9, 1.25)),
        ],
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    class_distribution()
    clean_results()
    corruption_results()
    regime_comparison()
    architecture_diagrams()


if __name__ == "__main__":
    main()
