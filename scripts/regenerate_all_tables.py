#!/usr/bin/env python3
"""Generate all percentage tables from canonical raw JSON sources."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"

METRIC_FIELDS = [
    "mAP50", "mAP50_95", "mAP75", "precision", "recall",
    "AP_BG_mAP50_95", "AP_WSSV_mAP50_95",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pp(value: float) -> str:
    return f"{value:+.3f} pp"


def fmt_rel(value: float) -> str:
    return f"{value:+.2f}%"


def pct(value: float) -> float:
    return value * 100.0


def generate_clean_tables() -> None:
    canonical = load_json(RAW / "clean_original_baseline_vs_recsra.json")
    baseline = canonical["baseline"]
    recsra = canonical["recsra"]

    rows = []
    for m in METRIC_FIELDS:
        if m not in baseline or m not in recsra:
            continue
        b = pct(baseline[m])
        r = pct(recsra[m])
        abs_gain = r - b
        rel_gain = ((r - b) / b * 100.0) if b else float("nan")
        rows.append({
            "metric": m,
            "baseline_percent": round(b, 4),
            "recsra_percent": round(r, 4),
            "absolute_gain_pp": round(abs_gain, 4),
            "relative_change_percent": round(rel_gain, 4),
        })

    csv_path = TABLES / "clean_original_yolo11s_vs_recsra_percent.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Clean test comparison — Original YOLO11s baseline vs YOLO11s-RECSRA",
        "",
        "| Metric | YOLO11s baseline | YOLO11s-RECSRA | Absolute change | Relative change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['metric']} "
            f"| {row['baseline_percent']:.3f}% "
            f"| {row['recsra_percent']:.3f}% "
            f"| {fmt_pp(row['absolute_gain_pp'])} "
            f"| {fmt_rel(row['relative_change_percent'])} |"
        )
    md_lines += [
        "",
        f"> **Baseline context:** {canonical['baseline_context']}.",
        f"> **RECSRA experiment ID:** {canonical['recsra_experiment_id']}.",
        "> Raw decimal metrics are preserved in `results/raw/clean_original_baseline_vs_recsra.json`.",
    ]
    (TABLES / "clean_original_yolo11s_vs_recsra.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    # Update legacy clean_baseline_vs_recsra files to point to the same canonical data
    # but keep backward compatibility by writing the same content
    with (TABLES / "clean_baseline_vs_recsra_percent.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    md_lines2 = [
        "# Clean test comparison",
        "",
        "| Metric | YOLO11s baseline | YOLO11s-RECSRA | Absolute change | Relative change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines2.append(
            f"| {row['metric']} "
            f"| {row['baseline_percent']:.3f}% "
            f"| {row['recsra_percent']:.3f}% "
            f"| {fmt_pp(row['absolute_gain_pp'])} "
            f"| {fmt_rel(row['relative_change_percent'])} |"
        )
    md_lines2 += [
        "",
        "> **Note:** This file is generated from `results/raw/clean_original_baseline_vs_recsra.json`.",
        "> Baseline context: `original_yolo11s_14_model_benchmark`.",
    ]
    (TABLES / "clean_baseline_vs_recsra.md").write_text(
        "\n".join(md_lines2) + "\n", encoding="utf-8"
    )


def generate_corruption_control_table() -> None:
    control = load_json(RAW / "corruption_control_clean_metrics.json")
    b = control["baseline"]
    r = control["recsra"]

    rows = []
    for m in METRIC_FIELDS:
        if m not in b or m not in r:
            continue
        bv, rv = pct(b[m]), pct(r[m])
        rows.append({
            "metric": m,
            "baseline_percent": round(bv, 4),
            "recsra_percent": round(rv, 4),
            "absolute_gain_pp": round(rv - bv, 4),
            "relative_change_percent": round(((rv - bv) / bv * 100.0) if bv else float("nan"), 4),
        })

    csv_path = TABLES / "corruption_control_clean_percent.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Corruption-control clean test comparison",
        "",
        "| Metric | Paired control baseline | RECSRA | Absolute change | Relative change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['metric']} "
            f"| {row['baseline_percent']:.3f}% "
            f"| {row['recsra_percent']:.3f}% "
            f"| {fmt_pp(row['absolute_gain_pp'])} "
            f"| {fmt_rel(row['relative_change_percent'])} |"
        )
    md_lines += [
        "",
        f"> **Baseline context:** {control['baseline_context']}.",
        f"> **Usage:** {control['usage']}.",
        "> These values are for the corruption benchmark paired control ONLY.",
        "> Do NOT use these as the main clean-test headline baseline.",
        "> The authoritative main baseline is in `clean_original_baseline_vs_recsra.json`.",
    ]
    (TABLES / "corruption_control_clean.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )


def verify_102_rows() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[SKIP] pandas not available, skipping 102-row verification")
        return
    metrics = pd.read_csv(RAW / "all_clean_and_50_condition_metrics.csv")
    assert len(metrics) == 102, f"Expected 102 rows, got {len(metrics)}"
    assert not metrics.duplicated(["condition", "noise_id", "severity", "model"]).any(), "Duplicate keys"


def verify_top5() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[SKIP] pandas not available, skipping top5 verification")
        return
    top5 = pd.read_csv(TABLES / "top5_corruptions_percent.csv")
    assert len(top5) == 5, f"Expected 5 rows, got {len(top5)}"
    assert set(top5["noise_id"]) == {"N07", "N06", "N01", "N03", "N10"}, f"Wrong top5 IDs: {set(top5['noise_id'])}"


if __name__ == "__main__":
    generate_clean_tables()
    generate_corruption_control_table()
    verify_102_rows()
    verify_top5()
    print("[OK] All tables generated and verified.")
