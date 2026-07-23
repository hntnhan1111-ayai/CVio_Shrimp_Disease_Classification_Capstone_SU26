#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    # NOTE: This legacy script reads from clean_metrics.csv.
    # For the authoritative source, use clean_original_baseline_vs_recsra.json
    # via scripts/regenerate_all_tables.py
    clean = pd.read_csv("results/raw/clean_metrics.csv")
    base = clean[clean.model.eq("baseline")].iloc[0]
    rec = clean[clean.model.eq("recsra")].iloc[0]
    metrics = [
        "mAP50", "mAP50_95", "mAP75", "precision", "recall",
        "AP_BG_mAP50_95", "AP_WSSV_mAP50_95",
    ]
    rows = []
    for metric in metrics:
        b, r = float(base[metric]), float(rec[metric])
        rows.append({
            "metric": metric,
            "baseline_percent": b * 100,
            "recsra_percent": r * 100,
            "absolute_gain_pp": (r - b) * 100,
            "relative_change_percent": ((r - b) / b * 100) if b else float("nan"),
        })
    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "clean_baseline_vs_recsra_percent.csv", index=False)


if __name__ == "__main__":
    main()
