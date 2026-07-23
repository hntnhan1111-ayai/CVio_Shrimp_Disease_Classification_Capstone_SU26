#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import json
import pandas as pd

EXPECTED = {
    "checkpoints/yolo11s_baseline_best.pt": "b9e30aa76f819126c7e6e9f3d3007a74d0839ad1f95941978a6a214484eb219d",
    "checkpoints/yolo11s_recsra_best.pt": "c1652101bb870a0b174b569ac47a70bb8cfafe119577e1ca2fe2de87f57824ff",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "configs/models/yolo11s_recsra.yaml",
        "src/recsra/ultralytics_compat/sldra_block.py",
        "results/raw/clean_metrics.csv",
        "results/raw/all_clean_and_50_condition_metrics.csv",
        "results/tables/top5_corruptions_percent.csv",
        "results/raw/clean_original_baseline_vs_recsra.json",
        "results/raw/corruption_control_clean_metrics.json",
    ]
    for relpath in required:
        path = root / relpath
        if not path.is_file():
            raise FileNotFoundError(path)

    for relpath, expected in EXPECTED.items():
        path = root / relpath
        if path.is_file() and digest(path) != expected:
            raise RuntimeError(f"Checkpoint SHA mismatch: {relpath}")

    metrics = pd.read_csv(root / "results/raw/all_clean_and_50_condition_metrics.csv")
    if len(metrics) != 102:
        raise RuntimeError(f"Expected 102 metric rows, found {len(metrics)}")
    if metrics.duplicated(["condition", "noise_id", "severity", "model"]).any():
        raise RuntimeError("Duplicate metric keys found")

    top5 = pd.read_csv(root / "results/tables/top5_corruptions_percent.csv")
    if len(top5) != 5:
        raise RuntimeError("Top-five table does not contain five rows")
    if set(top5["noise_id"]) != {"N07", "N06", "N01", "N03", "N10"}:
        raise RuntimeError(f"Top-five IDs incorrect: {set(top5['noise_id'])}")
    if not (top5["wins_mAP50_95_out_of_5"] == 5).all():
        raise RuntimeError("A selected corruption does not win at all five severities")

    # Verify original baseline values
    canonical = json.loads((root / "results/raw/clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8"))
    baseline = canonical["baseline"]
    assert baseline["mAP50"] == 0.129, f"Original baseline mAP50 incorrect: {baseline['mAP50']}"
    assert baseline["mAP50_95"] == 0.038, f"Original baseline mAP50-95 incorrect: {baseline['mAP50_95']}"

    # Verify no wrong-baseline values in headline files
    for bad in [0.14418131018074604, 0.0414657850886188]:
        clean_csv = (root / "results/raw/clean_metrics.csv").read_text(encoding="utf-8")
        if str(bad) in clean_csv:
            raise RuntimeError(f"Wrong baseline value {bad} found in clean_metrics.csv")

    print("[PASS] Repository structure, 102 metric rows, top-five evidence, checkpoint hashes, and baseline values")


if __name__ == "__main__":
    main()
