"""Metric helpers for Ultralytics detection results."""

from __future__ import annotations


def detection_metrics(result, prefix: str = "") -> dict[str, float]:
    maps = list(result.box.maps)
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}mAP50": float(result.box.map50),
        f"{key}mAP50_95": float(result.box.map),
        f"{key}mAP75": float(result.box.map75),
        f"{key}precision": float(result.box.mp),
        f"{key}recall": float(result.box.mr),
        f"{key}AP_BG_mAP50_95": float(maps[0]),
        f"{key}AP_WSSV_mAP50_95": float(maps[1]),
    }
