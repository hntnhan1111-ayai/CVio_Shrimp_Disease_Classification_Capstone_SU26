#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from recsra.metrics import detection_metrics
from recsra.ultralytics_patch import patch_ultralytics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="checkpoints/yolo11s_baseline_best.pt")
    ap.add_argument("--recsra", default="checkpoints/yolo11s_recsra_best.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--output", default="outputs/clean_test")
    args = ap.parse_args()

    patch_ultralytics()
    from ultralytics import YOLO

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, checkpoint in (("baseline", args.baseline), ("recsra", args.recsra)):
        model = YOLO(checkpoint)
        result = model.val(
            data=args.data,
            split="test",
            imgsz=1536,
            iou=0.55,
            conf=0.0005,
            max_det=600,
            batch=1,
            workers=4,
            device=args.device,
            project=str(out / "ultralytics"),
            name=label,
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        rows.append({"model": label, **detection_metrics(result)})

    with (out / "clean_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out / "clean_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
