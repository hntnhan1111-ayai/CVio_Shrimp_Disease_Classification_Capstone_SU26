#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def metric_payload(metrics) -> dict:
    box = metrics.box
    maps = list(map(float, getattr(box, "maps", [])))
    all_ap = getattr(box, "all_ap", None)
    map75 = float("nan")
    try:
        if all_ap is not None and getattr(all_ap, "shape", (0, 0))[1] > 5:
            map75 = float(all_ap[:, 5].mean())
    except Exception:
        pass
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "mAP75": map75,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "AP_BG_mAP50_95": maps[0] if len(maps) > 0 else float("nan"),
        "AP_WSSV_mAP50_95": maps[1] if len(maps) > 1 else float("nan"),
        "speed": getattr(metrics, "speed", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--conf", type=float, default=0.0005)
    ap.add_argument("--max-det", type=int, default=600)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    # CUDA_VISIBLE_DEVICES must be set by the parent before this module imports torch/Ultralytics.
    from ultralytics import YOLO

    model = YOLO(args.model)
    result = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        iou=args.iou,
        conf=args.conf,
        max_det=args.max_det,
        batch=1,
        workers=args.workers,
        device=0,
        half=False,
        plots=False,
        verbose=False,
        project=str(Path(args.out).parent / "eval_runs"),
        name=args.run_name,
        exist_ok=True,
    )
    payload = metric_payload(result)
    payload.update({
        "model_path": str(Path(args.model).resolve()),
        "data_yaml": str(Path(args.data).resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
