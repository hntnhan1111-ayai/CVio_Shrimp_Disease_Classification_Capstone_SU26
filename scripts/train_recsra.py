#!/usr/bin/env python3
"""Exact single-seed training and clean evaluation flow for the frozen winner."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from recsra.pretrained import clean_cuda, load_fair_pretrained
from recsra.ultralytics_patch import patch_ultralytics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/models/yolo11s_recsra.yaml")
    ap.add_argument("--pretrain", default="checkpoints/yolo11s_baseline_best.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="outputs/recsra_seed42")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    patch_ultralytics()
    from ultralytics import YOLO

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output = Path(args.output).resolve()
    run_name = f"YOLO11s_RECSRA_200E_SEED{args.seed}"
    record = {"status": "fail", "run_name": run_name, "error": ""}
    started = time.time()

    try:
        model = YOLO(args.model)
        model.load(args.pretrain)
        if not getattr(model, "ckpt", None):
            raise RuntimeError("Checkpoint bookkeeping was not initialized")
        load_fair_pretrained(model, args.pretrain)

        model.train(
            data=args.data,
            pretrained=True,
            epochs=args.epochs,
            patience=0,
            imgsz=1280,
            batch=2,
            workers=8,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            project=str(output / "train"),
            name=run_name,
            exist_ok=True,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            nbs=64,
            cos_lr=True,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.0,
            close_mosaic=0,
            hsv_h=0.01,
            hsv_s=0.35,
            hsv_v=0.25,
            degrees=3.0,
            translate=0.08,
            scale=0.2,
            shear=1.0,
            perspective=0.0002,
            flipud=0.05,
            fliplr=0.5,
            box=10.0,
            cls=0.35,
            dfl=2.0,
            plots=True,
            verbose=True,
        )

        best = output / "train" / run_name / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(best)
        record["best_pt"] = str(best)
        del model
        clean_cuda()

        for split in ("val", "test"):
            model = YOLO(str(best))
            result = model.val(
                data=args.data,
                split=split,
                imgsz=1536,
                iou=0.55,
                conf=0.0005,
                max_det=600,
                batch=1,
                workers=4,
                device=args.device,
                project=str(output / "evaluation"),
                name=f"{run_name}_{split}",
                exist_ok=True,
                plots=True,
                verbose=True,
            )
            record[f"{split}_mAP50"] = float(result.box.map50)
            record[f"{split}_mAP50_95"] = float(result.box.map)
            del model, result
            clean_cuda()

        record["status"] = "ok"
    except Exception as exc:
        record["error"] = repr(exc)
        clean_cuda()
        raise
    finally:
        record["elapsed_sec"] = time.time() - started
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
