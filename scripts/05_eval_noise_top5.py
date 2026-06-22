#!/usr/bin/env python3
"""Evaluate the top-5 official corruptions at severities 1-3.

Corruptions:
    impulse_noise, gaussian_noise, contrast_reduction, defocus_blur, low_light

Does not train. Does not save corrupted image folders (generates in memory).
Outputs a per-severity CSV and a mean summary CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _test_rows(manifest: Path) -> list[dict[str, str]]:
    import csv
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", default="artifacts/manifests/split_manifest_seed42.csv")
    parser.add_argument("--dataset-config", default="configs/dataset/shrimpdiseasebd_seed42.yaml")
    parser.add_argument("--noise-config", default="configs/noise/top5_noise.yaml")
    parser.add_argument("--data-dir", help="Raw dataset root.")
    parser.add_argument("--output-dir", default="artifacts/evaluation/noise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()

    import csv

    import numpy as np
    from PIL import Image

    from cvio_asl_ldam.attention.patch_yolo import register_checkpoint_safe_globals
    from cvio_asl_ldam.data.audit_dataset import resolve_dataset_root
    from cvio_asl_ldam.data.corruptions import TOP5_CORRUPTIONS, apply_corruption
    from cvio_asl_ldam.evaluation.metrics import classification_metrics
    from cvio_asl_ldam.utils.io import load_yaml, write_json
    from cvio_asl_ldam.utils.paths import resolve_device
    from cvio_asl_ldam.utils.seed import seed_everything

    register_checkpoint_safe_globals()
    from ultralytics import YOLO

    dataset_config = load_yaml(args.dataset_config)
    noise_config = load_yaml(args.noise_config)
    if args.data_dir:
        dataset_root, _ = resolve_dataset_root(args.data_dir, dataset_config)
    else:
        dataset_root = Path("runs/prepared_seed42")
        if not dataset_root.is_dir():
            raise SystemExit("Provide --data-dir or run 01_prepare_dataset_and_split.py first.")

    rows = _test_rows(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device(args.device).split(",")[0]
    model = YOLO(args.weights)

    result_rows: list[dict[str, object]] = []
    for corruption in TOP5_CORRUPTIONS:
        if corruption not in noise_config["corruptions"]:
            raise AssertionError(f"Missing official corruption config: {corruption}")
        for severity in noise_config["severities"]:
            images = []
            for index, row in enumerate(rows):
                src = dataset_root / row["rel_path"]
                if not src.is_file():
                    class_dir = f"{int(row['label']):02d}_{row['class_name']}"
                    src = dataset_root / "test" / class_dir / row["filename"]
                image = Image.open(src).convert("RGB")
                images.append(apply_corruption(image, corruption, int(severity), seed=args.seed + index))
            predictions = model.predict(source=images, imgsz=args.imgsz, device=device, verbose=False)
            y_true = [int(row["label"]) for row in rows]
            y_pred = [int(result.probs.top1) for result in predictions]
            metrics = classification_metrics(y_true, y_pred)
            result_rows.append(
                {
                    "corruption": corruption,
                    "severity": int(severity),
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "cohen_kappa": metrics["cohen_kappa"],
                }
            )

    with (output_dir / "top5_noise_by_severity.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    summary = []
    for corruption in TOP5_CORRUPTIONS:
        selected = [row for row in result_rows if row["corruption"] == corruption]
        summary.append(
            {
                "corruption": corruption,
                "mean_macro_f1_s1_s3": float(np.mean([row["macro_f1"] for row in selected])),
                "mean_accuracy_s1_s3": float(np.mean([row["accuracy"] for row in selected])),
                "mean_cohen_kappa_s1_s3": float(np.mean([row["cohen_kappa"] for row in selected])),
            }
        )
    with (output_dir / "top5_noise_mean_summary.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    write_json(
        output_dir / "run_metadata.json",
        {
            "seed": args.seed,
            "weights": str(Path(args.weights).resolve()),
            "corruptions": list(TOP5_CORRUPTIONS),
            "severities": noise_config["severities"],
            "note": "Corruptions generated in memory; no corrupted image folders saved.",
        },
    )
    print(f"Wrote noise robustness results: {output_dir}")


if __name__ == "__main__":
    main()
