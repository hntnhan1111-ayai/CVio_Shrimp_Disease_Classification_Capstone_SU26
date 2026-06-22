#!/usr/bin/env python3
"""Evaluate a trained model on the clean seed-42 test set.

Does not train. Outputs metrics JSON + CSV predictions + confusion matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvio_asl_ldam.evaluation.metrics import classification_metrics
from cvio_asl_ldam.evaluation.confusion_matrix import save_confusion_matrix
from cvio_asl_ldam.utils.io import write_json

CLASS_NAMES = ("Healthy", "BG", "WSSV", "WSSV_BG")


def _test_rows(manifest: Path) -> list[dict[str, str]]:
    import csv
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--manifest", default="artifacts/manifests/split_manifest_seed42.csv")
    parser.add_argument("--dataset-config", default="configs/dataset/shrimpdiseasebd_seed42.yaml")
    parser.add_argument("--data-dir", help="Raw dataset root (to resolve image paths).")
    parser.add_argument("--output-dir", default="artifacts/evaluation/clean")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()

    from cvio_asl_ldam.attention.patch_yolo import register_checkpoint_safe_globals
    from cvio_asl_ldam.data.audit_dataset import resolve_dataset_root
    from cvio_asl_ldam.utils.io import load_yaml
    from cvio_asl_ldam.utils.paths import resolve_device

    register_checkpoint_safe_globals()
    from ultralytics import YOLO

    dataset_config = load_yaml(args.dataset_config)
    if args.data_dir:
        dataset_root, _ = resolve_dataset_root(args.data_dir, dataset_config)
    else:
        # Fall back to the prepared split tree.
        dataset_root = Path("runs/prepared_seed42")
        if not dataset_root.is_dir():
            raise SystemExit(
                "Provide --data-dir or run 01_prepare_dataset_and_split.py first."
            )

    rows = _test_rows(Path(args.manifest))
    if not rows:
        raise SystemExit(f"No test rows in manifest: {args.manifest}")
    device = resolve_device(args.device).split(",")[0]
    model = YOLO(args.weights)

    # Resolve image paths. If using prepared tree, the rel_path is under the
    # class folder; map label -> split/class dir.
    def _resolve_path(row: dict[str, str]) -> str:
        if (dataset_root / row["rel_path"]).is_file():
            return str(dataset_root / row["rel_path"])
        # prepared tree layout: <root>/test/NN_<class>/<filename>
        class_dir = f"{int(row['label']):02d}_{row['class_name']}"
        candidate = dataset_root / "test" / class_dir / row["filename"]
        return str(candidate)

    paths = [_resolve_path(row) for row in rows]
    predictions = model.predict(source=paths, imgsz=args.imgsz, device=device, verbose=False)
    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(result.probs.top1) for result in predictions]

    metrics = classification_metrics(y_true, y_pred)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "clean_test_metrics.json", metrics)
    save_confusion_matrix(metrics["confusion_matrix"], output_dir / "clean_test_confusion_matrix")

    import csv
    with (output_dir / "clean_test_predictions.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["rel_path", "class_name", "label", "prediction"])
        writer.writeheader()
        for row, pred in zip(rows, y_pred, strict=True):
            writer.writerow(
                {
                    "rel_path": row["rel_path"],
                    "class_name": row["class_name"],
                    "label": row["label"],
                    "prediction": pred,
                }
            )
    print(f"Clean test accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"Wrote metrics to {output_dir}")


if __name__ == "__main__":
    main()
