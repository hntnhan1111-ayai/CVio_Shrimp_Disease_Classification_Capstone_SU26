"""Portable Ultralytics classification training entry point."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from cvio_asl_ldam.attention.patch_yolo import (
    get_custom_trainer_class,
    register_checkpoint_safe_globals,
)
from cvio_asl_ldam.data.audit_dataset import resolve_dataset_root
from cvio_asl_ldam.data.make_split import create_split, materialize_split_tree
from cvio_asl_ldam.evaluation.confusion_matrix import save_confusion_matrix
from cvio_asl_ldam.evaluation.metrics import classification_metrics
from cvio_asl_ldam.utils.io import load_yaml, write_json
from cvio_asl_ldam.utils.paths import ProjectPaths, resolve_device
from cvio_asl_ldam.utils.seed import seed_everything


def _manifest_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def _evaluate_yolo(weights: Path, dataset_root: Path, manifest: Path, output_dir: Path, device: str) -> None:
    from ultralytics import YOLO

    register_checkpoint_safe_globals()
    model = YOLO(str(weights))
    rows = _manifest_rows(manifest, "test")
    paths = [str(dataset_root / row["rel_path"]) for row in rows]
    predictions = model.predict(
        source=paths,
        imgsz=224,
        device=device.split(",")[0],
        verbose=False,
    )
    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(result.probs.top1) for result in predictions]
    metrics = classification_metrics(y_true, y_pred)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "clean_test_metrics.json", metrics)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        output_dir / "clean_test_confusion_matrix",
    )
    with (output_dir / "clean_test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rel_path", "class_name", "label", "prediction"],
        )
        writer.writeheader()
        for row, prediction in zip(rows, y_pred, strict=True):
            writer.writerow(
                {
                    "rel_path": row["rel_path"],
                    "class_name": row["class_name"],
                    "label": row["label"],
                    "prediction": prediction,
                }
            )


def run(config_path: str | Path, args: argparse.Namespace) -> Path:
    config = load_yaml(config_path)
    paths = ProjectPaths.from_environment()
    dataset_config_path = Path(config["dataset_config"])
    dataset_config = load_yaml(dataset_config_path)
    data_dir = Path(getattr(args, "data_dir", None) or paths.data_dir)
    dataset_root, _candidates = resolve_dataset_root(data_dir, dataset_config)
    output_root = Path(getattr(args, "output_dir", None) or paths.output_dir)
    manifest_dir = output_root / "manifests"
    manifest = manifest_dir / "split_manifest_seed42_generated.csv"
    if not manifest.is_file():
        manifest = create_split(
            dataset_config_path,
            dataset_root,
            manifest_dir,
            int(args.seed or config["seed"]),
        )
    prepared = materialize_split_tree(
        dataset_root,
        manifest,
        output_root / "prepared_seed42",
    )

    seed = int(getattr(args, "seed", None) or config["seed"])
    seed_everything(seed)
    device = resolve_device(getattr(args, "device", None) or config.get("device", "auto"))
    model_name = getattr(args, "model", None) or str(config["model"])
    method = str(config.get("method", "baseline_ce"))
    run_name = f"{model_name.replace('-', '_')}__{method}__seed{seed}"
    project = output_root / "training"
    train_kwargs: dict[str, Any] = {
        "data": str(prepared),
        "task": "classify",
        "imgsz": int(config["imgsz"]),
        "epochs": 1 if getattr(args, "smoke_test", False) else int(config["epochs"]),
        "patience": int(config["patience"]),
        "batch": config["batch"],
        "workers": int(config["workers"]),
        "amp": bool(config["amp"]),
        "device": device,
        "optimizer": str(config["optimizer"]),
        "lr0": float(config.get("lr0", 0.00125)),
        "lrf": float(config.get("lrf", 0.01)),
        "cos_lr": bool(config.get("cos_lr", True)),
        "cache": bool(config.get("cache", False)),
        "seed": seed,
        "project": str(project),
        "name": run_name,
        "exist_ok": True,
        "plots": False,
        "verbose": True,
        "resume": bool(getattr(args, "resume", False)),
    }
    resume_weights = getattr(args, "resume_weights", None)
    if config.get("auto_augment"):
        train_kwargs["auto_augment"] = config["auto_augment"]

    from ultralytics import YOLO

    if resume_weights and getattr(args, "resume", False):
        model = YOLO(str(resume_weights))
    else:
        model = YOLO(f"{model_name}.pt")
    if method == "asl_ldam_simam_dcfr":
        os.environ["CVIO_YOLO_LOSS"] = "asl_ldam"
        os.environ["CVIO_YOLO_ATTENTION"] = "simam_dcfr"
        model.train(trainer=get_custom_trainer_class(), **train_kwargs)
    else:
        os.environ["CVIO_YOLO_LOSS"] = "baseline_ce"
        os.environ["CVIO_YOLO_ATTENTION"] = "none"
        model.train(**train_kwargs)

    run_dir = project / run_name
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without expected checkpoint: {best}")
    write_json(
        run_dir / "paper_run_config.json",
        {
            "config": config,
            "dataset_root": str(dataset_root),
            "manifest": str(manifest),
            "device": device,
            "seed": seed,
            "runtime_reference": "Kaggle T4x2 GPU, Python 3.12.3",
        },
    )
    _evaluate_yolo(best, dataset_root, manifest, run_dir / "paper_evaluation", device)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_dir = run(args.config, args)
    print(f"Completed run: {run_dir}")


if __name__ == "__main__":
    main()
