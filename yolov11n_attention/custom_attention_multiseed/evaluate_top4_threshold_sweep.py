"""Evaluate top-4 multiseed checkpoints with confidence/NMS threshold sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

if __name__ == "__main__" and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_top4_multiseed import (  # noqa: E402
    IMAGE_EXTENSIONS,
    REPO_ROOT,
    TOP4_KEYS,
    append_csv_row,
    copy_dataset_snapshot,
    count_labeled_images,
    expected_run_dir,
    find_image_for_label,
    load_project_config,
    load_seeds,
    register_attention_modules,
    repo_path,
    selected_modules,
    write_data_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to top4_modules.yaml.")
    parser.add_argument("--seeds", default=None, help="Path to seeds.yaml.")
    parser.add_argument("--base-data-dir", default=None, help="Prepared grouped split dataset directory.")
    parser.add_argument("--data-yaml", default=None, help="Source data.yaml for class names and nc.")
    parser.add_argument("--module", action="append", choices=TOP4_KEYS, help="Evaluate only this module key. Repeatable.")
    parser.add_argument("--seed", action="append", type=int, help="Evaluate only this seed. Repeatable.")
    parser.add_argument("--conf", action="append", type=float, help="Evaluate only this confidence threshold. Repeatable.")
    parser.add_argument("--iou", action="append", type=float, help="Evaluate only this NMS IoU threshold. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned evaluations without running YOLO.")
    return parser.parse_args()


def existing_sweep_keys(path: Path) -> set[tuple[str, int, float, float]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int, float, float]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                keys.add((row["module"], int(row["seed"]), round(float(row["conf"]), 4), round(float(row["iou"]), 4)))
            except Exception:
                continue
    return keys


def read_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_run_dataset(
    config: dict[str, Any],
    module_key: str,
    seed: int,
    run_dir: Path,
    base_data_dir: Path | None,
    base_data_yaml: Path | None,
) -> tuple[Path, Path]:
    metadata = read_metadata(run_dir)
    data_yaml = repo_path(metadata.get("data_yaml"))
    dataset_dir = repo_path(metadata.get("dataset_dir"))
    if data_yaml and data_yaml.exists() and dataset_dir and dataset_dir.exists():
        return dataset_dir, data_yaml

    train_cfg = config["training"]
    snapshot_root = repo_path(train_cfg["dataset_snapshot_root"])
    assert snapshot_root is not None
    dataset_dir = snapshot_root / module_key / f"seed_{seed}" / "dataset"
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        return dataset_dir, data_yaml

    if base_data_dir is None or base_data_yaml is None:
        raise FileNotFoundError(
            f"No dataset metadata/snapshot found for {module_key} seed {seed}; pass --base-data-dir and --data-yaml."
        )
    dataset_dir = copy_dataset_snapshot(base_data_dir, dataset_dir)
    data_yaml = write_data_yaml(base_data_yaml, dataset_dir, data_yaml)
    return dataset_dir, data_yaml


def copy_split_by_label_state(src_dataset: Path, dst_dataset: Path, split: str, want_labeled: bool) -> int:
    src_images = src_dataset / split / "images"
    src_labels = src_dataset / split / "labels"
    dst_images = dst_dataset / split / "images"
    dst_labels = dst_dataset / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for label_path in sorted(src_labels.glob("*.txt")):
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if bool(lines) != want_labeled:
            continue
        image_path = find_image_for_label(src_images, label_path.name)
        if image_path is None:
            continue
        image_dst = dst_images / image_path.name
        label_dst = dst_labels / label_path.name
        if not image_dst.exists():
            import shutil

            shutil.copy2(image_path, image_dst)
        if not label_dst.exists():
            import shutil

            shutil.copy2(label_path, label_dst)
        copied += 1
    return copied


def make_state_eval_dataset(
    config: dict[str, Any],
    module_key: str,
    seed: int,
    src_dataset: Path,
    source_data_yaml: Path,
    state_name: str,
    want_labeled: bool,
) -> tuple[Path, Path, dict[str, int]]:
    eval_root = repo_path(config["evaluation"]["eval_dataset_root"])
    assert eval_root is not None
    dst = eval_root / module_key / f"seed_{seed}" / f"dataset_{state_name}_eval"
    yaml_path = dst / f"data_{state_name}.yaml"
    if yaml_path.exists():
        copied = {
            "valid": count_labeled_images(dst / "valid" / "labels")["labeled_images" if want_labeled else "healthy_images"],
            "test": count_labeled_images(dst / "test" / "labels")["labeled_images" if want_labeled else "healthy_images"],
        }
        return dst, yaml_path, copied

    for sub in ("images", "labels"):
        (dst / "train" / sub).mkdir(parents=True, exist_ok=True)
    copied = {}
    for split in ("valid", "test"):
        copied[split] = copy_split_by_label_state(src_dataset, dst, split, want_labeled=want_labeled)
    yaml_path = write_data_yaml(source_data_yaml, dst, yaml_path)
    print(f"{state_name} eval dataset for {module_key} seed {seed}: {copied}")
    return dst, yaml_path, copied


def metric_value(metrics: Any, dotted_path: str, default: float = float("nan")) -> float:
    obj = metrics
    for part in dotted_path.split("."):
        if isinstance(obj, dict):
            if part not in obj:
                return default
            obj = obj[part]
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return default
    try:
        return float(obj)
    except Exception:
        return default


def result_dict_value(metrics: Any, keys: tuple[str, ...], default: float = float("nan")) -> float:
    results_dict = getattr(metrics, "results_dict", None)
    if not isinstance(results_dict, dict):
        return default
    for key in keys:
        if key in results_dict:
            try:
                return float(results_dict[key])
            except Exception:
                return default
    return default


def extract_val_metrics(metrics: Any, prefix: str) -> dict[str, float]:
    speed = getattr(metrics, "speed", {}) or {}
    preprocess_ms = float(speed.get("preprocess", float("nan"))) if isinstance(speed, dict) else float("nan")
    inference_ms = float(speed.get("inference", float("nan"))) if isinstance(speed, dict) else float("nan")
    postprocess_ms = float(speed.get("postprocess", float("nan"))) if isinstance(speed, dict) else float("nan")
    total_ms = sum(v for v in (preprocess_ms, inference_ms, postprocess_ms) if not math.isnan(v))
    fps = 1000.0 / total_ms if total_ms > 0 else float("nan")
    return {
        f"{prefix}_box_precision": metric_value(metrics, "box.mp", result_dict_value(metrics, ("metrics/precision(B)",))),
        f"{prefix}_box_recall": metric_value(metrics, "box.mr", result_dict_value(metrics, ("metrics/recall(B)",))),
        f"{prefix}_box_map50": metric_value(metrics, "box.map50", result_dict_value(metrics, ("metrics/mAP50(B)",))),
        f"{prefix}_box_map50_95": metric_value(metrics, "box.map", result_dict_value(metrics, ("metrics/mAP50-95(B)",))),
        f"{prefix}_mask_precision": metric_value(metrics, "seg.mp", result_dict_value(metrics, ("metrics/precision(M)",))),
        f"{prefix}_mask_recall": metric_value(metrics, "seg.mr", result_dict_value(metrics, ("metrics/recall(M)",))),
        f"{prefix}_mask_map50": metric_value(metrics, "seg.map50", result_dict_value(metrics, ("metrics/mAP50(M)",))),
        f"{prefix}_mask_map50_95": metric_value(metrics, "seg.map", result_dict_value(metrics, ("metrics/mAP50-95(M)",))),
        f"{prefix}_preprocess_ms": preprocess_ms,
        f"{prefix}_inference_ms": inference_ms,
        f"{prefix}_postprocess_ms": postprocess_ms,
        f"{prefix}_fps": fps,
    }


def image_paths_in_dir(images_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(images_dir.glob(f"*{ext}"))
    return sorted(image_paths)


def count_prediction_errors(model: Any, images_dir: Path, labels_dir: Path, imgsz: int, conf: float, iou: float) -> dict[str, Any]:
    image_paths = image_paths_in_dir(images_dir)
    if not image_paths:
        return {
            "images": 0,
            "gt_total": 0,
            "pred_box_total": 0,
            "pred_mask_total": 0,
            "box_count_mae": float("nan"),
            "mask_count_mae": float("nan"),
            "box_count_exact": float("nan"),
            "mask_count_exact": float("nan"),
            "disease_images": 0,
            "disease_box_miss_images": 0,
            "disease_mask_miss_images": 0,
            "disease_box_miss_rate": float("nan"),
            "disease_mask_miss_rate": float("nan"),
        }

    results = model.predict(source=[str(p) for p in image_paths], imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    box_errors: list[float] = []
    mask_errors: list[float] = []
    box_exact: list[float] = []
    mask_exact: list[float] = []
    gt_total = 0
    pred_box_total = 0
    pred_mask_total = 0
    disease_images = 0
    disease_box_miss_images = 0
    disease_mask_miss_images = 0

    for image_path, result in zip(image_paths, results):
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_count = 0
        if label_path.exists():
            gt_count = len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        box_count = len(result.boxes) if result.boxes is not None else 0
        mask_count = len(result.masks) if result.masks is not None else 0
        denom = max(1, gt_count)
        box_errors.append(abs(box_count - gt_count) / denom)
        mask_errors.append(abs(mask_count - gt_count) / denom)
        box_exact.append(float(box_count == gt_count))
        mask_exact.append(float(mask_count == gt_count))
        if gt_count > 0:
            disease_images += 1
            disease_box_miss_images += int(box_count == 0)
            disease_mask_miss_images += int(mask_count == 0)
        gt_total += gt_count
        pred_box_total += box_count
        pred_mask_total += mask_count

    return {
        "images": len(image_paths),
        "gt_total": gt_total,
        "pred_box_total": pred_box_total,
        "pred_mask_total": pred_mask_total,
        "box_count_mae": sum(box_errors) / len(box_errors),
        "mask_count_mae": sum(mask_errors) / len(mask_errors),
        "box_count_exact": sum(box_exact) / len(box_exact),
        "mask_count_exact": sum(mask_exact) / len(mask_exact),
        "disease_images": disease_images,
        "disease_box_miss_images": disease_box_miss_images,
        "disease_mask_miss_images": disease_mask_miss_images,
        "disease_box_miss_rate": disease_box_miss_images / disease_images if disease_images else float("nan"),
        "disease_mask_miss_rate": disease_mask_miss_images / disease_images if disease_images else float("nan"),
    }


def healthy_false_positive_summary(model: Any, images_dir: Path, imgsz: int, conf: float, iou: float) -> dict[str, Any]:
    image_paths = image_paths_in_dir(images_dir)
    if not image_paths:
        return {
            "healthy_images": 0,
            "healthy_images_with_box_fp": 0,
            "healthy_images_with_mask_fp": 0,
            "healthy_box_fp_rate": float("nan"),
            "healthy_mask_fp_rate": float("nan"),
            "healthy_fp_boxes_total": 0,
            "healthy_fp_masks_total": 0,
            "healthy_fp_boxes_per_image": float("nan"),
            "healthy_fp_masks_per_image": float("nan"),
            "healthy_avg_fp_confidence": float("nan"),
        }

    results = model.predict(source=[str(p) for p in image_paths], imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    images_with_box_fp = 0
    images_with_mask_fp = 0
    box_total = 0
    mask_total = 0
    confidences: list[float] = []
    for result in results:
        box_count = len(result.boxes) if result.boxes is not None else 0
        mask_count = len(result.masks) if result.masks is not None else 0
        if box_count > 0:
            images_with_box_fp += 1
            try:
                confidences.extend([float(v) for v in result.boxes.conf.detach().cpu().tolist()])
            except Exception:
                pass
        if mask_count > 0:
            images_with_mask_fp += 1
        box_total += box_count
        mask_total += mask_count

    n = len(image_paths)
    return {
        "healthy_images": n,
        "healthy_images_with_box_fp": images_with_box_fp,
        "healthy_images_with_mask_fp": images_with_mask_fp,
        "healthy_box_fp_rate": images_with_box_fp / n,
        "healthy_mask_fp_rate": images_with_mask_fp / n,
        "healthy_fp_boxes_total": box_total,
        "healthy_fp_masks_total": mask_total,
        "healthy_fp_boxes_per_image": box_total / n,
        "healthy_fp_masks_per_image": mask_total / n,
        "healthy_avg_fp_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }


def healthy_aware_score(labeled_map50: float, count_summary: dict[str, Any], healthy_fp_summary: dict[str, Any], weights: dict[str, Any]) -> float:
    return (
        labeled_map50
        - float(weights["count_penalty_weight"]) * float(count_summary["mask_count_mae"])
        - float(weights["disease_miss_penalty_weight"]) * float(count_summary["disease_box_miss_rate"])
        - float(weights["healthy_fp_penalty_weight"]) * float(healthy_fp_summary["healthy_mask_fp_rate"])
    )


def params_count(model: Any) -> int | None:
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        return None


def evaluate_one_threshold(
    config: dict[str, Any],
    module: dict[str, Any],
    seed: int,
    checkpoint: Path,
    dataset_dir: Path,
    data_yaml: Path,
    source_data_yaml: Path,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    from ultralytics import YOLO

    eval_cfg = config["evaluation"]
    imgsz = int(config["training"]["imgsz"])
    module_key = module["key"]
    register_attention_modules()
    model = YOLO(str(checkpoint))
    params = params_count(model)

    labeled_dir, labeled_yaml, labeled_copied = make_state_eval_dataset(
        config, module_key, seed, dataset_dir, source_data_yaml, "labeled_only", want_labeled=True
    )
    healthy_dir, healthy_yaml, healthy_copied = make_state_eval_dataset(
        config, module_key, seed, dataset_dir, source_data_yaml, "healthy_only", want_labeled=False
    )

    start = time.time()
    full_val_metrics = model.val(data=str(data_yaml), split="val", imgsz=imgsz, conf=conf, iou=iou, plots=False, verbose=False)
    full_test_metrics = model.val(data=str(data_yaml), split="test", imgsz=imgsz, conf=conf, iou=iou, plots=False, verbose=False)
    labeled_test_metrics = model.val(
        data=str(labeled_yaml), split="test", imgsz=imgsz, conf=conf, iou=iou, plots=False, verbose=False
    )
    eval_time_sec = time.time() - start

    labeled_test_count = count_prediction_errors(
        model, labeled_dir / "test" / "images", labeled_dir / "test" / "labels", imgsz=imgsz, conf=conf, iou=iou
    )
    full_test_count = count_prediction_errors(
        model, dataset_dir / "test" / "images", dataset_dir / "test" / "labels", imgsz=imgsz, conf=conf, iou=iou
    )
    healthy_test_fp = healthy_false_positive_summary(
        model, healthy_dir / "test" / "images", imgsz=imgsz, conf=conf, iou=iou
    )

    row: dict[str, Any] = {
        "module": module_key,
        "display_name": module.get("display_name", module_key),
        "seed": seed,
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "conf": conf,
        "iou": iou,
        "params": params,
        "eval_time_sec": round(eval_time_sec, 4),
        "labeled_eval_valid_images": labeled_copied.get("valid"),
        "labeled_eval_test_images": labeled_copied.get("test"),
        "healthy_eval_valid_images": healthy_copied.get("valid"),
        "healthy_eval_test_images": healthy_copied.get("test"),
    }
    row.update(extract_val_metrics(full_val_metrics, "val"))
    row.update(extract_val_metrics(full_test_metrics, "full_test"))
    row.update(extract_val_metrics(labeled_test_metrics, "labeled_test"))

    for key, value in labeled_test_count.items():
        row[f"labeled_test_{key}"] = value
    for key, value in full_test_count.items():
        row[f"full_test_count_{key}"] = value
    for key, value in healthy_test_fp.items():
        row[f"healthy_test_{key}"] = value

    row["healthy_aware_score"] = healthy_aware_score(
        row["labeled_test_mask_map50"], labeled_test_count, healthy_test_fp, config["healthy_aware_score"]
    )
    row["primary_default_threshold"] = (
        round(float(conf), 4) == round(float(eval_cfg["default_conf"]), 4)
        and round(float(iou), 4) == round(float(eval_cfg["default_iou"]), 4)
    )
    return row


def main() -> None:
    args = parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    config = load_project_config(args.config)
    seeds = load_seeds(args.seeds)
    if args.seed:
        requested = set(args.seed)
        seeds = [seed for seed in seeds if seed in requested]
    modules = selected_modules(config, set(args.module) if args.module else None)

    dataset_cfg = config.get("dataset", {})
    base_data_dir = repo_path(args.base_data_dir or dataset_cfg.get("base_path"))
    base_data_yaml = repo_path(args.data_yaml or dataset_cfg.get("data_yaml"))
    conf_list = [float(v) for v in (args.conf if args.conf else config["evaluation"]["conf_list"])]
    iou_list = [float(v) for v in (args.iou if args.iou else config["evaluation"]["iou_list"])]

    sweep_csv = repo_path(config["evaluation"]["threshold_sweep_csv"])
    errors_csv = repo_path(config["evaluation"]["threshold_sweep_errors_csv"])
    assert sweep_csv is not None and errors_csv is not None
    done_keys = existing_sweep_keys(sweep_csv)

    planned = [
        (module, seed, conf, iou)
        for module in modules
        for seed in seeds
        for conf in conf_list
        for iou in iou_list
    ]
    print(f"Planned evaluations: {len(planned)}")
    if args.dry_run:
        for module, seed, conf, iou in planned:
            run_dir = expected_run_dir(config, module["key"], seed)
            print(f"DRY-RUN module={module['key']} seed={seed} conf={conf} iou={iou} checkpoint={run_dir / 'weights' / 'best.pt'}")
        return

    for module, seed, conf, iou in planned:
        module_key = module["key"]
        key = (module_key, seed, round(conf, 4), round(iou, 4))
        if key in done_keys:
            print(f"SKIP existing sweep row: module={module_key} seed={seed} conf={conf} iou={iou}")
            continue

        run_dir = expected_run_dir(config, module_key, seed)
        checkpoint = run_dir / "weights" / "best.pt"
        if not checkpoint.exists():
            row = {
                "module": module_key,
                "display_name": module.get("display_name", module_key),
                "seed": seed,
                "conf": conf,
                "iou": iou,
                "checkpoint": str(checkpoint),
                "status": "missing_checkpoint",
                "error": f"Checkpoint not found: {checkpoint}",
            }
            print(row["error"])
            append_csv_row(errors_csv, row)
            continue

        try:
            dataset_dir, data_yaml = ensure_run_dataset(config, module_key, seed, run_dir, base_data_dir, base_data_yaml)
            row = evaluate_one_threshold(
                config=config,
                module=module,
                seed=seed,
                checkpoint=checkpoint,
                dataset_dir=dataset_dir,
                data_yaml=data_yaml,
                source_data_yaml=base_data_yaml or data_yaml,
                conf=conf,
                iou=iou,
            )
            append_csv_row(sweep_csv, row)
            done_keys.add(key)
            print(
                "OK "
                f"module={module_key} seed={seed} conf={conf} iou={iou} "
                f"score={row['healthy_aware_score']:.4f} "
                f"mask_mAP50={row['labeled_test_mask_map50']:.4f} "
                f"healthy_fp={row['healthy_test_healthy_mask_fp_rate']:.4f}"
            )
        except Exception as exc:  # noqa: BLE001 - continue through grid
            row = {
                "module": module_key,
                "display_name": module.get("display_name", module_key),
                "seed": seed,
                "conf": conf,
                "iou": iou,
                "checkpoint": str(checkpoint),
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            print(f"ERROR module={module_key} seed={seed} conf={conf} iou={iou}: {exc.__class__.__name__}: {exc}")
            append_csv_row(errors_csv, row)


if __name__ == "__main__":
    main()
