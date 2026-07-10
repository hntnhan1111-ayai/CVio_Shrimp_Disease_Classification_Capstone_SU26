"""Run the top-4 custom attention modules across fixed random seeds.

This script intentionally does not rebuild the grouped/no-leakage split. It
expects the baseline-prepared dataset directory to already contain train,
valid, and test splits, then copies that split into per-module/per-seed
snapshots so training cannot mutate the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TOP4_KEYS = (
    "triplet_attention_segment_head",
    "cesa_lite_segment_head",
    "context_suppression_gate_lite",
    "sge_eca_head_gate",
)


def repo_path(path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return data


def default_config_file(filename: str) -> Path:
    """Prefer the organized configs/ layout, but keep flat-layout fallback."""

    organized = SCRIPT_DIR / "configs" / filename
    if organized.exists():
        return organized
    return SCRIPT_DIR / filename


def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(repo_path(config_path) if config_path else default_config_file("top4_modules.yaml"))
    modules = config.get("modules", [])
    keys = [module.get("key") for module in modules]
    unexpected = sorted(set(keys) - set(TOP4_KEYS))
    missing = sorted(set(TOP4_KEYS) - set(keys))
    if unexpected or missing:
        raise ValueError(f"Top-4 module config mismatch. missing={missing}, unexpected={unexpected}")
    return config


def load_seeds(seeds_path: str | Path | None = None) -> list[int]:
    data = load_yaml(repo_path(seeds_path) if seeds_path else default_config_file("seeds.yaml"))
    seeds = [int(seed) for seed in data.get("seeds", [])]
    if seeds != [42, 3407, 2026]:
        raise ValueError(f"Expected seeds [42, 3407, 2026], got {seeds}")
    baseline_seed = int(data.get("baseline_seed", 42))
    if baseline_seed not in seeds:
        raise ValueError(f"Baseline seed {baseline_seed} must be present in seeds {seeds}")
    return seeds


def selected_modules(config: dict[str, Any], only: set[str] | None = None) -> list[dict[str, Any]]:
    modules = [module for module in config["modules"] if module["key"] in TOP4_KEYS]
    if only:
        modules = [module for module in modules if module["key"] in only]
    priority = {key: idx for idx, key in enumerate(TOP4_KEYS)}
    return sorted(modules, key=lambda item: priority[item["key"]])


def register_attention_modules() -> None:
    """Register both legacy and priority attention classes in one parser patch."""

    import ultralytics.nn.tasks as tasks

    from custom_attention.common.attention_modules import (
        AttentionGate,
        CALiteSpatialGate,
        CESALite,
        LowFPCBAMLite,
        NAMAttention,
        TripletAttention,
    )
    from custom_attention._shared.attention_modules import (
        BoundaryAwareLiteAttention,
        CASpatialLowFPGate,
        CESLite,
        ContextSuppressionGateLite,
        LowFPResidualSpatialGate,
        P3P4SemanticAttentionGate,
        PrototypeAwareMaskGateLite,
        SGEECAHeadGate,
    )

    single_input_modules = (
        CESALite,
        TripletAttention,
        CALiteSpatialGate,
        NAMAttention,
        LowFPCBAMLite,
        SGEECAHeadGate,
        CESLite,
        LowFPResidualSpatialGate,
        P3P4SemanticAttentionGate,
        BoundaryAwareLiteAttention,
        ContextSuppressionGateLite,
        CASpatialLowFPGate,
        PrototypeAwareMaskGateLite,
    )
    for module_cls in (*single_input_modules, AttentionGate):
        setattr(tasks, module_cls.__name__, module_cls)

    tasks.CUSTOM_ATTENTION_MULTISEED_MODULES = single_input_modules
    tasks.AttentionGate = AttentionGate

    if getattr(tasks, "_shrimp_multiseed_attention_parse_patched", False):
        return

    source = inspect.getsource(tasks.parse_model)
    marker = "        elif m in frozenset(\n            {\n                Detect,"
    insert = """        elif m in CUSTOM_ATTENTION_MULTISEED_MODULES:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m is AttentionGate:
            c1 = ch[f[0]]
            cg = ch[f[1]]
            c2 = c1
            args = [c1, cg, *args]
"""
    if marker not in source:
        raise RuntimeError("Could not patch ultralytics.nn.tasks.parse_model: Detect marker not found.")

    patched = source.replace(marker, insert + marker, 1)
    exec(compile(patched, "<shrimp_multiseed_attention_parse_model>", "exec"), tasks.__dict__)
    tasks._shrimp_multiseed_attention_parse_patched = True


def disable_ultralytics_albumentations() -> None:
    try:
        import ultralytics.data.augment as yolo_augment
    except Exception as exc:  # noqa: BLE001
        print(f"Could not patch Ultralytics Albumentations hook: {exc}")
        return

    class NoOpAlbumentations:
        contains_spatial = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.transform = None

        def __call__(self, labels: Any) -> Any:
            return labels

    yolo_augment.Albumentations = NoOpAlbumentations
    print("Ultralytics Albumentations hook disabled for this run.")


def find_image_for_label(image_dir: Path, label_file: str | Path) -> Path | None:
    stem = Path(label_file).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def count_labeled_images(label_dir: Path) -> dict[str, int]:
    labeled = 0
    healthy = 0
    instances = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            labeled += 1
            instances += len(lines)
        else:
            healthy += 1
    return {"labeled_images": labeled, "healthy_images": healthy, "instances": instances}


def write_data_yaml(source_data_yaml: Path, dataset_dir: Path, yaml_path: Path, val_dir: str = "valid", test_dir: str = "test") -> Path:
    if not source_data_yaml.exists():
        raise FileNotFoundError(f"Source data YAML not found: {source_data_yaml}")
    with source_data_yaml.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    content["train"] = str(dataset_dir / "train" / "images")
    content["val"] = str(dataset_dir / val_dir / "images")
    content["test"] = str(dataset_dir / test_dir / "images")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)
    return yaml_path


def validate_split_exists(dataset_dir: Path) -> None:
    missing = []
    for split in ("train", "valid", "test"):
        for subdir in ("images", "labels"):
            path = dataset_dir / split / subdir
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Prepared grouped split is incomplete. Missing: " + ", ".join(missing))


def copy_dataset_snapshot(source_dataset: Path, snapshot_dir: Path) -> Path:
    validate_split_exists(source_dataset)
    if snapshot_dir.exists():
        validate_split_exists(snapshot_dir)
        print(f"Dataset snapshot already exists, reusing: {snapshot_dir}")
        return snapshot_dir
    print(f"Copying fixed split dataset snapshot: {source_dataset} -> {snapshot_dir}")
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("runs", "*.cache", ".clahe_applied")
    shutil.copytree(source_dataset, snapshot_dir, ignore=ignore)
    validate_split_exists(snapshot_dir)
    return snapshot_dir


def read_best_epoch_from_results(run_path: Path) -> dict[str, Any]:
    results_csv = run_path / "results.csv"
    if not results_csv.exists():
        return {}
    try:
        import pandas as pd
    except Exception:
        return {"results_csv": str(results_csv)}
    df = pd.read_csv(results_csv)
    df.columns = [str(c).strip() for c in df.columns]
    mask_col = "metrics/mAP50(M)"
    if mask_col not in df.columns:
        return {"epochs_ran": int(len(df)), "results_csv": str(results_csv)}
    best_idx = df[mask_col].idxmax()
    first = df.iloc[0]
    best = df.iloc[best_idx]
    last = df.iloc[-1]
    return {
        "epochs_ran": int(len(df)),
        "best_epoch_by_mask_map50": int(best["epoch"]) if "epoch" in df.columns else int(best_idx + 1),
        "first_train_seg_loss": float(first.get("train/seg_loss", float("nan"))),
        "best_val_mask_map50": float(best.get(mask_col, float("nan"))),
        "best_val_mask_map50_95": float(best.get("metrics/mAP50-95(M)", float("nan"))),
        "last_val_mask_map50": float(last.get(mask_col, float("nan"))),
        "last_val_mask_map50_95": float(last.get("metrics/mAP50-95(M)", float("nan"))),
        "last_train_seg_loss": float(last.get("train/seg_loss", float("nan"))),
        "last_val_seg_loss": float(last.get("val/seg_loss", float("nan"))),
        "seg_loss_gap_val_minus_train": float(
            last.get("val/seg_loss", float("nan")) - last.get("train/seg_loss", float("nan"))
        ),
        "results_csv": str(results_csv),
    }


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_fields: list[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_fields = next(reader, [])
    fieldnames = list(existing_fields)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

    rows: list[dict[str, Any]] = []
    if path.exists() and existing_fields:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def expected_run_dir(config: dict[str, Any], module_key: str, seed: int) -> Path:
    output_root = repo_path(config["training"]["output_root"])
    assert output_root is not None
    return output_root / module_key / f"seed_{seed}"


def train_one(
    config: dict[str, Any],
    module: dict[str, Any],
    seed: int,
    base_data_dir: Path,
    base_data_yaml: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    from ultralytics import YOLO

    train_cfg = config["training"]
    module_key = module["key"]
    model_yaml = repo_path(module["model_yaml"])
    assert model_yaml is not None
    if not model_yaml.exists():
        raise FileNotFoundError(f"Model YAML not found for {module_key}: {model_yaml}")

    run_dir = expected_run_dir(config, module_key, seed)
    best_path = run_dir / "weights" / "best.pt"
    metadata_path = run_dir / "run_metadata.json"
    row: dict[str, Any] = {
        "module": module_key,
        "display_name": module.get("display_name", module_key),
        "seed": seed,
        "model_yaml": str(model_yaml),
        "output_dir": str(run_dir),
        "best_checkpoint": str(best_path),
    }
    if best_path.exists():
        row.update({"status": "skipped_existing", "reason": "best checkpoint already exists"})
        print(f"SKIP existing checkpoint: {best_path}")
        return row
    if run_dir.exists():
        row.update({"status": "skipped_existing_incomplete", "reason": "run directory exists without best.pt"})
        print(f"SKIP existing incomplete run directory: {run_dir}")
        return row

    dataset_root = repo_path(train_cfg["dataset_snapshot_root"])
    assert dataset_root is not None
    snapshot_dir = dataset_root / module_key / f"seed_{seed}" / "dataset"
    dataset_dir = copy_dataset_snapshot(base_data_dir, snapshot_dir)
    data_yaml = write_data_yaml(base_data_yaml, dataset_dir, dataset_dir / "data.yaml")

    split_counts = {
        split: count_labeled_images(dataset_dir / split / "labels")
        for split in ("train", "valid", "test")
    }

    imgsz = 320 if smoke else int(train_cfg["imgsz"])
    epochs = 1 if smoke else int(train_cfg["epochs"])
    batch = 8 if smoke else int(train_cfg["batch"])
    patience = 1 if smoke else int(train_cfg["patience"])
    plots = bool(train_cfg.get("plots", True)) and not smoke
    verbose = bool(train_cfg.get("verbose", True))
    train_args = dict(train_cfg.get("clean_light_aug_train_args", {}))

    print("\n" + "#" * 90)
    print(f"Module: {module_key}")
    print(f"Seed: {seed}")
    print(f"Model YAML: {model_yaml}")
    print(f"Data YAML: {data_yaml}")
    print(f"Output directory: {run_dir}")
    print(f"Split counts: {split_counts}")
    print("#" * 90)

    register_attention_modules()
    disable_ultralytics_albumentations()

    yolo = YOLO(str(model_yaml))
    pretrained_weights = train_cfg.get("pretrained_weights")
    if train_cfg.get("load_pretrained_weights", True) and pretrained_weights:
        print(f"Loading pretrained baseline weights: {pretrained_weights}")
        yolo = yolo.load(str(pretrained_weights))

    start = time.time()
    yolo.train(
        data=str(data_yaml),
        task=str(train_cfg.get("task", "segment")),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        seed=seed,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=False,
        pretrained=True,
        plots=plots,
        verbose=verbose,
        **train_args,
    )
    train_time_min = (time.time() - start) / 60.0

    row.update(
        {
            "status": "trained",
            "train_time_min": round(train_time_min, 4),
            "data_yaml": str(data_yaml),
            "dataset_dir": str(dataset_dir),
            "split_counts_json": json.dumps(split_counts, sort_keys=True),
        }
    )
    row.update(read_best_epoch_from_results(run_dir))
    row["best_checkpoint_exists"] = best_path.exists()

    metadata = dict(row)
    metadata["config"] = {
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "patience": patience,
        "train_args": train_args,
        "pretrained_weights": pretrained_weights,
        "load_pretrained_weights": bool(train_cfg.get("load_pretrained_weights", True)),
    }
    write_json(metadata_path, metadata)
    print(f"Train time: {train_time_min:.2f} min")
    print(f"Best checkpoint path: {best_path}")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to top4_modules.yaml.")
    parser.add_argument("--seeds", default=None, help="Path to seeds.yaml.")
    parser.add_argument("--base-data-dir", default=None, help="Prepared grouped split dataset directory.")
    parser.add_argument("--data-yaml", default=None, help="Source data.yaml for class names and nc.")
    parser.add_argument("--module", action="append", choices=TOP4_KEYS, help="Run only this module key. Repeatable.")
    parser.add_argument("--seed", action="append", type=int, help="Run only this seed. Repeatable.")
    parser.add_argument("--smoke", action="store_true", help="Use 1 epoch and 320 image size for pipeline testing.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")
    return parser.parse_args()


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
    if base_data_dir is None or base_data_yaml is None:
        raise ValueError("Dataset paths are required. Set config dataset.base_path/data_yaml or pass CLI args.")

    print(f"Repo root: {REPO_ROOT}")
    print(f"Prepared grouped split dataset: {base_data_dir}")
    print(f"Source data YAML: {base_data_yaml}")
    print(f"Modules: {[module['key'] for module in modules]}")
    print(f"Seeds: {seeds}")

    planned = [(module, seed) for module in modules for seed in seeds]
    if args.dry_run:
        for module, seed in planned:
            run_dir = expected_run_dir(config, module["key"], seed)
            print(f"DRY-RUN module={module['key']} seed={seed} model={repo_path(module['model_yaml'])} output={run_dir}")
        return

    if not base_data_dir.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {base_data_dir}. Run the baseline data-prep cells first or pass --base-data-dir."
        )
    if not base_data_yaml.exists():
        raise FileNotFoundError(f"Source data YAML not found: {base_data_yaml}")

    train_csv = repo_path(config["training"]["train_results_csv"])
    assert train_csv is not None

    for module, seed in planned:
        try:
            row = train_one(config, module, seed, base_data_dir, base_data_yaml, smoke=args.smoke)
        except Exception as exc:  # noqa: BLE001 - continue through the grid
            row = {
                "module": module["key"],
                "display_name": module.get("display_name", module["key"]),
                "seed": seed,
                "model_yaml": str(repo_path(module["model_yaml"])),
                "output_dir": str(expected_run_dir(config, module["key"], seed)),
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            print(f"ERROR module={module['key']} seed={seed}: {exc.__class__.__name__}: {exc}")
        append_csv_row(train_csv, row)


if __name__ == "__main__":
    main()
