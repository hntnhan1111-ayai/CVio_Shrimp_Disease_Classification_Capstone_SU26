"""Evaluation and calibration helpers for custom-attention Group A.

Group A is intentionally inference/protocol focused:
- A1: TTA inference at the default threshold.
- A2: confidence/NMS threshold sweep.
- A3: matched strong-augmentation baseline.

The helpers reuse the same split/evaluation ideas from the clean baseline and
the custom_attention_multiseed pipeline. They do not edit existing notebooks.
"""

from __future__ import annotations

import csv
import glob
import inspect
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70
DEFAULT_CONF_LIST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
DEFAULT_IOU_LIST = [0.50, 0.60, 0.70]

COUNT_PENALTY_WEIGHT = 0.05
DISEASE_MISS_PENALTY_WEIGHT = 0.15
HEALTHY_FP_PENALTY_WEIGHT = 0.10

STRONG_AUG_TRAIN_ARGS: dict[str, Any] = {
    "auto_augment": None,
    "erasing": 0.15,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.3,
    "hsv_h": 0.05,
    "hsv_s": 0.50,
    "hsv_v": 0.40,
    "degrees": 10.0,
    "translate": 0.10,
    "scale": 0.50,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": 0.0,
    "bgr": 0.0,
}


DEFAULT_CANDIDATES: list[dict[str, Any]] = [
    {
        "key": "baseline_clean",
        "display_name": "Baseline clean YOLO11n-seg",
        "family": "baseline",
        "augmentation": "clean_light",
        "model_yaml": "yolo11n-seg.pt",
        "checkpoint_patterns": [
            "/kaggle/working/runs/segment/yolo11n-seg_shrimp_seg_clean_light_aug_baseline_clean_light_aug_baseline/weights/best.pt",
            "/kaggle/working/runs/segment/*clean_light_aug_baseline*/weights/best.pt",
        ],
    },
    {
        "key": "triplet_attention_segment_head",
        "display_name": "Triplet Attention Segment Head",
        "family": "custom_attention",
        "augmentation": "clean_light",
        "model_yaml": "custom_attention/triplet_attention_segment_head/yolov11n_triplet_segment.yaml",
        "checkpoint_patterns": [
            "/kaggle/working/runs/segment/*triplet_attention_segment_head*/weights/best.pt",
            "/kaggle/working/runs/custom_attention/*triplet_attention_segment_head*/weights/best.pt",
            "/kaggle/working/runs/custom_attention_multiseed/triplet_attention_segment_head/seed_42/weights/best.pt",
            "runs/custom_attention_multiseed/triplet_attention_segment_head/seed_42/weights/best.pt",
        ],
    },
    {
        "key": "cote_gate",
        "display_name": "CoTE Gate",
        "family": "custom_attention_research_modules",
        "augmentation": "clean_light",
        "model_yaml": "custom_attention_research_modules/cote_gate/cote_gate.yaml",
        "checkpoint_patterns": [
            "/kaggle/working/runs/custom_attention_research_modules/cote_gate/weights/best.pt",
        ],
    },
    {
        "key": "cote_gate_strong_augmentation",
        "display_name": "CoTE Gate strong augmentation",
        "family": "custom_attention_research_modules",
        "augmentation": "strong",
        "model_yaml": "custom_attention_research_modules/cote_gate/generated_yamls/cote_gate_strong_augmentation.yaml",
        "checkpoint_patterns": [
            "/kaggle/working/runs/custom_attention_research_modules/cote_gate_strong_augmentation/weights/best.pt",
        ],
    },
    {
        "key": "lpsc_gate",
        "display_name": "LPSC Gate",
        "family": "custom_attention_research_modules",
        "augmentation": "clean_light",
        "model_yaml": "custom_attention_research_modules/lpsc_gate/lpsc_gate.yaml",
        "checkpoint_patterns": [
            "/kaggle/working/runs/custom_attention_research_modules/lpsc_gate/weights/best.pt",
        ],
    },
    {
        "key": "lpsc_gate_strong_augmentation",
        "display_name": "LPSC Gate strong augmentation",
        "family": "custom_attention_research_modules",
        "augmentation": "strong",
        "model_yaml": "custom_attention_research_modules/lpsc_gate/generated_yamls/lpsc_gate_strong_augmentation.yaml",
        "checkpoint_patterns": [
            "/kaggle/working/runs/custom_attention_research_modules/lpsc_gate_strong_augmentation/weights/best.pt",
        ],
    },
    {
        "key": "cesa_lite_segment_head",
        "display_name": "CESA-Lite Segment Head",
        "family": "custom_attention",
        "augmentation": "clean_light",
        "model_yaml": "custom_attention/cesa_lite_segment_head/yolov11n_cesa_lite.yaml",
        "checkpoint_patterns": [
            "/kaggle/working/runs/segment/*cesa_lite_segment_head*/weights/best.pt",
            "/kaggle/working/runs/custom_attention/*cesa_lite_segment_head*/weights/best.pt",
            "/kaggle/working/runs/custom_attention_multiseed/cesa_lite_segment_head/seed_42/weights/best.pt",
            "runs/custom_attention_multiseed/cesa_lite_segment_head/seed_42/weights/best.pt",
        ],
    },
    {
        "key": "sge_eca_head_gate",
        "display_name": "SGE-ECA Head Gate",
        "family": "custom_attention",
        "augmentation": "clean_light",
        "model_yaml": "custom_attention/sge_eca_head_gate/model.yaml",
        "checkpoint_patterns": [
            "/kaggle/working/runs/custom_attention/*sge_eca_head_gate*/weights/best.pt",
            "/kaggle/working/runs/segment/*sge_eca_head_gate*/weights/best.pt",
            "/kaggle/working/runs/custom_attention_multiseed/sge_eca_head_gate/seed_42/weights/best.pt",
            "runs/custom_attention_multiseed/sge_eca_head_gate/seed_42/weights/best.pt",
        ],
    },
]


def attention_root() -> Path:
    """Return the yolov11n_attention root when this helper lives in the repo."""

    here = Path(__file__).resolve()
    for parent in [*here.parents, Path.cwd(), Path.cwd() / "yolov11n_attention"]:
        if (parent / "custom_attention").exists() and (parent / "custom_attention_research_modules").exists():
            return parent
    return here.parents[2]


def ensure_project_import_paths() -> Path:
    """Add vendored Ultralytics and attention module roots to sys.path."""

    root = attention_root()
    vendored = root / "ultralytics"
    paths = []
    if (vendored / "ultralytics" / "__init__.py").exists():
        paths.append(vendored)
    paths.extend([root, root.parent, Path.cwd()])
    for path in paths:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


def register_all_custom_attention_modules() -> None:
    """Register all custom attention modules needed to load checkpoints/YAMLs."""

    ensure_project_import_paths()
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
    from custom_attention_research_modules._shared.attention_modules import RESEARCH_ATTENTION_MODULES

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
        *RESEARCH_ATTENTION_MODULES,
    )
    for module_cls in (*single_input_modules, AttentionGate):
        setattr(tasks, module_cls.__name__, module_cls)
        setattr(sys.modules["__main__"], module_cls.__name__, module_cls)

    tasks.NHOMA_SINGLE_INPUT_ATTENTION_MODULES = single_input_modules
    tasks.AttentionGate = AttentionGate

    if not getattr(tasks, "_shrimp_nhoma_attention_parse_patched", False):
        source = inspect.getsource(tasks.parse_model)
        marker = "        elif m in frozenset(\n            {\n                Detect,"
        insert = """        elif m in NHOMA_SINGLE_INPUT_ATTENTION_MODULES:
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
        exec(compile(patched, "<shrimp_nhoma_attention_parse_model>", "exec"), tasks.__dict__)
        tasks._shrimp_nhoma_attention_parse_patched = True

    _try_add_torch_safe_globals()


def _try_add_torch_safe_globals() -> None:
    """Help newer torch versions unpickle checkpoints with custom classes."""

    try:
        import torch
        import ultralytics.nn.tasks as tasks
    except Exception:
        return

    names = {
        candidate
        for candidate in [
            "AttentionGate",
            "CALiteSpatialGate",
            "CESALite",
            "CoTEGate",
            "LPSCGate",
            "SGEECAHeadGate",
            "TripletAttention",
            "CESLite",
            "ContextSuppressionGateLite",
        ]
        if hasattr(tasks, candidate)
    }
    classes = [getattr(tasks, name) for name in sorted(names)]
    try:
        torch.serialization.add_safe_globals(classes)
    except Exception:
        pass


def candidate_by_key(key: str) -> dict[str, Any]:
    for candidate in DEFAULT_CANDIDATES:
        if candidate["key"] == key:
            return dict(candidate)
    raise KeyError(f"Unknown candidate key: {key}")


def selected_candidates(module_keys: list[str] | None = None) -> list[dict[str, Any]]:
    if module_keys is None:
        return [dict(item) for item in DEFAULT_CANDIDATES]
    requested = set(module_keys)
    return [dict(item) for item in DEFAULT_CANDIDATES if item["key"] in requested]


def _path_roots() -> list[Path]:
    root = attention_root()
    roots = [Path.cwd(), root, root.parent]
    unique: list[Path] = []
    for item in roots:
        if item not in unique:
            unique.append(item)
    return unique


def _candidate_pattern_strings(pattern: str | Path) -> list[str]:
    text = str(pattern)
    path = Path(text)
    if path.is_absolute():
        return [text]
    return [str(root / text) for root in _path_roots()]


def resolve_existing_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    for text in _candidate_pattern_strings(path_value):
        path = Path(text)
        if path.exists():
            return path
    return None


def glob_existing(pattern: str | Path) -> list[Path]:
    matches: list[Path] = []
    for text in _candidate_pattern_strings(pattern):
        for value in glob.glob(text):
            path = Path(value)
            if path.exists() and path not in matches:
                matches.append(path)
    return sorted(matches, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def resolve_checkpoint(candidate: dict[str, Any], overrides: dict[str, str | Path] | None = None) -> Path | None:
    overrides = overrides or {}
    override = overrides.get(candidate["key"])
    if override:
        path = resolve_existing_path(override)
        if path is None:
            raise FileNotFoundError(f"Override checkpoint does not exist for {candidate['key']}: {override}")
        return path

    for pattern in candidate.get("checkpoint_patterns", []):
        matches = glob_existing(pattern)
        if matches:
            return matches[0]
    return None


def checkpoint_table(
    module_keys: list[str] | None = None,
    overrides: dict[str, str | Path] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in selected_candidates(module_keys):
        checkpoint = resolve_checkpoint(candidate, overrides=overrides)
        model_yaml = resolve_existing_path(candidate.get("model_yaml"))
        rows.append(
            {
                "module_key": candidate["key"],
                "display_name": candidate["display_name"],
                "family": candidate.get("family"),
                "augmentation": candidate.get("augmentation"),
                "checkpoint": str(checkpoint) if checkpoint else "",
                "checkpoint_found": checkpoint is not None,
                "model_yaml": str(model_yaml or candidate.get("model_yaml", "")),
                "model_yaml_found": bool(model_yaml) or str(candidate.get("model_yaml", "")).endswith(".pt"),
            }
        )
    return rows


def validate_split_exists(dataset_dir: Path) -> None:
    missing = []
    for split in ("train", "valid", "test"):
        for subdir in ("images", "labels"):
            path = dataset_dir / split / subdir
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Prepared grouped split is incomplete. Missing: " + ", ".join(missing))


def remove_yolo_label_caches(root: Path) -> None:
    for cache_path in Path(root).glob("**/*.cache"):
        cache_path.unlink()


def copy_dataset_snapshot(source_dataset: Path, snapshot_dir: Path, refresh: bool = False) -> Path:
    validate_split_exists(source_dataset)
    if snapshot_dir.exists() and refresh:
        shutil.rmtree(snapshot_dir)
    if snapshot_dir.exists():
        validate_split_exists(snapshot_dir)
        remove_yolo_label_caches(snapshot_dir)
        return snapshot_dir
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("runs", "*.cache", ".clahe_applied")
    shutil.copytree(source_dataset, snapshot_dir, ignore=ignore)
    validate_split_exists(snapshot_dir)
    remove_yolo_label_caches(snapshot_dir)
    return snapshot_dir


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
            shutil.copy2(image_path, image_dst)
        if not label_dst.exists():
            shutil.copy2(label_path, label_dst)
        copied += 1
    return copied


def make_state_eval_dataset(
    src_dataset: Path,
    source_data_yaml: Path,
    output_root: Path,
    tag: str,
    state_name: str,
    want_labeled: bool,
) -> tuple[Path, Path, dict[str, int]]:
    dst = output_root / "eval_datasets" / tag / f"dataset_{state_name}_eval"
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


def count_prediction_errors(
    model: Any,
    images_dir: Path,
    labels_dir: Path,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    augment: bool = False,
) -> dict[str, Any]:
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

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        augment=augment,
        verbose=False,
    )
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


def healthy_false_positive_summary(
    model: Any,
    images_dir: Path,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    augment: bool = False,
) -> dict[str, Any]:
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

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        augment=augment,
        verbose=False,
    )
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


def healthy_aware_score(labeled_map50: float, count_summary: dict[str, Any], healthy_fp_summary: dict[str, Any]) -> float:
    return (
        float(labeled_map50)
        - COUNT_PENALTY_WEIGHT * float(count_summary["mask_count_mae"])
        - DISEASE_MISS_PENALTY_WEIGHT * float(count_summary["disease_box_miss_rate"])
        - HEALTHY_FP_PENALTY_WEIGHT * float(healthy_fp_summary["healthy_mask_fp_rate"])
    )


def params_count(model: Any) -> int | None:
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        return None


def evaluate_checkpoint(
    candidate: dict[str, Any],
    checkpoint: Path,
    dataset_dir: Path,
    data_yaml: Path,
    output_root: Path,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    augment: bool = False,
    run_val: bool = False,
    plots: bool = False,
) -> dict[str, Any]:
    from ultralytics import YOLO

    validate_split_exists(dataset_dir)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    register_all_custom_attention_modules()
    model = YOLO(str(checkpoint))
    tag = f"{candidate['key']}_{'tta' if augment else 'plain'}_conf{conf:.2f}_iou{iou:.2f}".replace(".", "p")
    labeled_dir, labeled_yaml, labeled_copied = make_state_eval_dataset(
        dataset_dir, data_yaml, output_root, tag, "labeled_only", want_labeled=True
    )
    healthy_dir, healthy_yaml, healthy_copied = make_state_eval_dataset(
        dataset_dir, data_yaml, output_root, tag, "healthy_only", want_labeled=False
    )

    start = time.time()
    row: dict[str, Any] = {
        "module_key": candidate["key"],
        "display_name": candidate.get("display_name", candidate["key"]),
        "family": candidate.get("family"),
        "augmentation_policy": candidate.get("augmentation"),
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "tta_augment": augment,
        "params": params_count(model),
        "labeled_eval_valid_images": labeled_copied.get("valid"),
        "labeled_eval_test_images": labeled_copied.get("test"),
        "healthy_eval_valid_images": healthy_copied.get("valid"),
        "healthy_eval_test_images": healthy_copied.get("test"),
    }
    if run_val:
        full_val_metrics = model.val(
            data=str(data_yaml), split="val", imgsz=imgsz, conf=conf, iou=iou, augment=augment, plots=False, verbose=False
        )
        row.update(extract_val_metrics(full_val_metrics, "val"))

    full_test_metrics = model.val(
        data=str(data_yaml), split="test", imgsz=imgsz, conf=conf, iou=iou, augment=augment, plots=plots, verbose=False
    )
    labeled_test_metrics = model.val(
        data=str(labeled_yaml), split="test", imgsz=imgsz, conf=conf, iou=iou, augment=augment, plots=False, verbose=False
    )
    eval_time_sec = time.time() - start
    row.update(extract_val_metrics(full_test_metrics, "full_test"))
    row.update(extract_val_metrics(labeled_test_metrics, "labeled_test"))

    labeled_count = count_prediction_errors(
        model, labeled_dir / "test" / "images", labeled_dir / "test" / "labels", imgsz=imgsz, conf=conf, iou=iou, augment=augment
    )
    full_count = count_prediction_errors(
        model, dataset_dir / "test" / "images", dataset_dir / "test" / "labels", imgsz=imgsz, conf=conf, iou=iou, augment=augment
    )
    healthy_fp = healthy_false_positive_summary(
        model, healthy_dir / "test" / "images", imgsz=imgsz, conf=conf, iou=iou, augment=augment
    )
    for key, value in labeled_count.items():
        row[f"labeled_test_{key}"] = value
    for key, value in full_count.items():
        row[f"full_test_count_{key}"] = value
    for key, value in healthy_fp.items():
        row[f"healthy_test_{key}"] = value
    row["healthy_aware_score"] = healthy_aware_score(row["labeled_test_mask_map50"], labeled_count, healthy_fp)
    row["eval_time_sec"] = round(eval_time_sec, 4)
    return row


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


def existing_eval_keys(path: Path) -> set[tuple[str, float, float, bool]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, float, float, bool]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                keys.add(
                    (
                        row["module_key"],
                        round(float(row["conf"]), 4),
                        round(float(row["iou"]), 4),
                        str(row.get("tta_augment", "")).lower() in {"true", "1", "yes"},
                    )
                )
            except Exception:
                continue
    return keys


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
    print("Ultralytics Albumentations hook disabled for matched strong baseline.")


def sanity_check_model_yaml(model_yaml: str | Path, imgsz: int = DEFAULT_IMGSZ, device: str = "cpu") -> dict[str, Any]:
    """Instantiate a model YAML and run one dummy forward pass."""

    from ultralytics import YOLO

    register_all_custom_attention_modules()
    model_path = resolve_existing_path(model_yaml)
    if model_path is None:
        raise FileNotFoundError(f"Model YAML not found: {model_yaml}")
    yolo = YOLO(str(model_path))
    try:
        import torch

        net = yolo.model.to(device).eval()
        x = torch.zeros(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            output = net(x)
        shapes = _shape_summary(output)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Dummy forward failed for {model_path}: {exc}") from exc

    return {
        "model_yaml": str(model_path),
        "imgsz": imgsz,
        "device": device,
        "status": "ok",
        "output_shapes": shapes,
    }


def _shape_summary(value: Any) -> Any:
    if hasattr(value, "shape"):
        return tuple(int(v) for v in value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape_summary(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape_summary(item) for key, item in value.items()}
    return type(value).__name__


def sanity_check_candidate_yamls(module_keys: list[str] | None = None, imgsz: int = DEFAULT_IMGSZ, device: str = "cpu") -> list[dict[str, Any]]:
    rows = []
    for candidate in selected_candidates(module_keys):
        model_yaml = candidate.get("model_yaml")
        if not model_yaml or str(model_yaml).endswith(".pt"):
            continue
        row = {"module_key": candidate["key"], "display_name": candidate.get("display_name"), "model_yaml": str(model_yaml)}
        try:
            row.update(sanity_check_model_yaml(model_yaml, imgsz=imgsz, device=device))
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "error", "error_type": exc.__class__.__name__, "error": str(exc)})
        rows.append(row)
    return rows
