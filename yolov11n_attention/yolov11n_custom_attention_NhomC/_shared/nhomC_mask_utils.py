"""Mask-head and boundary-quality helpers for custom-attention Group C.

Group C is intentionally ablation focused:
- C1: increase Segment mask coefficient/prototype count from nm=32 to nm=64.
- C2: add a runtime boundary-aware mask objective.
- C3: replace nearest-neighbor neck/head upsampling with bilinear upsampling.

The helper reuses Group A for custom module registration and evaluation, and
Group B for training protocol defaults. It does not edit vendored Ultralytics
files or existing experiment notebooks.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml


DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 30

MASK_HEAD_MODULE_KEYS = [
    "triplet_attention_segment_head",
    "cote_gate",
    "cesa_lite_segment_head",
    "sge_eca_head_gate",
    "lpsc_gate",
]

BOUNDARY_LOSS_MODULE_KEYS = [
    "triplet_attention_segment_head",
    "cote_gate",
    "cesa_lite_segment_head",
]

BILINEAR_UPSAMPLE_MODULE_KEYS = [
    "baseline_clean",
    "triplet_attention_segment_head",
    "cote_gate",
    "cesa_lite_segment_head",
    "sge_eca_head_gate",
]


def attention_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [*here.parents, Path.cwd(), Path.cwd() / "yolov11n_attention"]:
        if (parent / "custom_attention").exists() and (parent / "yolov11n_custom_attention_NhomA").exists():
            return parent
    return here.parents[2]


def _load_module_from_file(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not path.exists():
        raise FileNotFoundError(f"Required helper not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_group_a_utils():
    root = attention_root()
    path = root / "yolov11n_custom_attention_NhomA" / "_shared" / "nhomA_eval_utils.py"
    return _load_module_from_file("nhomA_eval_utils_for_nhomC", path)


def load_group_b_utils():
    root = attention_root()
    path = root / "yolov11n_custom_attention_NhomB" / "_shared" / "nhomB_train_utils.py"
    return _load_module_from_file("nhomB_train_utils_for_nhomC", path)


def ensure_project_import_paths() -> Path:
    return load_group_a_utils().ensure_project_import_paths()


def register_all_custom_attention_modules() -> None:
    load_group_a_utils().register_all_custom_attention_modules()


def candidate_by_key(key: str) -> dict[str, Any]:
    return load_group_a_utils().candidate_by_key(key)


def selected_candidates(module_keys: list[str] | None = None) -> list[dict[str, Any]]:
    return load_group_a_utils().selected_candidates(module_keys)


def resolve_existing_path(path_value: str | Path | None) -> Path | None:
    return load_group_a_utils().resolve_existing_path(path_value)


def copy_dataset_snapshot(source_dataset: Path, snapshot_dir: Path, refresh: bool = False) -> Path:
    return load_group_a_utils().copy_dataset_snapshot(source_dataset, snapshot_dir, refresh=refresh)


def write_data_yaml(source_data_yaml: Path, dataset_dir: Path, yaml_path: Path) -> Path:
    return load_group_a_utils().write_data_yaml(source_data_yaml, dataset_dir, yaml_path)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    load_group_a_utils().append_csv_row(path, row)


def disable_ultralytics_albumentations() -> None:
    load_group_a_utils().disable_ultralytics_albumentations()


def evaluate_checkpoint(*args, **kwargs) -> dict[str, Any]:
    return load_group_a_utils().evaluate_checkpoint(*args, **kwargs)


def sanity_check_model_yaml(model_yaml: str | Path, imgsz: int = DEFAULT_IMGSZ, device: str = "cpu") -> dict[str, Any]:
    return load_group_a_utils().sanity_check_model_yaml(model_yaml, imgsz=imgsz, device=device)


def read_best_epoch_from_results(run_path: Path) -> dict[str, Any]:
    return load_group_b_utils().read_best_epoch_from_results(run_path)


def train_args_without_removed_keys(train_args: dict[str, Any]) -> dict[str, Any]:
    return load_group_b_utils().train_args_without_removed_keys(train_args)


def clean_light_train_args() -> dict[str, Any]:
    return copy.deepcopy(load_group_b_utils().CLEAN_LIGHT_AUG_TRAIN_ARGS)


def stable_train_control() -> dict[str, Any]:
    return copy.deepcopy(load_group_b_utils().STABLE_TRAIN_CONTROL)


def run_dir_for(runs_root: Path, experiment_key: str, module_key: str, seed: int) -> Path:
    return load_group_b_utils().run_dir_for(Path(runs_root), experiment_key, module_key, seed)


def _module_name(module_value: Any) -> str:
    if isinstance(module_value, str):
        return module_value
    return getattr(module_value, "__name__", str(module_value))


def _iter_model_layers(cfg: dict[str, Any]):
    for section in ("backbone", "head"):
        for index, layer in enumerate(cfg.get(section, [])):
            if isinstance(layer, list) and len(layer) >= 4:
                yield section, index, layer


def mutate_segment_nm(cfg: dict[str, Any], nm: int = 64) -> list[dict[str, Any]]:
    """Change only the Segment nm argument while preserving npr."""

    changes: list[dict[str, Any]] = []
    for section, index, layer in _iter_model_layers(cfg):
        if _module_name(layer[2]) != "Segment":
            continue
        args = layer[3]
        if not isinstance(args, list) or len(args) < 2:
            raise ValueError(f"Segment layer at {section}[{index}] does not have [nc, nm, ...] args: {layer}")
        old_nm = args[1]
        args[1] = int(nm)
        changes.append(
            {
                "section": section,
                "layer_index": index,
                "old_nm": old_nm,
                "new_nm": int(nm),
                "segment_args_after": list(args),
            }
        )
    if not changes:
        raise ValueError("No Segment layer found while applying nm mutation.")
    return changes


def mutate_upsample_mode(cfg: dict[str, Any], mode: str = "bilinear") -> list[dict[str, Any]]:
    """Change neck/head nn.Upsample mode, leaving size/scale_factor unchanged."""

    changes: list[dict[str, Any]] = []
    for section, index, layer in _iter_model_layers(cfg):
        if _module_name(layer[2]) != "nn.Upsample":
            continue
        args = layer[3]
        if not isinstance(args, list) or len(args) < 3:
            raise ValueError(f"nn.Upsample layer at {section}[{index}] does not have [size, scale, mode]: {layer}")
        old_mode = args[2]
        args[2] = str(mode)
        changes.append(
            {
                "section": section,
                "layer_index": index,
                "old_mode": old_mode,
                "new_mode": str(mode),
                "upsample_args_after": list(args),
            }
        )
    if not changes:
        raise ValueError("No nn.Upsample layer found while applying upsample mutation.")
    return changes


def source_yaml_for_candidate(candidate_key: str) -> Path:
    if candidate_key == "baseline_clean":
        baseline_yaml = (
            attention_root()
            / "yolov11n_custom_attention_NhomC"
            / "_shared"
            / "yolo11n_seg_resolved_baseline.yaml"
        )
        if baseline_yaml.exists():
            return baseline_yaml
        raise FileNotFoundError(f"Resolved baseline YAML not found: {baseline_yaml}")

    candidate = candidate_by_key(candidate_key)
    model_yaml = candidate.get("model_yaml")
    if not model_yaml or str(model_yaml).endswith(".pt"):
        raise ValueError(
            f"Candidate {candidate_key!r} does not expose a source YAML. "
            "Pass a custom YAML candidate or add a baseline YAML before generating mask-head variants."
        )
    resolved = resolve_existing_path(model_yaml)
    if resolved is None:
        raise FileNotFoundError(f"Model YAML not found for {candidate_key}: {model_yaml}")
    return resolved


def create_variant_yaml(
    candidate_key: str,
    variant_key: str,
    output_dir: Path,
    nm: int | None = None,
    upsample_mode: str | None = None,
) -> dict[str, Any]:
    """Create a Group C YAML variant for a candidate module."""

    source_yaml = source_yaml_for_candidate(candidate_key)
    with source_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML did not parse to a mapping: {source_yaml}")

    segment_changes: list[dict[str, Any]] = []
    upsample_changes: list[dict[str, Any]] = []
    if nm is not None:
        segment_changes = mutate_segment_nm(cfg, nm=int(nm))
    if upsample_mode is not None:
        upsample_changes = mutate_upsample_mode(cfg, mode=str(upsample_mode))

    out_dir = Path(output_dir) / "generated_yamls" / candidate_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{candidate_key}_{variant_key}.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    metadata = {
        "candidate_key": candidate_key,
        "variant_key": variant_key,
        "source_yaml": str(source_yaml),
        "variant_yaml": str(out_yaml),
        "segment_nm_changes": len(segment_changes),
        "upsample_mode_changes": len(upsample_changes),
        "segment_change_details": json.dumps(segment_changes, sort_keys=True),
        "upsample_change_details": json.dumps(upsample_changes, sort_keys=True),
    }
    return metadata


def create_variant_yamls(
    module_keys: list[str],
    variant_key: str,
    output_dir: Path,
    nm: int | None = None,
    upsample_mode: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for module_key in module_keys:
        rows.append(create_variant_yaml(module_key, variant_key, output_dir, nm=nm, upsample_mode=upsample_mode))
    return rows


def sanity_check_variant_yamls(rows: list[dict[str, Any]], imgsz: int = DEFAULT_IMGSZ, device: str = "cpu") -> list[dict[str, Any]]:
    checked = []
    for row in rows:
        item = dict(row)
        try:
            item.update(sanity_check_model_yaml(item["variant_yaml"], imgsz=imgsz, device=device))
        except Exception as exc:  # noqa: BLE001
            item.update({"status": "error", "error_type": exc.__class__.__name__, "error": str(exc)})
        checked.append(item)
    return checked


def _sobel_boundary_map(mask):
    import torch
    import torch.nn.functional as F

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.numel() == 0:
        return mask
    x = mask.float().unsqueeze(1)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-6).squeeze(1)
    max_per_mask = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (mag / max_per_mask).clamp(0.0, 1.0)


def enable_boundary_aware_mask_loss(boundary_weight: float = 0.10, edge_mode: str = "sobel") -> None:
    """Patch v8SegmentationLoss.single_mask_loss with a boundary objective.

    The patch is process-local. It keeps the original BCE mask loss and adds a
    cropped, area-normalized L1 penalty between Sobel boundary maps of predicted
    probabilities and ground-truth masks.
    """

    if edge_mode != "sobel":
        raise ValueError("Only edge_mode='sobel' is currently supported.")

    import torch
    import torch.nn.functional as F
    import ultralytics.utils.loss as loss_mod

    cls = loss_mod.v8SegmentationLoss
    if not hasattr(cls, "_nhomc_original_single_mask_loss"):
        cls._nhomc_original_single_mask_loss = cls.single_mask_loss

    weight = float(boundary_weight)

    def boundary_single_mask_loss(gt_mask, pred, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        base_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")

        pred_edge = _sobel_boundary_map(pred_mask.sigmoid())
        gt_edge = _sobel_boundary_map(gt_mask.float())
        boundary_loss = F.l1_loss(pred_edge, gt_edge, reduction="none")

        total_loss = base_loss + weight * boundary_loss
        safe_area = area.clamp_min(1e-6)
        return (loss_mod.crop_mask(total_loss, xyxy).mean(dim=(1, 2)) / safe_area).sum()

    cls.single_mask_loss = staticmethod(boundary_single_mask_loss)
    cls._nhomc_boundary_loss_patched = True
    cls._nhomc_boundary_loss_weight = weight
    cls._nhomc_boundary_loss_edge_mode = edge_mode


def disable_boundary_aware_mask_loss() -> None:
    import ultralytics.utils.loss as loss_mod

    cls = loss_mod.v8SegmentationLoss
    original = getattr(cls, "_nhomc_original_single_mask_loss", None)
    if original is not None:
        cls.single_mask_loss = staticmethod(original)
    cls._nhomc_boundary_loss_patched = False


def boundary_loss_patch_status() -> dict[str, Any]:
    import ultralytics.utils.loss as loss_mod

    cls = loss_mod.v8SegmentationLoss
    return {
        "patched": bool(getattr(cls, "_nhomc_boundary_loss_patched", False)),
        "boundary_weight": getattr(cls, "_nhomc_boundary_loss_weight", None),
        "edge_mode": getattr(cls, "_nhomc_boundary_loss_edge_mode", None),
    }


def run_mask_head_variant_experiment(
    candidate_key: str,
    experiment_key: str,
    variant_key: str,
    base_data_dir: Path,
    source_data_yaml: Path,
    output_root: Path,
    runs_root: Path,
    train_args: dict[str, Any],
    seed: int = DEFAULT_SEED,
    imgsz: int = DEFAULT_IMGSZ,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH,
    patience: int = DEFAULT_PATIENCE,
    smoke: bool = False,
    save_period: int = -1,
    pretrained_weights: str = "yolo11n-seg.pt",
    load_pretrained_weights: bool = True,
    nm: int | None = None,
    upsample_mode: str | None = None,
    boundary_loss_weight: float | None = None,
    evaluate_after: bool = True,
    eval_conf: float = 0.25,
    eval_iou: float = 0.70,
) -> dict[str, Any]:
    from ultralytics import YOLO

    ensure_project_import_paths()
    register_all_custom_attention_modules()
    if boundary_loss_weight is not None:
        enable_boundary_aware_mask_loss(boundary_loss_weight)

    candidate = candidate_by_key(candidate_key)
    exp_output = Path(output_root) / experiment_key
    yaml_row = create_variant_yaml(
        candidate_key=candidate_key,
        variant_key=variant_key,
        output_dir=exp_output,
        nm=nm,
        upsample_mode=upsample_mode,
    )
    model_spec = yaml_row["variant_yaml"]
    variant_module_key = f"{candidate_key}_{variant_key}"

    dataset_dir = exp_output / "datasets" / variant_module_key / f"seed_{seed}" / "dataset"
    dataset_dir = copy_dataset_snapshot(Path(base_data_dir), dataset_dir, refresh=False)
    data_yaml = write_data_yaml(Path(source_data_yaml), dataset_dir, dataset_dir / "data.yaml")

    run_dir = run_dir_for(Path(runs_root), experiment_key, variant_module_key, seed)
    best_path = run_dir / "weights" / "best.pt"
    train_csv = exp_output / "reports" / f"{experiment_key}_train_runs.csv"
    eval_csv = exp_output / "reports" / f"{experiment_key}_eval_results.csv"

    train_row: dict[str, Any] = {
        "experiment_key": experiment_key,
        "variant_key": variant_key,
        "module_key": candidate_key,
        "variant_module_key": variant_module_key,
        "display_name": candidate.get("display_name", candidate_key),
        "seed": seed,
        "model_spec": model_spec,
        "source_yaml": yaml_row["source_yaml"],
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "smoke": smoke,
        "nm": nm,
        "upsample_mode": upsample_mode,
        "boundary_loss_weight": boundary_loss_weight,
        **{f"yaml_{k}": v for k, v in yaml_row.items() if k not in {"candidate_key", "variant_key"}},
    }

    if best_path.exists():
        train_row.update({"status": "skipped_existing", "reason": "best checkpoint already exists"})
    else:
        disable_ultralytics_albumentations()
        yolo = YOLO(model_spec)
        if load_pretrained_weights:
            yolo = yolo.load(str(pretrained_weights))

        actual_imgsz = 320 if smoke else int(imgsz)
        actual_epochs = 1 if smoke else int(epochs)
        actual_batch = 8 if smoke else int(batch)
        actual_patience = 1 if smoke else int(patience)
        clean_train_args = train_args_without_removed_keys(train_args)

        start = time.time()
        yolo.train(
            data=str(data_yaml),
            task="segment",
            imgsz=actual_imgsz,
            epochs=actual_epochs,
            batch=actual_batch,
            patience=actual_patience,
            seed=seed,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=False,
            pretrained=True,
            plots=not smoke,
            verbose=True,
            save_period=save_period,
            **clean_train_args,
        )
        train_row.update(
            {
                "status": "trained",
                "train_time_min": round((time.time() - start) / 60.0, 4),
                "imgsz": actual_imgsz,
                "epochs": actual_epochs,
                "batch": actual_batch,
                "patience": actual_patience,
                "save_period": save_period,
                "train_args_json": json.dumps(clean_train_args, sort_keys=True),
            }
        )

    train_row.update(read_best_epoch_from_results(run_dir))
    append_csv_row(train_csv, train_row)

    if evaluate_after and best_path.exists():
        eval_row = evaluate_checkpoint(
            candidate={**candidate, "key": variant_module_key, "augmentation": experiment_key},
            checkpoint=best_path,
            dataset_dir=dataset_dir,
            data_yaml=data_yaml,
            output_root=exp_output,
            imgsz=320 if smoke else int(imgsz),
            conf=eval_conf,
            iou=eval_iou,
            augment=False,
            run_val=True,
            plots=False,
        )
        eval_row.update(
            {
                "experiment_key": experiment_key,
                "variant_key": variant_key,
                "module_key": candidate_key,
                "variant_module_key": variant_module_key,
                "seed": seed,
                "protocol_train_status": train_row.get("status"),
            }
        )
        append_csv_row(eval_csv, eval_row)
        train_row["eval_csv"] = str(eval_csv)

    train_row["train_csv"] = str(train_csv)
    return train_row
