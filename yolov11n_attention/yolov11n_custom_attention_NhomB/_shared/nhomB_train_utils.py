"""Training-protocol helpers for custom-attention Group B.

Group B changes training protocol rather than architecture:
- B1: warmup + cosine LR + shorter patience.
- B2: runtime label smoothing for classification targets.
- B3: matched underwater/strong augmentation.
- B4: checkpoint averaging / SWA after training.

The evaluation path delegates to Group A utilities so metric definitions stay
identical across calibration and retraining experiments.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import re
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

TOP_PROTOCOL_MODULE_KEYS = [
    "baseline_clean",
    "triplet_attention_segment_head",
    "cote_gate",
    "lpsc_gate",
    "cesa_lite_segment_head",
    "sge_eca_head_gate",
]

CLEAN_LIGHT_AUG_TRAIN_ARGS: dict[str, Any] = {
    "auto_augment": None,
    "erasing": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.35,
    "hsv_v": 0.20,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.20,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": 0.0,
    "bgr": 0.0,
}

STABLE_TRAIN_CONTROL: dict[str, Any] = {
    "epochs": 150,
    "patience": 20,
    "cos_lr": True,
    "warmup_epochs": 8,
    "lr0": 0.01,
    "lrf": 0.001,
}

UNDERWATER_AUG_ARGS: dict[str, Any] = {
    "auto_augment": None,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "fliplr": 0.5,
    "flipud": 0.3,
    "hsv_h": 0.05,
    "hsv_s": 0.50,
    "hsv_v": 0.40,
    "scale": 0.50,
    "translate": 0.10,
    "degrees": 10.0,
    "erasing": 0.10,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": 0.0,
    "bgr": 0.0,
}


def attention_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [*here.parents, Path.cwd(), Path.cwd() / "yolov11n_attention"]:
        if (parent / "custom_attention").exists() and (parent / "yolov11n_custom_attention_NhomA").exists():
            return parent
    return here.parents[2]


def load_group_a_utils():
    """Load Group A utilities under a private module name to avoid _shared name clashes."""

    root = attention_root()
    path = root / "yolov11n_custom_attention_NhomA" / "_shared" / "nhomA_eval_utils.py"
    if not path.exists():
        raise FileNotFoundError(f"Group A helper not found: {path}")
    module_name = "nhomA_eval_utils_for_nhomB"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Group A helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_project_import_paths() -> Path:
    a = load_group_a_utils()
    return a.ensure_project_import_paths()


def register_all_custom_attention_modules() -> None:
    a = load_group_a_utils()
    a.register_all_custom_attention_modules()


def selected_candidates(module_keys: list[str] | None = None) -> list[dict[str, Any]]:
    a = load_group_a_utils()
    return a.selected_candidates(module_keys or TOP_PROTOCOL_MODULE_KEYS)


def candidate_by_key(key: str) -> dict[str, Any]:
    a = load_group_a_utils()
    return a.candidate_by_key(key)


def resolve_existing_path(path_value: str | Path | None) -> Path | None:
    a = load_group_a_utils()
    return a.resolve_existing_path(path_value)


def validate_split_exists(dataset_dir: Path) -> None:
    a = load_group_a_utils()
    a.validate_split_exists(dataset_dir)


def write_data_yaml(source_data_yaml: Path, dataset_dir: Path, yaml_path: Path) -> Path:
    a = load_group_a_utils()
    return a.write_data_yaml(source_data_yaml, dataset_dir, yaml_path)


def copy_dataset_snapshot(source_dataset: Path, snapshot_dir: Path, refresh: bool = False) -> Path:
    a = load_group_a_utils()
    return a.copy_dataset_snapshot(source_dataset, snapshot_dir, refresh=refresh)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    a = load_group_a_utils()
    a.append_csv_row(path, row)


def disable_ultralytics_albumentations() -> None:
    a = load_group_a_utils()
    a.disable_ultralytics_albumentations()


def evaluate_checkpoint(*args, **kwargs) -> dict[str, Any]:
    a = load_group_a_utils()
    return a.evaluate_checkpoint(*args, **kwargs)


def sanity_check_candidate_yamls(module_keys: list[str] | None = None, imgsz: int = DEFAULT_IMGSZ, device: str = "cpu") -> list[dict[str, Any]]:
    a = load_group_a_utils()
    return a.sanity_check_candidate_yamls(module_keys or TOP_PROTOCOL_MODULE_KEYS, imgsz=imgsz, device=device)


def read_best_epoch_from_results(run_path: Path) -> dict[str, Any]:
    results_csv = Path(run_path) / "results.csv"
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


def train_args_without_removed_keys(train_args: dict[str, Any]) -> dict[str, Any]:
    """Avoid passing removed Ultralytics keys that silently become no-ops."""

    removed = {"label_smoothing", "save_hybrid", "crop_fraction"}
    return {k: v for k, v in dict(train_args).items() if k not in removed}


def enable_runtime_label_smoothing(eps: float = 0.05) -> None:
    """Patch v8DetectionLoss target scores for true runtime label smoothing.

    Current vendored Ultralytics removes the train arg `label_smoothing`, so this
    patch applies smoothing directly to BCE classification targets. It is
    process-local and does not edit Ultralytics files on disk.
    """

    import torch
    import ultralytics.utils.loss as loss_mod
    from ultralytics.utils.loss import make_anchors, xywh2xyxy

    cls = loss_mod.v8DetectionLoss
    if getattr(cls, "_nhomb_label_smoothing_patched", False):
        cls._nhomb_label_smoothing_eps = float(eps)
        return

    original = cls.get_assigned_targets_and_loss

    def patched_get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        eps_value = float(getattr(cls, "_nhomb_label_smoothing_eps", eps))
        if eps_value > 0:
            smooth_neg = eps_value / max(1, self.nc)
            cls_targets = target_scores.to(dtype) * (1.0 - eps_value) + smooth_neg
        else:
            cls_targets = target_scores.to(dtype)

        bce_loss = self.bce(pred_scores, cls_targets)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()

    patched_get_assigned_targets_and_loss.__name__ = original.__name__
    patched_get_assigned_targets_and_loss.__doc__ = original.__doc__
    cls.get_assigned_targets_and_loss = patched_get_assigned_targets_and_loss
    cls._nhomb_label_smoothing_patched = True
    cls._nhomb_label_smoothing_eps = float(eps)


def candidate_model_spec(candidate: dict[str, Any]) -> str:
    value = candidate.get("model_yaml")
    if not value or str(value).endswith(".pt"):
        return str(value or "yolo11n-seg.pt")
    resolved = resolve_existing_path(value)
    if resolved is None:
        raise FileNotFoundError(f"Model YAML not found for {candidate['key']}: {value}")
    return str(resolved)


def run_dir_for(runs_root: Path, experiment_key: str, module_key: str, seed: int) -> Path:
    return Path(runs_root) / experiment_key / module_key / f"seed_{seed}"


def run_training_experiment(
    candidate_key: str,
    experiment_key: str,
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
    label_smoothing_eps: float | None = None,
    evaluate_after: bool = True,
    eval_conf: float = 0.25,
    eval_iou: float = 0.70,
) -> dict[str, Any]:
    from ultralytics import YOLO

    ensure_project_import_paths()
    register_all_custom_attention_modules()
    if label_smoothing_eps is not None:
        enable_runtime_label_smoothing(label_smoothing_eps)

    candidate = candidate_by_key(candidate_key)
    model_spec = candidate_model_spec(candidate)
    exp_output = Path(output_root) / experiment_key
    dataset_dir = exp_output / "datasets" / candidate_key / f"seed_{seed}" / "dataset"
    dataset_dir = copy_dataset_snapshot(Path(base_data_dir), dataset_dir, refresh=False)
    data_yaml = write_data_yaml(Path(source_data_yaml), dataset_dir, dataset_dir / "data.yaml")

    run_dir = run_dir_for(Path(runs_root), experiment_key, candidate_key, seed)
    best_path = run_dir / "weights" / "best.pt"
    train_csv = exp_output / "reports" / f"{experiment_key}_train_runs.csv"
    eval_csv = exp_output / "reports" / f"{experiment_key}_eval_results.csv"

    train_row: dict[str, Any] = {
        "experiment_key": experiment_key,
        "module_key": candidate_key,
        "display_name": candidate.get("display_name", candidate_key),
        "seed": seed,
        "model_spec": model_spec,
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "smoke": smoke,
        "label_smoothing_eps": label_smoothing_eps,
    }

    if best_path.exists():
        train_row.update({"status": "skipped_existing", "reason": "best checkpoint already exists"})
    else:
        disable_ultralytics_albumentations()
        yolo = YOLO(model_spec)
        if load_pretrained_weights and not str(model_spec).endswith(".pt"):
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
            candidate={**candidate, "augmentation": experiment_key},
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
        eval_row.update({"experiment_key": experiment_key, "seed": seed, "protocol_train_status": train_row.get("status")})
        append_csv_row(eval_csv, eval_row)
        train_row["eval_csv"] = str(eval_csv)

    train_row["train_csv"] = str(train_csv)
    return train_row


def checkpoint_epoch_number(path: Path) -> int:
    match = re.search(r"epoch(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def swa_source_checkpoints(run_dir: Path, max_checkpoints: int = 5, include_last: bool = True) -> list[Path]:
    weights_dir = Path(run_dir) / "weights"
    epoch_paths = sorted(weights_dir.glob("epoch*.pt"), key=checkpoint_epoch_number)
    selected = epoch_paths[-max_checkpoints:]
    last = weights_dir / "last.pt"
    if include_last and last.exists() and last not in selected:
        selected.append(last)
    return [path for path in selected if path.exists()]


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def average_yolo_checkpoints(checkpoints: list[Path], output_path: Path) -> Path:
    """Average floating parameters from multiple YOLO checkpoints."""

    import torch

    if len(checkpoints) < 2:
        raise ValueError("SWA needs at least two checkpoints to average.")
    register_all_custom_attention_modules()

    ckpts = [_torch_load(Path(path)) for path in checkpoints]
    model_key = "ema" if ckpts[0].get("ema") is not None else "model"
    models = [ckpt.get(model_key) or ckpt.get("model") for ckpt in ckpts]
    if any(model is None for model in models):
        raise ValueError("At least one checkpoint does not contain a YOLO model.")

    states = [model.float().state_dict() for model in models]
    first_state = states[0]
    avg_state = {}
    for key, tensor in first_state.items():
        if not torch.is_tensor(tensor):
            avg_state[key] = tensor
            continue
        if tensor.dtype.is_floating_point and all(key in state and torch.is_tensor(state[key]) for state in states[1:]):
            stacked = torch.stack([state[key].float() for state in states], dim=0)
            avg_state[key] = stacked.mean(dim=0).to(dtype=tensor.dtype)
        else:
            avg_state[key] = tensor

    averaged_model = copy.deepcopy(models[0]).float()
    averaged_model.load_state_dict(avg_state, strict=False)
    out_ckpt = dict(ckpts[0])
    out_ckpt["model"] = averaged_model
    out_ckpt["ema"] = averaged_model
    out_ckpt["train_args"] = dict(out_ckpt.get("train_args", {}), nhomb_swa_sources=[str(p) for p in checkpoints])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, output_path)
    return output_path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

