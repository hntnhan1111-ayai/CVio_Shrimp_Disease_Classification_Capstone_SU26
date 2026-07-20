"""Controlled architecture-attention helpers for custom-attention Group D.

Group D runs architecture ablations after A/B/C:
- D1: SimAM at backbone P3/P4.
- D2: CoordAtt at backbone P3/P4.
- D3: lightweight LKA head refinement before Segment.

The helper reuses Group A evaluation and Group B training defaults, but keeps
all new architecture registration process-local. It does not edit vendored
Ultralytics files or existing notebooks.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 30

BACKBONE_ATTENTION_TARGETS = [4, 6]

SIMAM_BACKBONE_MODULE_KEYS = [
    "triplet_attention_segment_head",
    "cote_gate",
    "sge_eca_head_gate",
    "lpsc_soft_gate",
]

CA_BACKBONE_MODULE_KEYS = [
    "triplet_attention_segment_head",
    "cote_gate",
]

LKA_HEAD_MODULE_KEYS = [
    "triplet_attention_segment_head",
    "cote_gate",
]

EXTRA_CANDIDATES: dict[str, dict[str, Any]] = {
    "lpsc_soft_gate": {
        "key": "lpsc_soft_gate",
        "display_name": "LPSC Soft Gate",
        "family": "custom_attention_research_modules",
        "augmentation": "clean_light",
        "model_yaml": "custom_attention_research_modules/lpsc_soft_gate/generated_yamls/lpsc_soft_strong_augmentation.yaml",
        "checkpoint_patterns": [
            "/kaggle/working/runs/custom_attention_research_modules/lpsc_soft_strong_augmentation/weights/best.pt",
        ],
    },
}

SHORT_KEY_NAMES = {
    "triplet_attention_segment_head": "triplet",
    "cote_gate": "cote",
    "sge_eca_head_gate": "sge_eca",
    "lpsc_soft_gate": "lpsc_soft",
    "simam_backbone_p3p4": "simam_p3p4",
    "ca_backbone_dual": "ca_p3p4",
    "lka_head_refine": "lka_head",
}


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
    return _load_module_from_file("nhomA_eval_utils_for_nhomD", path)


def load_group_b_utils():
    root = attention_root()
    path = root / "yolov11n_custom_attention_NhomB" / "_shared" / "nhomB_train_utils.py"
    return _load_module_from_file("nhomB_train_utils_for_nhomD", path)


def ensure_project_import_paths() -> Path:
    return load_group_a_utils().ensure_project_import_paths()


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
    register_group_d_modules()
    return load_group_a_utils().evaluate_checkpoint(*args, **kwargs)


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


def candidate_by_key(key: str) -> dict[str, Any]:
    if key in EXTRA_CANDIDATES:
        return dict(EXTRA_CANDIDATES[key])
    return load_group_a_utils().candidate_by_key(key)


def selected_candidates(module_keys: list[str]) -> list[dict[str, Any]]:
    return [candidate_by_key(key) for key in module_keys]


def _resolve_channels(c1: int | None = None, channels: int | None = None) -> int:
    value = channels if channels is not None else c1
    if value is None:
        raise ValueError("A channel count must be provided.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"channels must be positive, got {value}")
    return value


def _safe_groups(channels: int, preferred_groups: int) -> int:
    channels = int(channels)
    preferred_groups = max(1, int(preferred_groups))
    if channels % preferred_groups == 0:
        return preferred_groups
    return max(1, math.gcd(channels, preferred_groups))


def _odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    return kernel_size if kernel_size % 2 else kernel_size + 1


class NAMGate(nn.Module):
    """Normalization-aware gate returning [B, C, H, W]."""

    def __init__(self, c1: int | None = None, groups: int = 16, eps: float = 1e-6, channels: int | None = None):
        super().__init__()
        channels = _resolve_channels(c1, channels)
        self.norm = nn.GroupNorm(_safe_groups(channels, groups), channels, affine=True)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        weight = self.norm.weight.abs()
        weight = weight / (weight.sum() + self.eps)
        return torch.sigmoid(y * weight.view(1, -1, 1, 1))


class SimAMGate(nn.Module):
    """Parameter-free SimAM gate returning [B, C, H, W]."""

    def __init__(self, c1: int | None = None, eps: float = 1e-4, channels: int | None = None):
        super().__init__()
        _resolve_channels(c1, channels)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        if n <= 0:
            return torch.ones_like(x)
        centered = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        denom = 4.0 * (centered.sum(dim=(2, 3), keepdim=True) / n + self.eps)
        return torch.sigmoid(centered / denom + 0.5)


class LPSCSoftGate(nn.Module):
    """Soft LPSC variant used by the existing lpsc_soft YAML."""

    def __init__(
        self,
        c1: int,
        simam_eps: float = 1e-4,
        nam_weight_init: float = 0.25,
        eta_init: float = 1.25,
        delta_init: float = 0.25,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.nam = NAMGate(channels)
        self.simam = SimAMGate(channels, eps=simam_eps)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.nam_weight = nn.Parameter(torch.tensor(float(nam_weight_init)))
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        nam = self.nam(x)
        simam = self.simam(x)
        prior = self.dw(x)
        prior = torch.sigmoid(prior - self.smooth(prior))
        gate = torch.sigmoid(self.nam_weight * nam + self.eta * simam + self.delta * prior)
        return identity + self.gamma * identity * gate


class LKAResidualAttention(nn.Module):
    """Large-kernel residual attention that preserves [B, C, H, W]."""

    def __init__(
        self,
        c1: int,
        kernel_size: int = 5,
        dilated_kernel_size: int = 7,
        dilation: int = 3,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        kernel_size = _odd_kernel(kernel_size)
        dilated_kernel_size = _odd_kernel(dilated_kernel_size)
        dilation = max(1, int(dilation))
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=False,
        )
        self.dw_dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=dilated_kernel_size,
            padding=dilation * ((dilated_kernel_size - 1) // 2),
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.pw(self.dw_dilated(self.dw(x))))
        return x + self.gamma * x * (gate - 0.5)


def register_group_d_modules() -> None:
    """Register D-only modules after Group A registration has patched parse_model."""

    ensure_project_import_paths()
    load_group_a_utils().register_all_custom_attention_modules()
    import ultralytics.nn.tasks as tasks

    extra_modules = (LPSCSoftGate, LKAResidualAttention)
    for module_cls in extra_modules:
        setattr(tasks, module_cls.__name__, module_cls)
        setattr(sys.modules["__main__"], module_cls.__name__, module_cls)

    existing = tuple(getattr(tasks, "NHOMA_SINGLE_INPUT_ATTENTION_MODULES", ()))
    merged = list(existing)
    for module_cls in extra_modules:
        if module_cls not in merged:
            merged.append(module_cls)
    tasks.NHOMA_SINGLE_INPUT_ATTENTION_MODULES = tuple(merged)

    try:
        torch.serialization.add_safe_globals(list(extra_modules))
    except Exception:
        pass


def _module_name(module_value: Any) -> str:
    if isinstance(module_value, str):
        return module_value
    return getattr(module_value, "__name__", str(module_value))


def short_key(value: str) -> str:
    text = str(value)
    return SHORT_KEY_NAMES.get(text, text.replace("attention_segment_head", "att").replace("_backbone", "_bb"))


def source_yaml_for_candidate(candidate_key: str) -> Path:
    candidate = candidate_by_key(candidate_key)
    model_yaml = candidate.get("model_yaml")
    if not model_yaml or str(model_yaml).endswith(".pt"):
        raise ValueError(f"Candidate {candidate_key!r} does not expose a source YAML.")
    resolved = resolve_existing_path(model_yaml)
    if resolved is None:
        raise FileNotFoundError(f"Model YAML not found for {candidate_key}: {model_yaml}")
    return resolved


def _remap_from_value(value: Any, original_to_new: dict[int, int], target_to_inserted: dict[int, int]) -> Any:
    if isinstance(value, int):
        if value < 0:
            return value
        if value in target_to_inserted:
            return target_to_inserted[value]
        return original_to_new[value]
    if isinstance(value, list):
        return [_remap_from_value(item, original_to_new, target_to_inserted) for item in value]
    return value


def insert_attention_after_backbone_indices(
    cfg: dict[str, Any],
    target_indices: list[int],
    module_name: str,
    module_args: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Insert a single-input module after original backbone layers and remap positive refs."""

    module_args = list(module_args or [])
    backbone = copy.deepcopy(cfg.get("backbone", []))
    head = copy.deepcopy(cfg.get("head", []))
    backbone_count = len(backbone)
    layers = backbone + head
    targets = set(int(i) for i in target_indices)
    invalid = [idx for idx in targets if idx < 0 or idx >= backbone_count]
    if invalid:
        raise ValueError(f"Backbone insertion targets out of range: {invalid}; backbone_count={backbone_count}")

    new_layers: list[list[Any]] = []
    inserted_indices: set[int] = set()
    original_to_new: dict[int, int] = {}
    target_to_inserted: dict[int, int] = {}
    changes: list[dict[str, Any]] = []

    for original_index, layer in enumerate(layers):
        original_to_new[original_index] = len(new_layers)
        new_layers.append(copy.deepcopy(layer))
        if original_index in targets:
            inserted_index = len(new_layers)
            new_layers.append([-1, 1, module_name, list(module_args)])
            inserted_indices.add(inserted_index)
            target_to_inserted[original_index] = inserted_index
            changes.append(
                {
                    "target_original_index": original_index,
                    "target_new_index": original_to_new[original_index],
                    "inserted_index": inserted_index,
                    "module": module_name,
                    "args": list(module_args),
                }
            )

    for new_index, layer in enumerate(new_layers):
        if new_index in inserted_indices:
            continue
        layer[0] = _remap_from_value(layer[0], original_to_new, target_to_inserted)

    new_backbone_count = backbone_count + len(targets)
    cfg["backbone"] = new_layers[:new_backbone_count]
    cfg["head"] = new_layers[new_backbone_count:]
    return changes


def insert_lka_on_segment_inputs(
    cfg: dict[str, Any],
    module_args: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Insert LKAResidualAttention on each Segment input and rewire Segment."""

    module_args = list(module_args or [5, 7, 3, 0.0])
    backbone = copy.deepcopy(cfg.get("backbone", []))
    head = copy.deepcopy(cfg.get("head", []))
    layers = backbone + head
    backbone_count = len(backbone)
    segment_index = None
    for index, layer in enumerate(layers):
        if isinstance(layer, list) and len(layer) >= 4 and _module_name(layer[2]) == "Segment":
            segment_index = index
    if segment_index is None:
        raise ValueError("No Segment layer found for LKA head insertion.")
    if segment_index != len(layers) - 1:
        raise ValueError("LKA insertion currently expects Segment to be the final model layer.")

    prefix = copy.deepcopy(layers[:segment_index])
    segment = copy.deepcopy(layers[segment_index])
    original_from = segment[0]
    from_list = original_from if isinstance(original_from, list) else [original_from]
    new_from: list[int] = []
    changes: list[dict[str, Any]] = []

    for source_index in from_list:
        if not isinstance(source_index, int) or source_index < 0:
            raise ValueError(f"Segment input must be a non-negative layer index, got {source_index!r}")
        inserted_index = len(prefix)
        prefix.append([source_index, 1, "LKAResidualAttention", list(module_args)])
        new_from.append(inserted_index)
        changes.append(
            {
                "segment_original_input": source_index,
                "inserted_index": inserted_index,
                "module": "LKAResidualAttention",
                "args": list(module_args),
            }
        )

    segment[0] = new_from if isinstance(original_from, list) else new_from[0]
    new_layers = prefix + [segment]
    cfg["backbone"] = new_layers[:backbone_count]
    cfg["head"] = new_layers[backbone_count:]
    return changes


def create_variant_yaml(
    candidate_key: str,
    variant_key: str,
    output_dir: Path,
    variant_kind: str,
    backbone_targets: list[int] | None = None,
    lka_args: list[Any] | None = None,
) -> dict[str, Any]:
    source_yaml = source_yaml_for_candidate(candidate_key)
    with source_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML did not parse to a mapping: {source_yaml}")

    if variant_kind == "simam_backbone_p3p4":
        changes = insert_attention_after_backbone_indices(
            cfg,
            backbone_targets or BACKBONE_ATTENTION_TARGETS,
            "SimAM",
            [],
        )
    elif variant_kind == "ca_backbone_dual":
        changes = insert_attention_after_backbone_indices(
            cfg,
            backbone_targets or BACKBONE_ATTENTION_TARGETS,
            "CoordAtt",
            [],
        )
    elif variant_kind == "lka_head_refine":
        changes = insert_lka_on_segment_inputs(cfg, module_args=lka_args or [5, 7, 3, 0.0])
    else:
        raise ValueError(f"Unknown Group D variant_kind: {variant_kind}")

    out_dir = Path(output_dir) / "generated_yamls" / short_key(candidate_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{short_key(candidate_key)}_{short_key(variant_key)}.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return {
        "candidate_key": candidate_key,
        "variant_key": variant_key,
        "variant_kind": variant_kind,
        "source_yaml": str(source_yaml),
        "variant_yaml": str(out_yaml),
        "change_count": len(changes),
        "change_details": json.dumps(changes, sort_keys=True),
    }


def create_variant_yamls(
    module_keys: list[str],
    variant_key: str,
    output_dir: Path,
    variant_kind: str,
    backbone_targets: list[int] | None = None,
    lka_args: list[Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        create_variant_yaml(
            candidate_key=module_key,
            variant_key=variant_key,
            output_dir=output_dir,
            variant_kind=variant_kind,
            backbone_targets=backbone_targets,
            lka_args=lka_args,
        )
        for module_key in module_keys
    ]


def _shape_summary(value: Any) -> Any:
    if hasattr(value, "shape"):
        return tuple(int(v) for v in value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape_summary(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape_summary(item) for key, item in value.items()}
    return type(value).__name__


def sanity_check_model_yaml(model_yaml: str | Path, imgsz: int = DEFAULT_IMGSZ, device: str = "cpu") -> dict[str, Any]:
    from ultralytics import YOLO

    register_group_d_modules()
    model_path = resolve_existing_path(model_yaml) or Path(model_yaml)
    if not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_yaml}")
    yolo = YOLO(str(model_path))
    try:
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


def run_arch_variant_experiment(
    candidate_key: str,
    experiment_key: str,
    variant_key: str,
    variant_kind: str,
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
    evaluate_after: bool = True,
    eval_conf: float = 0.25,
    eval_iou: float = 0.70,
) -> dict[str, Any]:
    from ultralytics import YOLO

    register_group_d_modules()
    candidate = candidate_by_key(candidate_key)
    exp_output = Path(output_root) / experiment_key
    yaml_row = create_variant_yaml(
        candidate_key=candidate_key,
        variant_key=variant_key,
        output_dir=exp_output,
        variant_kind=variant_kind,
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
        "variant_kind": variant_kind,
        "module_key": candidate_key,
        "variant_module_key": variant_module_key,
        "display_name": candidate.get("display_name", candidate_key),
        "seed": seed,
        "model_spec": model_spec,
        "source_yaml": yaml_row["source_yaml"],
        "change_count": yaml_row["change_count"],
        "change_details": yaml_row["change_details"],
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "smoke": smoke,
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
                "variant_kind": variant_kind,
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
