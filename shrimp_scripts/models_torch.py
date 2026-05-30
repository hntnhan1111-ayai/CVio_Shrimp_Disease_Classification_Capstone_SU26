"""PyTorch, torchvision, and timm model training for shrimp classification."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from . import config
from .evaluate import compute_metrics, confusion_count_frame, prediction_frame, save_prediction_artifacts
from .losses import LOSS_CONFIG, make_loss
from .progress import log_event, progress_iter
from .utils import (
    cleanup_memory,
    ensure_dir,
    is_oom_error,
    is_run_completed,
    save_csv,
    set_seed,
    sha256_file,
    stable_hash,
    utc_now,
    write_json,
    write_status,
)


def _torch_stack():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from torchvision.transforms import InterpolationMode

    return torch, nn, F, optim, DataLoader, Dataset, models, transforms, InterpolationMode


def torch_run_id(model_key: str, condition: dict[str, Any]) -> str:
    backend = "torchvision" if model_key in {config.CONVNEXT_CORE_MODEL_KEY, "mobilenet_v3_large", "shufflenet_v2_x1_0", "squeezenet1_1"} else "timm"
    return f"{backend}_{model_key}_{condition['condition_key']}_seed{config.SEED}_repeat{config.REPEAT}"


def list_core_torch_runs(smoke_test: bool = False) -> list[dict[str, Any]]:
    conditions = config.CORE_CONDITIONS[:1] if smoke_test else config.CORE_CONDITIONS
    return [{
        "run_id": torch_run_id(config.CONVNEXT_CORE_MODEL_KEY, condition),
        "backend": "torchvision",
        "model_key": config.CONVNEXT_CORE_MODEL_KEY,
        "model": config.CONVNEXT_CORE_MODEL_NAME,
        **condition,
    } for condition in conditions]


def list_lightweight_runs(smoke_test: bool = False, start: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    models = config.LIGHTWEIGHT_MODELS[:1] if smoke_test else config.LIGHTWEIGHT_MODELS
    rows: list[dict[str, Any]] = []
    for model_key in models:
        for condition in config.FAMILY_CONDITIONS:
            rows.append({
                "run_id": torch_run_id(model_key, condition),
                "backend": "torchvision" if model_key in {"mobilenet_v3_large", "shufflenet_v2_x1_0", "squeezenet1_1"} else "timm",
                "model_key": model_key,
                "model": model_key,
                **condition,
            })
    if start is not None:
        rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]
    return rows


class ManifestDataset:
    def __new__(cls, frame: pd.DataFrame, transform):
        _torch, _nn, _F, _optim, _DataLoader, Dataset, _models, _transforms, _InterpolationMode = _torch_stack()

        class _ManifestDataset(Dataset):
            def __init__(self) -> None:
                self.frame = frame.reset_index(drop=True).copy()
                self.transform = transform

            def __len__(self) -> int:
                return len(self.frame)

            def __getitem__(self, index: int):
                row = self.frame.iloc[index]
                image = Image.open(row["processed_path"]).convert("RGB")
                return self.transform(image), int(row["label"]), int(index)

        return _ManifestDataset()


def train_transform(randaugment: bool):
    _torch, _nn, _F, _optim, _DataLoader, _Dataset, _models, transforms, InterpolationMode = _torch_stack()
    steps = [
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.82, 1.0), ratio=(0.90, 1.10), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
    ]
    if randaugment:
        steps.append(transforms.RandAugment(num_ops=config.RAND_AUGMENT_NUM_OPS, magnitude=config.RAND_AUGMENT_MAGNITUDE))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)


def eval_transform():
    _torch, _nn, _F, _optim, _DataLoader, _Dataset, models, _transforms, _InterpolationMode = _torch_stack()
    return models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()


def transform_audit(randaugment: bool) -> dict[str, Any]:
    return {
        "resize": 256,
        "random_resized_crop": {"size": config.IMG_SIZE, "scale": [0.82, 1.0], "ratio": [0.90, 1.10]},
        "horizontal_flip_p": 0.5,
        "rotation_degrees": 10,
        "color_jitter": {"brightness": 0.10, "contrast": 0.10, "saturation": 0.05},
        "randaugment": bool(randaugment),
        "randaugment_num_ops": config.RAND_AUGMENT_NUM_OPS if randaugment else None,
        "randaugment_magnitude": config.RAND_AUGMENT_MAGNITUDE if randaugment else None,
        "normalization": "ImageNet mean/std",
    }


class ShrimpXNetConvNeXt:
    def __new__(cls):
        _torch, nn, _F, _optim, _DataLoader, _Dataset, models, _transforms, _InterpolationMode = _torch_stack()

        class _ShrimpXNetConvNeXt(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                backbone = models.convnext_tiny(weights=weights)
                for param in backbone.parameters():
                    param.requires_grad = False
                self.features = backbone.features
                self.avgpool = backbone.avgpool
                self.norm = backbone.classifier[0]
                num_features = backbone.classifier[2].in_features
                self.classifier = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(512, config.NUM_CLASSES),
                )

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = self.norm(x)
                x = torch.flatten(x, 1)
                return self.classifier(x)

            def unfreeze_finetune_layers(self, from_feature_index: int = config.CONVNEXT_UNFREEZE_FROM_FEATURE_INDEX) -> None:
                for module in self.features[from_feature_index:]:
                    for param in module.parameters():
                        param.requires_grad = True
                for param in self.norm.parameters():
                    param.requires_grad = True

        torch = _torch
        return _ShrimpXNetConvNeXt()


def create_torchvision_fallback(model_key: str):
    _torch, nn, _F, _optim, _DataLoader, _Dataset, models, _transforms, _InterpolationMode = _torch_stack()
    if model_key == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, config.NUM_CLASSES)
        return model, {"library": "torchvision", "pretrained_source": "MobileNet_V3_Large_Weights.IMAGENET1K_V2"}
    if model_key == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        return model, {"library": "torchvision", "pretrained_source": "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1"}
    if model_key == "squeezenet1_1":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Conv2d(512, config.NUM_CLASSES, kernel_size=(1, 1))
        model.num_classes = config.NUM_CLASSES
        return model, {"library": "torchvision", "pretrained_source": "SqueezeNet1_1_Weights.IMAGENET1K_V1"}
    return None, None


def create_torch_model(model_key: str):
    if model_key == config.CONVNEXT_CORE_MODEL_KEY:
        return ShrimpXNetConvNeXt(), {
            "library": "torchvision",
            "pretrained_source": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "model_key": model_key,
            "training_recipe": "ShrimpXNet-style frozen warmup then features[5:] fine-tune",
        }
    fallback_model, fallback_config = create_torchvision_fallback(model_key)
    if fallback_model is not None:
        fallback_config["model_key"] = model_key
        return fallback_model, fallback_config
    import timm

    errors: list[str] = []
    for timm_name in config.TIMM_MODEL_ALIASES.get(model_key, [model_key]):
        try:
            model = timm.create_model(timm_name, pretrained=True, num_classes=config.NUM_CLASSES)
            return model, {"library": "timm", "pretrained_source": timm_name, "model_key": model_key}
        except Exception as exc:
            errors.append(f"{timm_name}: {repr(exc)}")
    raise RuntimeError(f"Could not create model {model_key}: " + " | ".join(errors))


def model_size_mb(model) -> float:
    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()) / (1024 * 1024)


def make_run_config(model_key: str, model_name: str, condition: dict[str, Any], output_dir: str | Path, smoke_test: bool, micro_batch: int | None = None) -> dict[str, Any]:
    return {
        "dataset_id": config.DATASET_ID,
        "model_key": model_key,
        "model_name": model_name,
        "condition": condition,
        "loss_key": condition["loss_key"],
        "randaugment": bool(condition["randaugment"]),
        "seed": config.SEED,
        "repeat": config.REPEAT,
        "split_seed": config.SPLIT_SEED,
        "epochs": config.epochs_for(smoke_test),
        "smoke_test": bool(smoke_test),
        "micro_batch": micro_batch,
        "torch_effective_batch": config.TORCH_EFFECTIVE_BATCH,
        "output_dir": str(Path(output_dir)),
    }


def predict_torch(model, loader, source_frame: pd.DataFrame, run_id: str, model_name: str, backend: str, loss_name: str, condition_key: str, randaugment: bool, split_name: str, device, batch_size: int, output_dir: str | Path | None = None, progress_enabled: bool = True):
    torch, _nn, F, _optim, _DataLoader, _Dataset, _models, _transforms, _InterpolationMode = _torch_stack()
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    indices: list[int] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    if progress_enabled:
        log_event("Starting Torch prediction.", run_id=run_id, output_dir=output_dir, extra={"split": split_name, "images": len(source_frame), "batch_size": batch_size})
    with torch.no_grad():
        for inputs, labels, batch_indices in progress_iter(loader, desc=f"{run_id} {split_name}", total=len(loader), enabled=progress_enabled, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs).float()
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(labels.numpy().astype(int).tolist())
            y_pred.extend(preds.astype(int).tolist())
            probabilities.extend(probs.tolist())
            indices.extend(batch_indices.numpy().astype(int).tolist())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    ordered = source_frame.reset_index(drop=True).iloc[indices].reset_index(drop=True)
    predictions = prediction_frame(ordered, run_id, model_name, backend, loss_name, condition_key, randaugment, split_name, y_true, y_pred, probabilities)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "inference_time_s": elapsed,
        "latency_ms_image": 1000.0 * elapsed / max(1, len(y_true)),
        "fps": len(y_true) / elapsed if elapsed > 0 else None,
        "prediction_batch": batch_size,
    })
    if progress_enabled:
        log_event("Finished Torch prediction.", run_id=run_id, output_dir=output_dir, extra={
            "split": split_name,
            "images": len(y_true),
            "batch_size": batch_size,
            "elapsed_s": round(elapsed, 3),
            "latency_ms_image": metrics["latency_ms_image"],
            "macro_f1": metrics["macro_f1"],
        })
    return metrics, predictions, y_true, y_pred


def train_torch_once(model_key: str, model_name: str, condition: dict[str, Any], split_manifest: pd.DataFrame, output_dir: str | Path, resume: bool, smoke_test: bool, micro_batch: int, progress_enabled: bool = True):
    torch, _nn, _F, optim, DataLoader, _Dataset, _models, _transforms, _InterpolationMode = _torch_stack()
    paths = config.output_paths(output_dir)
    ensure_dir(paths["runs"])
    run_id = torch_run_id(model_key, condition)
    run_dir = ensure_dir(paths["runs"] / run_id)
    run_config = make_run_config(model_key, model_name, condition, output_dir, smoke_test, micro_batch=micro_batch)
    run_hash = stable_hash(run_config)
    backend_hint = "torchvision" if model_key in {config.CONVNEXT_CORE_MODEL_KEY, "mobilenet_v3_large", "shufflenet_v2_x1_0", "squeezenet1_1"} else "timm"
    if resume:
        completed, reason = is_run_completed(run_id, output_dir, run_hash, backend_hint)
        if completed:
            if progress_enabled:
                log_event("Skipping completed Torch run after strict resume check.", run_id=run_id, output_dir=output_dir, extra={"reason": reason})
            return {"run_id": run_id, "status": "skipped_completed", "skip_reason": reason}
    if progress_enabled:
        log_event("Starting Torch run.", run_id=run_id, output_dir=output_dir, extra={
            "model": model_name,
            "model_key": model_key,
            "condition": condition["condition_key"],
            "loss": condition["loss_key"],
            "randaugment": condition["randaugment"],
            "micro_batch": micro_batch,
            "epochs": config.epochs_for(smoke_test),
        })
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_frame = split_manifest[split_manifest["split"] == "train"].copy()
    val_frame = split_manifest[split_manifest["split"] == "val"].copy()
    test_frame = split_manifest[split_manifest["split"] == "test"].copy()
    if progress_enabled:
        log_event("Torch split sizes resolved.", run_id=run_id, output_dir=output_dir, extra={"train": len(train_frame), "val": len(val_frame), "test": len(test_frame)})
    accumulation_steps = max(1, config.TORCH_EFFECTIVE_BATCH // micro_batch)
    if progress_enabled:
        log_event("Creating Torch model.", run_id=run_id, output_dir=output_dir, extra={"model_key": model_key, "device": str(device)})
    model, model_cfg = create_torch_model(model_key)
    backend = model_cfg["library"]
    model = model.to(device)
    criterion = make_loss(condition["loss_key"]).to(device)
    train_loader = DataLoader(ManifestDataset(train_frame, train_transform(condition["randaugment"])), batch_size=micro_batch, shuffle=True, num_workers=config.TORCH_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(ManifestDataset(val_frame, eval_transform()), batch_size=config.TORCH_EVAL_BATCH, shuffle=False, num_workers=config.TORCH_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(ManifestDataset(test_frame, eval_transform()), batch_size=config.TORCH_EVAL_BATCH, shuffle=False, num_workers=config.TORCH_WORKERS, pin_memory=torch.cuda.is_available())
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "resolved_model_config.json", model_cfg)
    write_json(run_dir / "resolved_transform_config.json", transform_audit(condition["randaugment"]))
    write_json(run_dir / "run_audit.json", {**run_config, "config_hash": run_hash, "backend": backend, "status": "started"})
    epochs = config.epochs_for(smoke_test)
    train_start = time.time()
    history: list[dict[str, Any]] = []
    best_metric = math.inf
    best_epoch = 0
    best_state = None
    no_improve = 0
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP and device.type == "cuda")

    if model_key == config.CONVNEXT_CORE_MODEL_KEY:
        phases = [
            {"name": "classifier_warmup", "start": 0, "end": min(config.CONVNEXT_WARMUP_EPOCHS, epochs), "optimizer_kind": "warmup"},
        ]
        if epochs > config.CONVNEXT_WARMUP_EPOCHS:
            phases.append({
                "name": "high_level_finetune",
                "start": config.CONVNEXT_WARMUP_EPOCHS,
                "end": epochs,
                "optimizer_kind": "finetune",
            })
        patience = config.CONVNEXT_PATIENCE
    else:
        phases = [{"name": "single_phase", "start": 0, "end": epochs, "optimizer_kind": "single"}]
        patience = config.LIGHTWEIGHT_TORCH_PATIENCE

    for phase in phases:
        phase_start = time.time()
        if phase["optimizer_kind"] == "warmup":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.CONVNEXT_WARMUP_LR)
        elif phase["optimizer_kind"] == "finetune":
            model.unfreeze_finetune_layers()
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("classifier"):
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            optimizer = optim.Adam([
                {"params": backbone_params, "lr": config.CONVNEXT_BACKBONE_LR},
                {"params": head_params, "lr": config.CONVNEXT_HEAD_LR},
            ])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=config.LIGHTWEIGHT_LR, weight_decay=config.TORCH_WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.CONVNEXT_STEP_SIZE, gamma=config.CONVNEXT_STEP_GAMMA)
        if progress_enabled:
            log_event("Starting Torch training phase.", run_id=run_id, output_dir=output_dir, extra={"phase": phase["name"], "start_epoch": phase["start"] + 1, "end_epoch": phase["end"], "optimizer_kind": phase["optimizer_kind"]})
        for epoch in range(phase["start"], phase["end"]):
            epoch_start = time.time()
            if progress_enabled:
                log_event("Starting Torch epoch.", run_id=run_id, output_dir=output_dir, extra={"phase": phase["name"], "epoch": epoch + 1, "total_epochs": epochs})
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            seen_batches = 0
            for step, (inputs, labels, _indices) in enumerate(progress_iter(train_loader, desc=f"{run_id} {phase['name']} epoch {epoch + 1}", total=len(train_loader), enabled=progress_enabled, leave=False)):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=config.USE_AMP and device.type == "cuda"):
                    logits = model(inputs)
                    loss = criterion(logits.float(), labels) / accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.detach().cpu()) * accumulation_steps
                seen_batches += 1
            scheduler.step()
            val_metrics, _val_predictions, _val_true, _val_pred = predict_torch(model, val_loader, val_frame, run_id, model_name, backend, LOSS_CONFIG[condition["loss_key"]]["name"], condition["condition_key"], condition["randaugment"], "val", device, config.TORCH_EVAL_BATCH, output_dir=output_dir, progress_enabled=progress_enabled)
            val_loss_proxy = 1.0 - float(val_metrics["macro_f1"])
            improved = val_loss_proxy < best_metric
            if improved:
                best_metric = val_loss_proxy
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            history.append({
                "epoch": epoch + 1,
                "phase": phase["name"],
                "train_loss": running_loss / max(1, seen_batches),
                "val_macro_f1": val_metrics["macro_f1"],
                "micro_batch": micro_batch,
                "accumulation_steps": accumulation_steps,
                "lr": optimizer.param_groups[0]["lr"],
            })
            save_csv(pd.DataFrame(history), run_dir / "training_history.csv")
            if progress_enabled:
                log_event("Finished Torch epoch.", run_id=run_id, output_dir=output_dir, extra={
                    "phase": phase["name"],
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": history[-1]["train_loss"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "best_val_macro_f1": 1.0 - best_metric,
                    "no_improve": no_improve,
                    "patience": patience,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_s": round(time.time() - epoch_start, 3),
                })
            if no_improve >= patience:
                if progress_enabled:
                    log_event("Early stopping patience reached.", run_id=run_id, output_dir=output_dir, extra={"phase": phase["name"], "no_improve": no_improve, "patience": patience})
                break
        if progress_enabled:
            log_event("Finished Torch training phase.", run_id=run_id, output_dir=output_dir, extra={"phase": phase["name"], "elapsed_s": round(time.time() - phase_start, 3)})
    if best_state is None:
        raise RuntimeError("No best checkpoint was captured.")
    model.load_state_dict(best_state)
    checkpoint_path = run_dir / "checkpoint_best.pt"
    torch.save({
        "model_state_dict": best_state,
        "model_key": model_key,
        "model_name": model_name,
        "backend": backend,
        "class_names": config.CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_val_macro_f1": 1.0 - best_metric,
        "run_config": run_config,
    }, checkpoint_path)
    if progress_enabled:
        log_event("Saved best Torch checkpoint.", run_id=run_id, output_dir=output_dir, extra={"checkpoint": str(checkpoint_path), "best_epoch": best_epoch, "best_val_macro_f1": 1.0 - best_metric})
        log_event("Starting final validation prediction.", run_id=run_id, output_dir=output_dir)
    val_metrics, val_predictions, val_true, val_pred = predict_torch(model, val_loader, val_frame, run_id, model_name, backend, LOSS_CONFIG[condition["loss_key"]]["name"], condition["condition_key"], condition["randaugment"], "val", device, config.TORCH_EVAL_BATCH, output_dir=output_dir, progress_enabled=progress_enabled)
    if progress_enabled:
        log_event("Starting final test prediction.", run_id=run_id, output_dir=output_dir)
    test_metrics, test_predictions, test_true, test_pred = predict_torch(model, test_loader, test_frame, run_id, model_name, backend, LOSS_CONFIG[condition["loss_key"]]["name"], condition["condition_key"], condition["randaugment"], "test", device, config.TORCH_EVAL_BATCH, output_dir=output_dir, progress_enabled=progress_enabled)
    save_prediction_artifacts(run_dir, val_metrics, val_predictions, val_true, val_pred, split_name="val")
    save_prediction_artifacts(run_dir, test_metrics, test_predictions, test_true, test_pred, split_name="test")
    save_csv(confusion_count_frame(test_metrics, run_id, model_name, backend, condition["condition_key"], condition["loss_key"]), run_dir / "confusion_counts.csv")
    metrics = {
        "run_id": run_id,
        "status": "completed",
        "model": model_name,
        "model_key": model_key,
        "backend": backend,
        "loss_key": condition["loss_key"],
        "loss": LOSS_CONFIG[condition["loss_key"]]["name"],
        "condition": condition["condition_key"],
        "randaugment": bool(condition["randaugment"]),
        "seed": config.SEED,
        "repeat": config.REPEAT,
        "split_seed": config.SPLIT_SEED,
        "chosen_micro_batch": micro_batch,
        "accumulation_steps": accumulation_steps,
        "training_time_s": time.time() - train_start,
        "params_m": sum(param.numel() for param in model.parameters()) / 1e6,
        "model_size_mb": model_size_mb(model),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run_audit.json", {**run_config, "config_hash": run_hash, "backend": backend, "checkpoint_path": str(checkpoint_path.resolve()), "completed_at": utc_now(), "status": "completed"})
    write_status(run_dir, "completed", run_id=run_id, completed_at=utc_now(), test_macro_f1=test_metrics["macro_f1"])
    if progress_enabled:
        log_event("Saved final Torch metrics and artifacts.", run_id=run_id, output_dir=output_dir, extra={"test_macro_f1": test_metrics["macro_f1"], "checkpoint": str(checkpoint_path.resolve())})
    cleanup_memory()
    return metrics


def train_torch_with_fallback(model_key: str, model_name: str, condition: dict[str, Any], split_manifest: pd.DataFrame, output_dir: str | Path, resume: bool = True, smoke_test: bool = False, progress_enabled: bool = True) -> dict[str, Any]:
    run_id = torch_run_id(model_key, condition)
    errors: list[dict[str, Any]] = []
    for micro_batch in config.TORCH_MICRO_BATCH_FALLBACKS:
        try:
            if progress_enabled:
                log_event("Attempting Torch batch fallback.", run_id=run_id, output_dir=output_dir, extra={"micro_batch": micro_batch})
            return train_torch_once(model_key, model_name, condition, split_manifest, output_dir, resume=resume, smoke_test=smoke_test, micro_batch=micro_batch, progress_enabled=progress_enabled)
        except Exception as exc:
            errors.append({"micro_batch": micro_batch, "error": repr(exc)})
            if is_oom_error(exc):
                if progress_enabled:
                    log_event("Torch OOM during batch fallback; trying next micro-batch.", level="WARNING", run_id=run_id, output_dir=output_dir, extra={"failed_micro_batch": micro_batch, "error": repr(exc)})
                cleanup_memory()
                continue
            if progress_enabled:
                log_event("Torch run failed with non-OOM error.", level="ERROR", run_id=run_id, output_dir=output_dir, extra={"micro_batch": micro_batch, "error": repr(exc)})
            break
    run_dir = ensure_dir(config.output_paths(output_dir)["runs"] / run_id)
    write_status(run_dir, "failed", run_id=run_id, errors=errors)
    if progress_enabled:
        log_event("Torch run failed after fallback attempts.", level="ERROR", run_id=run_id, output_dir=output_dir, extra={"errors": errors})
    return {"run_id": run_id, "status": "failed", "errors": errors}


def probe_lightweight_availability(output_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_key in config.LIGHTWEIGHT_MODELS:
        row = {"model_key": model_key, "backend": "", "availability_status": "unknown", "pretrained_source": "", "params_m": None, "model_size_mb": None, "error": ""}
        try:
            model, cfg = create_torch_model(model_key)
            row.update({
                "backend": cfg["library"],
                "availability_status": "available_pretrained",
                "pretrained_source": cfg["pretrained_source"],
                "params_m": sum(param.numel() for param in model.parameters()) / 1e6,
                "model_size_mb": model_size_mb(model),
            })
            del model
            cleanup_memory()
        except Exception as exc:
            row["availability_status"] = "unavailable"
            row["error"] = repr(exc)
        rows.append(row)
    frame = pd.DataFrame(rows)
    save_csv(frame, Path(output_dir) / "model_availability_lightweight.csv")
    return frame
