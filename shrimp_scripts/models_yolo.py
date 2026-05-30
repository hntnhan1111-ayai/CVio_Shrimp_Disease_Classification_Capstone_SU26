"""Ultralytics YOLO classification training and prediction."""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .evaluate import compute_metrics, confusion_count_frame, prediction_frame, save_prediction_artifacts
from .losses import LOSS_CONFIG, make_loss
from .progress import log_event
from .utils import cleanup_memory, ensure_dir, is_oom_error, is_run_completed, save_csv, set_seed, sha256_file, stable_hash, utc_now, write_json, write_status


ACTIVE_YOLO_LOSS_KEY = "baseline_ce"


def _torch():
    import torch
    return torch


def _yolo_classification_types():
    import torch.nn as nn
    from ultralytics.nn.tasks import ClassificationModel
    try:
        from ultralytics.models.yolo.classify.train import ClassificationTrainer
    except Exception:
        from ultralytics.models.yolo.classify import ClassificationTrainer
    return nn, ClassificationModel, ClassificationTrainer


def yolo_run_id(model_name: str, condition: dict[str, Any]) -> str:
    return f"ultralytics_{model_name.replace('-', '_')}_{condition['condition_key']}_seed{config.SEED}_repeat{config.REPEAT}"


def list_core_yolo_runs(smoke_test: bool = False) -> list[dict[str, Any]]:
    conditions = config.CORE_CONDITIONS[:1] if smoke_test else config.CORE_CONDITIONS
    return [{"run_id": yolo_run_id(config.YOLO_CORE_MODEL, condition), "backend": "ultralytics", "model": config.YOLO_CORE_MODEL, **condition} for condition in conditions]


def list_yolo_family_runs(smoke_test: bool = False) -> list[dict[str, Any]]:
    models = [config.YOLO_CORE_MODEL] if smoke_test else config.YOLO_FAMILY_MODELS
    return [{"run_id": yolo_run_id(model_name, condition), "backend": "ultralytics", "model": model_name, **condition} for model_name in models for condition in config.FAMILY_CONDITIONS]


def get_custom_trainer_class():
    torch = _torch()
    nn, ClassificationModel, ClassificationTrainer = _yolo_classification_types()

    class PaperClassificationLoss(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.loss_key = getattr(model, "loss_key", ACTIVE_YOLO_LOSS_KEY)
            self.loss_fcn = make_loss(self.loss_key)

        def extract_logits(self, preds, targets):
            if torch.is_tensor(preds):
                return preds
            if isinstance(preds, (list, tuple)):
                for item in preds:
                    if torch.is_tensor(item) and item.ndim == 2 and item.shape[0] == targets.shape[0] and item.shape[1] == config.NUM_CLASSES:
                        return item
                for item in preds:
                    if torch.is_tensor(item) and item.ndim == 2 and item.shape[0] == targets.shape[0]:
                        return item
            raise TypeError("unable_to_extract_yolo_classification_logits")

        def forward(self, preds, batch):
            targets = batch["cls"].long().view(-1)
            logits = self.extract_logits(preds, targets).float()
            targets = targets.to(logits.device)
            self.loss_fcn = self.loss_fcn.to(logits.device)
            loss = self.loss_fcn(logits, targets)
            return loss, loss.detach()

    class PaperClassificationModel(ClassificationModel):
        def init_criterion(self):
            return PaperClassificationLoss(self)

    class PaperClassificationTrainer(ClassificationTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            nc = self.data["nc"] if isinstance(self.data, dict) and "nc" in self.data else config.NUM_CLASSES
            try:
                model = PaperClassificationModel(cfg, nc=nc, verbose=verbose)
            except TypeError:
                model = PaperClassificationModel(cfg, ch=3, nc=nc, verbose=verbose)
            model.loss_key = ACTIVE_YOLO_LOSS_KEY
            if weights:
                model.load(weights)
            return model

    return PaperClassificationTrainer


def yolo_device_arg():
    torch = _torch()
    return 0 if torch.cuda.is_available() else "cpu"


def yolo_supports_auto_augment() -> tuple[bool, str]:
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "auto_augment" in DEFAULT_CFG_DICT:
            return True, "DEFAULT_CFG_DICT"
    except Exception as exc:
        cfg_error = repr(exc)
    try:
        from ultralytics import YOLO
        signature = inspect.signature(YOLO.train)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return True, "YOLO.train accepts **kwargs"
        if "auto_augment" in signature.parameters:
            return True, "YOLO.train signature"
    except Exception as exc:
        return False, f"auto_augment support check failed: {repr(exc)}"
    return False, f"auto_augment not found in Ultralytics config; cfg_error={locals().get('cfg_error', '')}"


def yolo_train_kwargs(run_id: str, condition: dict[str, Any], output_dir: str | Path, batch: int, smoke_test: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    supports_auto, support_reason = yolo_supports_auto_augment()
    auto_augment = "randaugment" if condition["randaugment"] else None
    kwargs = {
        "data": str(config.output_paths(output_dir)["yolo_dataset"]),
        "task": "classify",
        "imgsz": config.IMG_SIZE,
        "epochs": config.epochs_for(smoke_test),
        "patience": config.YOLO_PATIENCE,
        "batch": batch,
        "workers": config.YOLO_WORKERS,
        "optimizer": config.YOLO_OPTIMIZER,
        "lr0": config.YOLO_LR0,
        "lrf": config.YOLO_LRF,
        "cos_lr": config.YOLO_COS_LR,
        "cache": config.YOLO_CACHE,
        "amp": config.USE_AMP,
        "seed": config.SEED,
        "project": str(config.output_paths(output_dir)["runs"] / "ultralytics_train"),
        "name": run_id,
        "exist_ok": True,
        "device": yolo_device_arg(),
        "verbose": True,
        "plots": False,
    }
    if supports_auto:
        kwargs["auto_augment"] = auto_augment
    audit = {
        "auto_augment_supported": supports_auto,
        "auto_augment_support_reason": support_reason,
        "requested_auto_augment": auto_augment,
        "train_kwargs_auto_augment": kwargs.get("auto_augment", "not_passed"),
        "augmentation_train_kwargs": {k: kwargs.get(k) for k in ["auto_augment", "imgsz", "cache", "amp"] if k in kwargs},
    }
    return kwargs, audit


def make_run_config(model_name: str, condition: dict[str, Any], output_dir: str | Path, smoke_test: bool, batch: int) -> dict[str, Any]:
    return {
        "dataset_id": config.DATASET_ID,
        "model_name": model_name,
        "model_key": model_name.replace("-", "_"),
        "condition": condition,
        "loss_key": condition["loss_key"],
        "randaugment": bool(condition["randaugment"]),
        "seed": config.SEED,
        "repeat": config.REPEAT,
        "split_seed": config.SPLIT_SEED,
        "epochs": config.epochs_for(smoke_test),
        "batch": batch,
        "output_dir": str(Path(output_dir)),
    }


def predict_yolo(best_model, frame: pd.DataFrame, run_id: str, model_name: str, loss_name: str, condition_key: str, randaugment: bool, split_name: str, batch: int, output_dir: str | Path | None = None, progress_enabled: bool = True):
    torch = _torch()
    paths = frame["yolo_path"].tolist()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    if progress_enabled:
        log_event("Starting YOLO prediction.", run_id=run_id, output_dir=output_dir, extra={"split": split_name, "images": len(paths), "batch_size": batch})
    results = best_model.predict(source=paths, imgsz=config.IMG_SIZE, batch=batch, device=yolo_device_arg(), verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    y_true = frame["label"].astype(int).tolist()
    probabilities = [[float(result.probs.data[index].detach().cpu()) for index in range(config.NUM_CLASSES)] for result in results]
    y_pred = [int(np.argmax(row)) for row in probabilities]
    predictions = prediction_frame(frame, run_id, model_name, "ultralytics", loss_name, condition_key, randaugment, split_name, y_true, y_pred, probabilities)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "inference_time_s": elapsed,
        "latency_ms_image": 1000.0 * elapsed / max(1, len(y_true)),
        "fps": len(y_true) / elapsed if elapsed > 0 else None,
        "prediction_batch": batch,
    })
    if progress_enabled:
        log_event("Finished YOLO prediction.", run_id=run_id, output_dir=output_dir, extra={
            "split": split_name,
            "images": len(y_true),
            "batch_size": batch,
            "elapsed_s": round(elapsed, 3),
            "latency_ms_image": metrics["latency_ms_image"],
            "macro_f1": metrics["macro_f1"],
        })
    return metrics, predictions, y_true, y_pred


def train_yolo_once(model_name: str, condition: dict[str, Any], yolo_manifest: pd.DataFrame, output_dir: str | Path, resume: bool, smoke_test: bool, batch: int, progress_enabled: bool = True) -> dict[str, Any]:
    global ACTIVE_YOLO_LOSS_KEY
    from ultralytics import YOLO

    paths = config.output_paths(output_dir)
    ensure_dir(paths["runs"])
    run_id = yolo_run_id(model_name, condition)
    run_dir = ensure_dir(paths["runs"] / run_id)
    run_config = make_run_config(model_name, condition, output_dir, smoke_test, batch=batch)
    run_hash = stable_hash(run_config)
    if resume:
        completed, reason = is_run_completed(run_id, output_dir, run_hash, "ultralytics")
        if completed:
            if progress_enabled:
                log_event("Skipping completed YOLO run after strict resume check.", run_id=run_id, output_dir=output_dir, extra={"reason": reason})
            return {"run_id": run_id, "status": "skipped_completed", "skip_reason": reason}
    train_kwargs, augment_audit = yolo_train_kwargs(run_id, condition, output_dir, batch, smoke_test)
    if condition["randaugment"] != (train_kwargs.get("auto_augment") == "randaugment"):
        if not augment_audit["auto_augment_supported"]:
            write_json(run_dir / "run_audit.json", {**run_config, "config_hash": run_hash, **augment_audit, "status": "skipped_auto_augment_unsupported"})
            write_status(run_dir, "skipped", run_id=run_id, error=augment_audit["auto_augment_support_reason"])
            if progress_enabled:
                log_event("Skipping YOLO run because auto_augment cannot be controlled.", level="WARNING", run_id=run_id, output_dir=output_dir, extra=augment_audit)
            return {"run_id": run_id, "status": "skipped", "error": augment_audit["auto_augment_support_reason"]}
    write_json(run_dir / "train_kwargs.json", train_kwargs)
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "run_audit.json", {**run_config, "config_hash": run_hash, **augment_audit, "status": "started"})
    if progress_enabled:
        log_event("Starting YOLO run.", run_id=run_id, output_dir=output_dir, extra={
            "model": model_name,
            "condition": condition["condition_key"],
            "loss": condition["loss_key"],
            "randaugment": condition["randaugment"],
            "batch": batch,
            "epochs": config.epochs_for(smoke_test),
            "device": train_kwargs.get("device"),
        })
        log_event("Resolved YOLO train kwargs.", run_id=run_id, output_dir=output_dir, extra={k: str(v) for k, v in train_kwargs.items()})
    set_seed(config.SEED)
    ACTIVE_YOLO_LOSS_KEY = condition["loss_key"]
    start = time.time()
    if progress_enabled:
        log_event("Loading YOLO pretrained weights.", run_id=run_id, output_dir=output_dir, extra={"weights": model_name + ".pt"})
    yolo = YOLO(model_name + ".pt")
    pretrained_path = str(getattr(yolo, "ckpt_path", ""))
    if progress_enabled:
        log_event("Starting native Ultralytics classification training.", run_id=run_id, output_dir=output_dir, extra={"custom_loss": condition["loss_key"] != "baseline_ce"})
    if condition["loss_key"] == "baseline_ce":
        yolo.train(**train_kwargs)
    else:
        yolo.train(trainer=get_custom_trainer_class(), **train_kwargs)
    train_time = time.time() - start
    if progress_enabled:
        log_event("Ultralytics training finished.", run_id=run_id, output_dir=output_dir, extra={"training_time_s": round(train_time, 3)})
    yolo_train_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    best_path = yolo_train_dir / "weights" / "best.pt"
    if not best_path.exists():
        candidates = sorted(yolo_train_dir.glob("**/best.pt"))
        if not candidates:
            raise FileNotFoundError(str(best_path))
        best_path = candidates[-1]
    best_model = YOLO(str(best_path))
    val_frame = yolo_manifest[yolo_manifest["split"] == "val"].copy()
    test_frame = yolo_manifest[yolo_manifest["split"] == "test"].copy()
    loss_name = LOSS_CONFIG[condition["loss_key"]]["name"]
    if progress_enabled:
        log_event("Starting YOLO validation prediction.", run_id=run_id, output_dir=output_dir, extra={"checkpoint": str(best_path)})
    val_metrics, val_predictions, val_true, val_pred = predict_yolo(best_model, val_frame, run_id, model_name, loss_name, condition["condition_key"], condition["randaugment"], "val", batch, output_dir=output_dir, progress_enabled=progress_enabled)
    if progress_enabled:
        log_event("Starting YOLO test prediction.", run_id=run_id, output_dir=output_dir)
    test_metrics, test_predictions, test_true, test_pred = predict_yolo(best_model, test_frame, run_id, model_name, loss_name, condition["condition_key"], condition["randaugment"], "test", batch, output_dir=output_dir, progress_enabled=progress_enabled)
    save_prediction_artifacts(run_dir, val_metrics, val_predictions, val_true, val_pred, split_name="val")
    save_prediction_artifacts(run_dir, test_metrics, test_predictions, test_true, test_pred, split_name="test")
    save_csv(confusion_count_frame(test_metrics, run_id, model_name, "ultralytics", condition["condition_key"], condition["loss_key"]), run_dir / "confusion_counts.csv")
    params_m = None
    try:
        params_m = sum(param.numel() for param in best_model.model.parameters()) / 1e6
    except Exception:
        pass
    metrics = {
        "run_id": run_id,
        "status": "completed",
        "model": model_name,
        "model_key": model_name.replace("-", "_"),
        "backend": "ultralytics",
        "loss_key": condition["loss_key"],
        "loss": loss_name,
        "condition": condition["condition_key"],
        "randaugment": bool(condition["randaugment"]),
        "seed": config.SEED,
        "repeat": config.REPEAT,
        "split_seed": config.SPLIT_SEED,
        "chosen_yolo_batch": batch,
        "training_time_s": train_time,
        "params_m": params_m,
        "model_size_mb": best_path.stat().st_size / (1024 * 1024),
        "checkpoint_path": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "pretrained_weight_path": pretrained_path,
        "pretrained_weight_sha256": sha256_file(pretrained_path) if pretrained_path and Path(pretrained_path).is_file() else "",
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run_audit.json", {**run_config, "config_hash": run_hash, **augment_audit, "backend": "ultralytics", "checkpoint_path": str(best_path.resolve()), "pretrained_weight_path": pretrained_path, "completed_at": utc_now(), "status": "completed"})
    write_status(run_dir, "completed", run_id=run_id, completed_at=utc_now(), test_macro_f1=test_metrics["macro_f1"])
    if progress_enabled:
        log_event("Saved final YOLO metrics and artifacts.", run_id=run_id, output_dir=output_dir, extra={"test_macro_f1": test_metrics["macro_f1"], "checkpoint": str(best_path.resolve())})
    cleanup_memory()
    return metrics


def train_yolo_with_fallback(model_name: str, condition: dict[str, Any], yolo_manifest: pd.DataFrame, output_dir: str | Path, resume: bool = True, smoke_test: bool = False, progress_enabled: bool = True) -> dict[str, Any]:
    run_id = yolo_run_id(model_name, condition)
    errors: list[dict[str, Any]] = []
    for batch in config.YOLO_BATCH_FALLBACKS:
        try:
            if progress_enabled:
                log_event("Attempting YOLO batch fallback.", run_id=run_id, output_dir=output_dir, extra={"batch": batch})
            return train_yolo_once(model_name, condition, yolo_manifest, output_dir, resume=resume, smoke_test=smoke_test, batch=batch, progress_enabled=progress_enabled)
        except Exception as exc:
            errors.append({"batch": batch, "error": repr(exc)})
            if is_oom_error(exc):
                if progress_enabled:
                    log_event("YOLO OOM during batch fallback; trying next batch.", level="WARNING", run_id=run_id, output_dir=output_dir, extra={"failed_batch": batch, "error": repr(exc)})
                cleanup_memory()
                continue
            if progress_enabled:
                log_event("YOLO run failed with non-OOM error.", level="ERROR", run_id=run_id, output_dir=output_dir, extra={"batch": batch, "error": repr(exc)})
            break
    run_dir = ensure_dir(config.output_paths(output_dir)["runs"] / run_id)
    write_status(run_dir, "failed", run_id=run_id, errors=errors)
    if progress_enabled:
        log_event("YOLO run failed after fallback attempts.", level="ERROR", run_id=run_id, output_dir=output_dir, extra={"errors": errors})
    return {"run_id": run_id, "status": "failed", "errors": errors}


def probe_yolo_availability(output_dir: str | Path, progress_enabled: bool = True) -> pd.DataFrame:
    from ultralytics import YOLO
    rows: list[dict[str, Any]] = []
    for model_name in config.YOLO_FAMILY_MODELS:
        if progress_enabled:
            log_event("Probing YOLO model availability.", output_dir=output_dir, extra={"model": model_name})
        row = {"model": model_name, "backend": "ultralytics", "availability_status": "unknown", "params_m": None, "model_size_mb": None, "pretrained_source": model_name + ".pt", "error": ""}
        try:
            model = YOLO(model_name + ".pt")
            ckpt_path = str(getattr(model, "ckpt_path", ""))
            row["availability_status"] = "available_pretrained"
            row["pretrained_weight_path"] = ckpt_path
            row["pretrained_weight_sha256"] = sha256_file(ckpt_path) if ckpt_path and Path(ckpt_path).is_file() else ""
            row["model_size_mb"] = Path(ckpt_path).stat().st_size / (1024 * 1024) if ckpt_path and Path(ckpt_path).is_file() else None
            row["params_m"] = sum(param.numel() for param in model.model.parameters()) / 1e6
        except Exception as exc:
            row["availability_status"] = "unavailable"
            row["error"] = repr(exc)
        if progress_enabled:
            log_event("YOLO availability probe finished.", output_dir=output_dir, extra={"model": model_name, "status": row["availability_status"], "error": row["error"]})
        rows.append(row)
    frame = pd.DataFrame(rows)
    save_csv(frame, Path(output_dir) / "model_availability_yolo.csv")
    return frame
