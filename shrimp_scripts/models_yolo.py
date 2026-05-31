"""Ultralytics YOLO classification training and prediction."""

from __future__ import annotations

import inspect
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .evaluate import compute_metrics, confusion_count_frame, prediction_frame, save_prediction_artifacts
from .losses import LOSS_CHECKPOINT_SAFE_CLASSES, LOSS_CONFIG, make_loss
from .progress import log_event
from .utils import cleanup_memory, ensure_dir, is_oom_error, is_run_completed, read_json, required_outputs_exist, save_csv, set_seed, sha256_file, stable_hash, utc_now, write_json, write_status


ACTIVE_YOLO_LOSS_KEY = "baseline_ce"
YOLO_TRAINING_IMPL_VERSION = "paper_class_order_custom_loss_v2"

try:
    import torch as _TORCH
    import torch.nn as _NN
    from ultralytics.data.dataset import ClassificationDataset as _UltralyticsClassificationDataset
    from ultralytics.nn.tasks import ClassificationModel as _UltralyticsClassificationModel
    try:
        from ultralytics.models.yolo.classify.train import ClassificationTrainer as _UltralyticsClassificationTrainer
    except Exception:
        from ultralytics.models.yolo.classify import ClassificationTrainer as _UltralyticsClassificationTrainer
    _YOLO_CUSTOM_CLASS_IMPORT_ERROR = None
except Exception as _import_exc:
    _TORCH = None
    _NN = None
    _UltralyticsClassificationDataset = object
    _UltralyticsClassificationModel = object
    _UltralyticsClassificationTrainer = object
    _YOLO_CUSTOM_CLASS_IMPORT_ERROR = _import_exc


def _torch():
    import torch
    return torch


def paper_yolo_names() -> dict[int, str]:
    return {index: name for index, name in enumerate(config.CLASS_NAMES)}


def paper_yolo_class_to_idx() -> dict[str, int]:
    return {name: index for index, name in enumerate(config.CLASS_NAMES)}


def normalize_yolo_folder_name(folder_name: str) -> str:
    candidates = [str(folder_name)]
    if "_" in folder_name:
        candidates.append(folder_name.split("_", 1)[1])
    if ". " in folder_name:
        candidates.append(folder_name.split(". ", 1)[1])
    for candidate in candidates:
        if candidate in config.CLASS_NAMES:
            return candidate
    for class_name in config.CLASS_NAMES:
        if str(folder_name).endswith(class_name):
            return class_name
    raise ValueError(f"unknown_yolo_class_folder:{folder_name}")


def yolo_checkpoint_paths(train_kwargs: dict[str, Any]) -> tuple[Path, Path]:
    train_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    return train_dir / "weights" / "best.pt", train_dir / "weights" / "last.pt"


def write_yolo_dataset_metadata(output_dir: str | Path) -> Path:
    paths = config.output_paths(output_dir)
    metadata_path = paths["output"] / "yolo_dataset_metadata.yaml"
    lines = [
        f"path: {paths['yolo_dataset'].as_posix()}",
        "train: train",
        "val: val",
        "test: test",
        "names:",
    ]
    for index, class_name in paper_yolo_names().items():
        lines.append(f"  {index}: {class_name}")
    ensure_dir(metadata_path.parent)
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata_path


def yolo_audit_context(
    output_dir: str | Path,
    train_kwargs: dict[str, Any] | None = None,
    *,
    model_names: Any = None,
    checkpoint_path: str | Path | None = None,
    last_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = config.output_paths(output_dir)
    dataset_yaml_path = write_yolo_dataset_metadata(output_dir)
    context: dict[str, Any] = {
        "backend": "ultralytics",
        "dataset_path": str(paths["yolo_dataset"]),
        "dataset_yaml_path": str(dataset_yaml_path),
        "dataset_source_type": "classification_folder",
        "class_names": list(config.CLASS_NAMES),
        "class_to_idx": paper_yolo_class_to_idx(),
        "yolo_data_yaml_names": paper_yolo_names(),
        "yolo_data_yaml_path": str(dataset_yaml_path),
        "yolo_label_order_source": "PaperClassificationDataset remaps torchvision ImageFolder labels to config.CLASS_NAMES order.",
        "yolo_training_impl_version": YOLO_TRAINING_IMPL_VERSION,
        "trainer_class": "shrimp_scripts.models_yolo.PaperClassificationTrainer",
        "model_class": "shrimp_scripts.models_yolo.PaperClassificationModel",
        "criterion_class": "shrimp_scripts.models_yolo.PaperClassificationLoss",
        "dataset_class": "shrimp_scripts.models_yolo.PaperClassificationDataset",
    }
    if train_kwargs is not None:
        planned_best, planned_last = yolo_checkpoint_paths(train_kwargs)
        context["train_kwargs"] = dict(train_kwargs)
        context["planned_checkpoint_path"] = str(planned_best)
        context["planned_last_checkpoint_path"] = str(planned_last)
        context["checkpoint_path"] = str(planned_best)
        context["last_checkpoint_path"] = str(planned_last)
    if model_names is not None:
        context["model_names"] = model_names
    if checkpoint_path is not None:
        context["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    if last_checkpoint_path is not None:
        context["last_checkpoint_path"] = str(Path(last_checkpoint_path).resolve())
    return context


if _YOLO_CUSTOM_CLASS_IMPORT_ERROR is None:

    class PaperClassificationDataset(_UltralyticsClassificationDataset):
        """Ultralytics classification dataset with paper class order."""

        def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
            super().__init__(root=root, args=args, augment=augment, prefix=prefix)
            self.original_class_to_idx = dict(getattr(self.base, "class_to_idx", {}))
            class_to_idx = paper_yolo_class_to_idx()
            remapped_samples = []
            for sample in self.samples:
                item = list(sample)
                class_name = normalize_yolo_folder_name(Path(item[0]).parent.name)
                item[1] = class_to_idx[class_name]
                remapped_samples.append(item)
            self.samples = remapped_samples
            self.targets = [int(sample[1]) for sample in self.samples]
            self.classes = list(config.CLASS_NAMES)
            self.class_to_idx = class_to_idx
            if hasattr(self.base, "classes"):
                self.base.classes = list(config.CLASS_NAMES)
            if hasattr(self.base, "class_to_idx"):
                self.base.class_to_idx = class_to_idx
            if hasattr(self.base, "targets"):
                self.base.targets = list(self.targets)

    class PaperClassificationLoss(_NN.Module):
        def __init__(self, model):
            super().__init__()
            self.loss_key = getattr(model, "loss_key", ACTIVE_YOLO_LOSS_KEY)
            self.loss_fcn = make_loss(self.loss_key)

        def extract_logits(self, preds, targets):
            if _TORCH.is_tensor(preds):
                return preds
            if isinstance(preds, (list, tuple)):
                if len(preds) > 1 and _TORCH.is_tensor(preds[1]):
                    return preds[1]
                if len(preds) == 1 and _TORCH.is_tensor(preds[0]):
                    return preds[0]
                for item in preds:
                    if _TORCH.is_tensor(item) and item.ndim == 2 and item.shape[0] == targets.shape[0] and item.shape[1] == config.NUM_CLASSES:
                        return item
                for item in preds:
                    if _TORCH.is_tensor(item) and item.ndim == 2 and item.shape[0] == targets.shape[0]:
                        return item
            raise TypeError("unable_to_extract_yolo_classification_logits")

        def forward(self, preds, batch):
            targets = batch["cls"].long().view(-1)
            logits = self.extract_logits(preds, targets).float()
            targets = targets.to(logits.device)
            self.loss_fcn = self.loss_fcn.to(logits.device)
            loss = self.loss_fcn(logits, targets)
            return loss, loss.detach()


    class PaperClassificationModel(_UltralyticsClassificationModel):
        def init_criterion(self):
            return PaperClassificationLoss(self)


    class PaperClassificationTrainer(_UltralyticsClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            return PaperClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

        def get_model(self, cfg=None, weights=None, verbose=True):
            nc = self.data["nc"] if isinstance(self.data, dict) and "nc" in self.data else config.NUM_CLASSES
            channels = self.data.get("channels", 3) if isinstance(self.data, dict) else 3
            try:
                model = PaperClassificationModel(cfg, nc=nc, ch=channels, verbose=verbose)
            except TypeError:
                model = PaperClassificationModel(cfg, nc=nc, verbose=verbose)
            if weights:
                model.load(weights)
            model.loss_key = ACTIVE_YOLO_LOSS_KEY
            model.names = paper_yolo_names()
            for module in model.modules():
                if getattr(self.args, "pretrained", True) is False and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                if isinstance(module, _TORCH.nn.Dropout) and getattr(self.args, "dropout", 0):
                    module.p = self.args.dropout
            for parameter in model.parameters():
                parameter.requires_grad = True
            return model

        def set_model_attributes(self):
            try:
                super().set_model_attributes()
            except AttributeError:
                pass
            if isinstance(self.data, dict):
                self.data["names"] = paper_yolo_names()
                self.data["nc"] = config.NUM_CLASSES
            self.model.names = paper_yolo_names()

else:
    PaperClassificationDataset = None
    PaperClassificationLoss = None
    PaperClassificationModel = None
    PaperClassificationTrainer = None


def register_yolo_checkpoint_safe_globals() -> dict[str, Any]:
    if _TORCH is None:
        return {"registered": False, "reason": "torch_unavailable"}
    add_safe_globals = getattr(getattr(_TORCH, "serialization", None), "add_safe_globals", None)
    if not callable(add_safe_globals):
        return {"registered": False, "reason": "torch_serialization_add_safe_globals_unavailable"}
    safe_classes = [
        cls for cls in [
            PaperClassificationLoss,
            PaperClassificationModel,
            PaperClassificationTrainer,
            PaperClassificationDataset,
            *LOSS_CHECKPOINT_SAFE_CLASSES,
        ] if cls is not None
    ]
    try:
        add_safe_globals(safe_classes)
        return {"registered": True, "classes": [f"{cls.__module__}.{cls.__qualname__}" for cls in safe_classes]}
    except Exception as exc:
        return {"registered": False, "reason": repr(exc)}


def yolo_run_id(model_name: str, condition: dict[str, Any]) -> str:
    base = f"ultralytics_{model_name.replace('-', '_')}_{condition['condition_key']}_seed{config.SEED}_repeat{config.REPEAT}"
    experiment_key = str(condition.get("experiment_key", "")).strip()
    return f"{experiment_key}_{base}" if experiment_key else base


def list_core_yolo_runs(smoke_test: bool = False) -> list[dict[str, Any]]:
    conditions = config.CORE_CONDITIONS[:1] if smoke_test else config.CORE_CONDITIONS
    return [{"run_id": yolo_run_id(config.YOLO_CORE_MODEL, condition), "backend": "ultralytics", "model": config.YOLO_CORE_MODEL, **condition} for condition in conditions]


def list_yolo_family_runs(smoke_test: bool = False) -> list[dict[str, Any]]:
    models = [config.YOLO_CORE_MODEL] if smoke_test else config.YOLO_FAMILY_MODELS
    return [{"run_id": yolo_run_id(model_name, condition), "backend": "ultralytics", "model": model_name, **condition} for model_name in models for condition in config.FAMILY_CONDITIONS]


def get_custom_trainer_class():
    if PaperClassificationTrainer is None:
        raise ImportError(f"Ultralytics custom classification trainer is unavailable: {_YOLO_CUSTOM_CLASS_IMPORT_ERROR!r}")
    return PaperClassificationTrainer


def exception_details(exc: BaseException) -> dict[str, str]:
    return {
        "exception_type": exc.__class__.__name__,
        "exception_message": str(exc),
        "exception_repr": repr(exc),
        "traceback": traceback.format_exc(),
    }


def yolo_device_arg():
    try:
        torch = _torch()
    except Exception:
        return "cpu"
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
        if "auto_augment" in signature.parameters:
            return True, "YOLO.train signature"
    except Exception as exc:
        return False, f"auto_augment support check failed: {repr(exc)}"
    return False, f"auto_augment not found in Ultralytics config; cfg_error={locals().get('cfg_error', '')}"


def yolo_dependency_unavailable(reason: str) -> bool:
    return "ModuleNotFoundError" in reason and "ultralytics" in reason


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
        "auto_augment_control_required": True,
        "augmentation_train_kwargs": {k: kwargs.get(k) for k in ["auto_augment", "imgsz", "cache", "amp"] if k in kwargs},
    }
    return kwargs, audit


def make_run_config(model_name: str, condition: dict[str, Any], output_dir: str | Path, smoke_test: bool, batch: int) -> dict[str, Any]:
    run_config = {
        "dataset_id": config.DATASET_ID,
        "backend": "ultralytics",
        "model": model_name,
        "model_name": model_name,
        "model_key": model_name.replace("-", "_"),
        "condition_key": condition["condition_key"],
        "condition": condition,
        "loss_key": condition["loss_key"],
        "loss_name": LOSS_CONFIG[condition["loss_key"]]["name"],
        "randaugment": bool(condition["randaugment"]),
        "seed": config.SEED,
        "repeat": config.REPEAT,
        "split_seed": config.SPLIT_SEED,
        "epochs": config.epochs_for(smoke_test),
        "batch": batch,
        "output_dir": str(Path(output_dir)),
        "paper_class_order": list(config.CLASS_NAMES),
        "class_to_idx": paper_yolo_class_to_idx(),
        "yolo_training_impl_version": YOLO_TRAINING_IMPL_VERSION,
        "trainer_class": "shrimp_scripts.models_yolo.PaperClassificationTrainer",
        "model_class": "shrimp_scripts.models_yolo.PaperClassificationModel",
        "criterion_class": "shrimp_scripts.models_yolo.PaperClassificationLoss",
        "dataset_class": "shrimp_scripts.models_yolo.PaperClassificationDataset",
    }
    if condition.get("experiment_key"):
        run_config["experiment_key"] = condition["experiment_key"]
        run_config["experiment_group"] = condition.get("experiment_group", condition["experiment_key"])
    return run_config


def _existing_file(path_value: Any) -> tuple[bool, str]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return False, ""
    path = Path(path_text)
    return path.is_file() and path.stat().st_size > 0, path_text


def _text_contains_original_pickle_error(*payloads: Any) -> bool:
    text = " ".join(str(payload) for payload in payloads)
    return "Can't get local object" in text and "PaperClassificationModel" in text


def verify_yolo_run_artifacts(model_name: str, condition: dict[str, Any], output_dir: str | Path, load_checkpoints: bool = True) -> dict[str, Any]:
    run_id = yolo_run_id(model_name, condition)
    run_dir = config.output_paths(output_dir)["runs"] / run_id
    row: dict[str, Any] = {
        "run_id": run_id,
        "model": model_name,
        "condition": condition["condition_key"],
        "loss_key": condition["loss_key"],
        "randaugment": bool(condition["randaugment"]),
        "verification_status": "failed",
        "checks": {},
        "errors": [],
    }
    row["checkpoint_safe_globals"] = register_yolo_checkpoint_safe_globals()
    if not run_dir.exists():
        row["errors"].append("missing_run_directory")
        return row
    status = read_json(run_dir / "status.json", default={})
    audit = read_json(run_dir / "run_audit.json", default={})
    metrics = read_json(run_dir / "metrics.json", default={})
    row["checks"]["run_config_exists"] = (run_dir / "run_config.json").is_file()
    row["checks"]["train_kwargs_exists"] = (run_dir / "train_kwargs.json").is_file()
    row["checks"]["status_completed"] = status.get("status") == "completed"
    row["checks"]["audit_completed"] = audit.get("status") == "completed"
    row["checks"]["metrics_completed"] = metrics.get("status") == "completed"
    row["checks"]["train_kwargs_in_audit"] = isinstance(audit.get("train_kwargs"), dict)
    row["checks"]["class_names_match"] = audit.get("class_names") == list(config.CLASS_NAMES)
    row["checks"]["class_to_idx_matches_paper_order"] = audit.get("class_to_idx") == paper_yolo_class_to_idx()
    yaml_names = {int(key): value for key, value in dict(audit.get("yolo_data_yaml_names", {})).items()} if audit.get("yolo_data_yaml_names") else {}
    row["checks"]["yolo_data_yaml_names_match"] = yaml_names == paper_yolo_names()
    ok_outputs, missing_outputs = required_outputs_exist(run_dir, "ultralytics")
    row["checks"]["required_outputs_exist"] = ok_outputs
    row["missing_outputs"] = missing_outputs
    best_ok, best_path = _existing_file(audit.get("checkpoint_path") or metrics.get("checkpoint_path"))
    last_ok, last_path = _existing_file(audit.get("last_checkpoint_path") or metrics.get("last_checkpoint_path"))
    row["checkpoint_path"] = best_path
    row["last_checkpoint_path"] = last_path
    row["checks"]["best_checkpoint_exists"] = best_ok
    row["checks"]["last_checkpoint_exists"] = last_ok
    row["checks"]["classification_report_exists"] = (run_dir / "classification_report.csv").is_file()
    row["checks"]["confusion_matrix_exists"] = (run_dir / "confusion_matrix.csv").is_file() or (run_dir / "confusion_matrix.json").is_file()
    row["checks"]["test_predictions_exists"] = (run_dir / "test_predictions.csv").is_file()
    row["checks"]["val_predictions_exists"] = (run_dir / "val_predictions.csv").is_file()
    row["checks"]["original_pickle_error_absent"] = not _text_contains_original_pickle_error(status, audit)
    if audit.get("batch") is not None and audit.get("config_hash"):
        smoke_test = int(audit.get("epochs", config.EPOCHS)) == config.SMOKE_TEST_EPOCHS
        expected_hash = stable_hash(make_run_config(model_name, condition, output_dir, smoke_test=smoke_test, batch=int(audit["batch"])))
        row["checks"]["config_hash_matches"] = audit.get("config_hash") == expected_hash
    else:
        row["checks"]["config_hash_matches"] = False
        row["errors"].append("missing_batch_or_config_hash")
    if load_checkpoints and best_ok and last_ok:
        try:
            from ultralytics import YOLO
            YOLO(best_path)
            YOLO(last_path)
            row["checks"]["checkpoints_loadable"] = True
        except Exception as exc:
            row["checks"]["checkpoints_loadable"] = False
            row["errors"].append(f"checkpoint_load_failed:{repr(exc)}")
    elif load_checkpoints:
        row["checks"]["checkpoints_loadable"] = False
    for check_name, passed in row["checks"].items():
        if not passed:
            row["errors"].append(check_name)
    if not missing_outputs and all(bool(value) for value in row["checks"].values()):
        row["verification_status"] = "passed"
    return row


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
    planned_best_path, planned_last_path = yolo_checkpoint_paths(train_kwargs)
    audit_context = yolo_audit_context(output_dir, train_kwargs)
    auto_augment_supported = bool(augment_audit["auto_augment_supported"])
    dependency_missing = yolo_dependency_unavailable(str(augment_audit["auto_augment_support_reason"]))
    auto_augment_matches_condition = condition["randaugment"] == (train_kwargs.get("auto_augment") == "randaugment")
    if not dependency_missing and ((not auto_augment_supported) or not auto_augment_matches_condition):
        write_json(run_dir / "train_kwargs.json", train_kwargs)
        write_json(run_dir / "run_config.json", run_config)
        write_json(run_dir / "run_audit.json", {**run_config, "config_hash": run_hash, **audit_context, **augment_audit, "status": "skipped_auto_augment_unsupported"})
        write_status(run_dir, "skipped", run_id=run_id, error=augment_audit["auto_augment_support_reason"])
        if progress_enabled:
            log_event("Skipping YOLO run because auto_augment cannot be controlled.", level="WARNING", run_id=run_id, output_dir=output_dir, extra=augment_audit)
        return {"run_id": run_id, "status": "skipped", "error": augment_audit["auto_augment_support_reason"]}
    write_json(run_dir / "train_kwargs.json", train_kwargs)
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "run_audit.json", {**run_config, "config_hash": run_hash, **audit_context, **augment_audit, "status": "started"})
    if progress_enabled:
        log_event("Starting YOLO run.", run_id=run_id, output_dir=output_dir, extra={
            "model": model_name,
            "condition": condition["condition_key"],
            "loss": condition["loss_key"],
            "randaugment": condition["randaugment"],
            "batch": batch,
            "epochs": config.epochs_for(smoke_test),
            "device": train_kwargs.get("device"),
            "class_to_idx": paper_yolo_class_to_idx(),
        })
        log_event("Resolved YOLO train kwargs.", run_id=run_id, output_dir=output_dir, extra={k: str(v) for k, v in train_kwargs.items()})
    set_seed(config.SEED)
    ACTIVE_YOLO_LOSS_KEY = condition["loss_key"]
    start = time.time()
    if progress_enabled:
        log_event("Loading YOLO pretrained weights.", run_id=run_id, output_dir=output_dir, extra={"weights": model_name + ".pt"})
    from ultralytics import YOLO
    yolo = YOLO(model_name + ".pt")
    pretrained_path = str(getattr(yolo, "ckpt_path", ""))
    if progress_enabled:
        log_event("Starting native Ultralytics classification training.", run_id=run_id, output_dir=output_dir, extra={"custom_loss": condition["loss_key"] != "baseline_ce"})
    yolo.train(trainer=get_custom_trainer_class(), **train_kwargs)
    train_time = time.time() - start
    if progress_enabled:
        log_event("Ultralytics training finished.", run_id=run_id, output_dir=output_dir, extra={"training_time_s": round(train_time, 3)})
    best_path = planned_best_path
    if not best_path.exists():
        candidates = sorted((Path(train_kwargs["project"]) / train_kwargs["name"]).glob("**/best.pt"))
        if not candidates:
            raise FileNotFoundError(str(best_path))
        best_path = candidates[-1]
    last_path = planned_last_path
    if not last_path.exists():
        candidates = sorted((Path(train_kwargs["project"]) / train_kwargs["name"]).glob("**/last.pt"))
        if not candidates:
            raise FileNotFoundError(str(last_path))
        last_path = candidates[-1]
    safe_globals_audit = register_yolo_checkpoint_safe_globals()
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
        "experiment_key": condition.get("experiment_key", ""),
        "experiment_group": condition.get("experiment_group", ""),
        "seed": config.SEED,
        "repeat": config.REPEAT,
        "split_seed": config.SPLIT_SEED,
        "chosen_yolo_batch": batch,
        "training_time_s": train_time,
        "params_m": params_m,
        "model_size_mb": best_path.stat().st_size / (1024 * 1024),
        "checkpoint_path": str(best_path.resolve()),
        "last_checkpoint_path": str(last_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "pretrained_weight_path": pretrained_path,
        "pretrained_weight_sha256": sha256_file(pretrained_path) if pretrained_path and Path(pretrained_path).is_file() else "",
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run_audit.json", {
        **run_config,
        "config_hash": run_hash,
        **yolo_audit_context(
            output_dir,
            train_kwargs,
            model_names=getattr(best_model, "names", getattr(getattr(best_model, "model", None), "names", None)),
            checkpoint_path=best_path,
            last_checkpoint_path=last_path,
        ),
        **augment_audit,
        "pretrained_weight_path": pretrained_path,
        "checkpoint_safe_globals": safe_globals_audit,
        "completed_at": utc_now(),
        "status": "completed",
    })
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
            failure = {"batch": batch, "error": repr(exc), **exception_details(exc)}
            errors.append(failure)
            if is_oom_error(exc):
                if progress_enabled:
                    log_event("YOLO OOM during batch fallback; trying next batch.", level="WARNING", run_id=run_id, output_dir=output_dir, extra=failure)
                cleanup_memory()
                continue
            if progress_enabled:
                log_event("YOLO run failed with non-OOM error.", level="ERROR", run_id=run_id, output_dir=output_dir, extra=failure)
            break
    run_dir = ensure_dir(config.output_paths(output_dir)["runs"] / run_id)
    existing_audit = read_json(run_dir / "run_audit.json", default={})
    failed_at = utc_now()
    last_error = errors[-1] if errors else {}
    failed_batch = int(last_error.get("batch") or config.YOLO_BATCH_FALLBACKS[0])
    failed_run_config = make_run_config(model_name, condition, output_dir, smoke_test, batch=failed_batch)
    failed_hash = existing_audit.get("config_hash") or stable_hash(failed_run_config)
    try:
        failed_train_kwargs, failed_augment_audit = yolo_train_kwargs(run_id, condition, output_dir, failed_batch, smoke_test)
        failed_context = {**yolo_audit_context(output_dir, failed_train_kwargs), **failed_augment_audit}
    except Exception as audit_exc:
        failed_context = {"failure_audit_error": repr(audit_exc), **yolo_audit_context(output_dir)}
    write_json(run_dir / "run_audit.json", {**failed_run_config, **failed_context, **existing_audit, "config_hash": failed_hash, "status": "failed", "failed_at": failed_at, "errors": errors})
    write_status(
        run_dir,
        "failed",
        run_id=run_id,
        run_config=failed_run_config,
        failed_at=failed_at,
        error=last_error.get("exception_repr", ""),
        exception_type=last_error.get("exception_type", ""),
        exception_message=last_error.get("exception_message", ""),
        traceback=last_error.get("traceback", ""),
        errors=errors,
    )
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
