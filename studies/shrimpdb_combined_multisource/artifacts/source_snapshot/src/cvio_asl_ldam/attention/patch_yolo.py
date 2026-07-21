"""Ultralytics classification trainer hooks for custom loss and attention."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cvio_asl_ldam.attention.simam_dcfr import SimAMDCFR
from cvio_asl_ldam.losses.asl_ldam import ASLLDAMLoss

try:
    import torch
    import torch.nn as nn
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.nn.tasks import ClassificationModel
    try:
        from ultralytics.models.yolo.classify.train import ClassificationTrainer
    except ImportError:
        from ultralytics.models.yolo.classify import ClassificationTrainer

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:
    torch = None
    nn = None
    ClassificationDataset = object
    ClassificationModel = object
    ClassificationTrainer = object
    _IMPORT_ERROR = exc


CLASS_NAMES = ("Healthy", "BG", "WSSV", "WSSV_BG")
CLASS_COUNTS = (282, 139, 229, 154)


def normalize_class_name(folder_name: str) -> str:
    candidates = [folder_name]
    if "_" in folder_name:
        candidates.append(folder_name.split("_", 1)[1])
    if ". " in folder_name:
        candidates.append(folder_name.split(". ", 1)[1])
    for candidate in candidates:
        if candidate in CLASS_NAMES:
            return candidate
    for class_name in CLASS_NAMES:
        if folder_name.endswith(class_name):
            return class_name
    raise ValueError(f"Unknown class folder: {folder_name}")


def _tensor_from_features(value: Any):
    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if torch.is_tensor(item)]
        if not tensors:
            raise TypeError("No tensor features were passed to the classification head")
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)
    return value


if nn is not None:

    class AttentionBeforeClassify(nn.Module):
        def __init__(self, head: nn.Module, attention: nn.Module) -> None:
            super().__init__()
            self.attention = attention
            self.head = head
            for attribute, fallback in (
                ("f", -1),
                ("i", -1),
                ("type", f"{self.__class__.__module__}.{self.__class__.__name__}"),
            ):
                setattr(self, attribute, getattr(head, attribute, fallback))
            self.np = int(sum(parameter.numel() for parameter in self.parameters()))

        def forward(self, value):
            return self.head(self.attention(_tensor_from_features(value)))


    class PaperClassificationLoss(nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.loss_key = os.environ.get("CVIO_YOLO_LOSS", "baseline_ce")
            self.loss_fcn = (
                ASLLDAMLoss(class_counts=CLASS_COUNTS)
                if self.loss_key == "asl_ldam"
                else nn.CrossEntropyLoss()
            )

        @staticmethod
        def _extract_logits(predictions, targets):
            if torch.is_tensor(predictions):
                return predictions
            if isinstance(predictions, (list, tuple)):
                for item in predictions:
                    if (
                        torch.is_tensor(item)
                        and item.ndim == 2
                        and item.shape[0] == targets.shape[0]
                        and item.shape[1] == len(CLASS_NAMES)
                    ):
                        return item
                for item in predictions:
                    if torch.is_tensor(item) and item.ndim == 2:
                        return item
            raise TypeError("Unable to extract YOLO classification logits")

        def forward(self, predictions, batch):
            targets = batch["cls"].long().view(-1)
            logits = self._extract_logits(predictions, targets).float()
            targets = targets.to(logits.device)
            self.loss_fcn = self.loss_fcn.to(logits.device)
            loss = self.loss_fcn(logits, targets)
            return loss, loss.detach()


    class PaperClassificationModel(ClassificationModel):
        def init_criterion(self):
            return PaperClassificationLoss(self)


    class PaperClassificationDataset(ClassificationDataset):
        """Force the paper class order independently of folder sorting."""

        def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
            super().__init__(root=root, args=args, augment=augment, prefix=prefix)
            class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
            remapped = []
            for sample in self.samples:
                item = list(sample)
                class_name = normalize_class_name(Path(item[0]).parent.name)
                item[1] = class_to_idx[class_name]
                remapped.append(tuple(item))
            self.samples = remapped
            self.targets = [int(sample[1]) for sample in remapped]
            self.classes = list(CLASS_NAMES)
            self.class_to_idx = class_to_idx


    class PaperClassificationTrainer(ClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            return PaperClassificationDataset(
                root=img_path,
                args=self.args,
                augment=mode == "train",
                prefix=mode,
            )

        def get_model(self, cfg=None, weights=None, verbose=True):
            nc = self.data.get("nc", len(CLASS_NAMES))
            channels = self.data.get("channels", 3)
            try:
                model = PaperClassificationModel(cfg, nc=nc, ch=channels, verbose=verbose)
            except TypeError:
                model = PaperClassificationModel(cfg, nc=nc, verbose=verbose)
            if weights:
                model.load(weights)
            model.names = {index: name for index, name in enumerate(CLASS_NAMES)}
            attention_key = os.environ.get("CVIO_YOLO_ATTENTION", "none")
            if attention_key == "simam_dcfr":
                inject_attention_before_classify(model)
            for parameter in model.parameters():
                parameter.requires_grad = True
            return model

        def set_model_attributes(self):
            try:
                super().set_model_attributes()
            except AttributeError:
                pass
            names = {index: name for index, name in enumerate(CLASS_NAMES)}
            if isinstance(self.data, dict):
                self.data["names"] = names
                self.data["nc"] = len(CLASS_NAMES)
            self.model.names = names

else:
    AttentionBeforeClassify = None
    PaperClassificationLoss = None
    PaperClassificationModel = None
    PaperClassificationDataset = None
    PaperClassificationTrainer = None


def _classification_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
    raise TypeError("Unable to extract classification output")


def _find_classification_head(model) -> tuple[Any, int, Any]:
    container = getattr(model, "model", None)
    if container is None:
        raise RuntimeError("Ultralytics model has no module container")
    for index, module in reversed(list(enumerate(container))):
        if module.__class__.__name__.lower() == "classify":
            return container, index, module
    raise RuntimeError("Ultralytics classification head was not found")


def inject_attention_before_classify(
    model,
    img_size: int = 224,
    num_classes: int = 4,
) -> dict[str, Any]:
    if torch is None or AttentionBeforeClassify is None:
        raise ImportError(f"torch/ultralytics unavailable: {_IMPORT_ERROR!r}")
    try:
        ClassificationModel.reshape_outputs(model, num_classes)
    except Exception:
        pass
    device = next(model.parameters()).device
    container, index, head = _find_classification_head(model)
    captured: dict[str, tuple[int, ...]] = {}

    def hook(_module, inputs):
        captured["shape"] = tuple(_tensor_from_features(inputs[0]).shape)

    handle = head.register_forward_pre_hook(hook)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        original = _classification_output(
            model(torch.zeros(1, 3, img_size, img_size, device=device))
        )
    handle.remove()
    if "shape" not in captured:
        raise RuntimeError("Could not infer classification-head input channels")
    channels = int(captured["shape"][1])
    attention = SimAMDCFR(channels).to(device)
    wrapped = AttentionBeforeClassify(head, attention).to(device)
    container[index] = wrapped
    with torch.no_grad():
        output = _classification_output(
            model(torch.zeros(1, 3, img_size, img_size, device=device))
        )
    if was_training:
        model.train()
    if tuple(output.shape) != (1, num_classes):
        raise RuntimeError(f"Attention injection changed output shape to {tuple(output.shape)}")
    audit = {
        "attention_key": "simam_gated_residual__dcfr_texture",
        "inserted": True,
        "target_index": index,
        "inferred_channels": channels,
        "head_input_shape": captured["shape"],
        "original_output_shape": tuple(original.shape),
        "verified_output_shape": tuple(output.shape),
        "params_added": sum(parameter.numel() for parameter in attention.parameters()),
    }
    model.attention_module_audit = audit
    return audit


def get_custom_trainer_class():
    if PaperClassificationTrainer is None:
        raise ImportError(f"Ultralytics custom trainer unavailable: {_IMPORT_ERROR!r}")
    return PaperClassificationTrainer


def register_checkpoint_safe_globals() -> None:
    if torch is None:
        return
    add_safe_globals = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if callable(add_safe_globals):
        classes = [
            cls
            for cls in (
                AttentionBeforeClassify,
                PaperClassificationLoss,
                PaperClassificationModel,
                PaperClassificationDataset,
                PaperClassificationTrainer,
                ASLLDAMLoss,
                SimAMDCFR,
            )
            if cls is not None
        ]
        add_safe_globals(classes)
