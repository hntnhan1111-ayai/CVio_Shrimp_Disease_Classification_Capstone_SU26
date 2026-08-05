"""Trusted-load architecture audit for CVio Ultralytics classification checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def extract_logits(value, torch):
    if torch.is_tensor(value) and value.ndim == 2:
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
    raise TypeError(f"No 2D classification output in {type(value)!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset-regime", required=True)
    parser.add_argument("--expected-classes", required=True, type=int)
    parser.add_argument("--imgsz", default=224, type=int)
    args = parser.parse_args()

    weights = args.weights.resolve()
    expected_hash = sha256_file(weights)

    import torch
    import ultralytics

    from cvio_asl_ldam.attention.patch_yolo import (
        AttentionBeforeClassify,
        register_checkpoint_safe_globals,
    )
    from cvio_asl_ldam.attention.simam_dcfr import SimAMDCFR

    # These files are trusted CVio project checkpoints whose hashes and source
    # archives were recorded before deserialization.
    register_checkpoint_safe_globals()
    from ultralytics import YOLO

    loaded = YOLO(str(weights))
    model = loaded.model
    model.eval()

    module_rows: list[dict[str, Any]] = []
    classify_modules = []
    simam_modules = []
    wrappers = []
    for name, module in model.named_modules():
        module_type = f"{module.__class__.__module__}.{module.__class__.__name__}"
        parameter_count = sum(
            parameter.numel() for parameter in module.parameters(recurse=False)
        )
        module_rows.append(
            {
                "name": name,
                "type": module_type,
                "direct_parameters": parameter_count,
            }
        )
        if module.__class__.__name__ == "Classify":
            classify_modules.append((name, module))
        if isinstance(module, SimAMDCFR):
            simam_modules.append((name, module))
        if AttentionBeforeClassify is not None and isinstance(
            module, AttentionBeforeClassify
        ):
            wrappers.append((name, module))

    captured: dict[str, list[int]] = {}
    handles = []
    for name, module in classify_modules:

        def hook(_module, inputs, module_name=name):
            value = inputs[0]
            if isinstance(value, (list, tuple)):
                tensors = [item for item in value if torch.is_tensor(item)]
                value = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)
            if torch.is_tensor(value):
                captured[module_name] = list(value.shape)

        handles.append(module.register_forward_pre_hook(hook))

    with torch.no_grad():
        output = extract_logits(model(torch.zeros(1, 3, args.imgsz, args.imgsz)), torch)
    for handle in handles:
        handle.remove()

    simam_structures: list[dict[str, Any]] = []
    for name, module in simam_modules:
        texture_convs = [
            child
            for child in module.texture.modules()
            if isinstance(child, torch.nn.Conv2d)
        ]
        gate_convs = [
            child
            for child in module.gate.modules()
            if isinstance(child, torch.nn.Conv2d)
        ]
        channels = texture_convs[0].in_channels if texture_convs else None
        simam_structures.append(
            {
                "name": name,
                "channels": channels,
                "e_lambda": module.e_lambda,
                "has_global_average_pool": any(
                    isinstance(child, torch.nn.AdaptiveAvgPool2d)
                    for child in module.gate.modules()
                ),
                "gate_pointwise_1x1": any(
                    conv.kernel_size == (1, 1) for conv in gate_convs
                ),
                "texture_depthwise_3x3": any(
                    conv.kernel_size == (3, 3)
                    and conv.groups == conv.in_channels
                    and conv.out_channels == conv.in_channels
                    for conv in texture_convs
                ),
                "texture_pointwise_1x1": any(
                    conv.kernel_size == (1, 1) for conv in texture_convs
                ),
                "has_sigmoid_texture_mask": any(
                    isinstance(child, torch.nn.Sigmoid)
                    for child in module.texture.modules()
                ),
            }
        )

    wrapper_order_valid = all(
        isinstance(wrapper.attention, SimAMDCFR)
        and wrapper.head.__class__.__name__ == "Classify"
        for _, wrapper in wrappers
    )
    intended_path_present = bool(simam_modules and wrappers and wrapper_order_valid)
    names = json_safe(getattr(model, "names", getattr(loaded, "names", None)))
    checkpoint = getattr(loaded, "ckpt", None)
    train_args = (
        checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    )

    report = {
        "candidate": {
            "source": args.source,
            "quarantine_path": str(weights),
            "sha256": expected_hash,
            "size_bytes": weights.stat().st_size,
            "mtime_utc": dt.datetime.fromtimestamp(
                weights.stat().st_mtime, tz=dt.timezone.utc
            ).isoformat(),
            "dataset_regime_claim": args.dataset_regime,
        },
        "trusted_load": {
            "project_safe_globals_registered": True,
            "pickle_source_trusted": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "device": "cpu",
        },
        "checkpoint_metadata": {
            "task": getattr(loaded, "task", None),
            "checkpoint_version": checkpoint.get("version")
            if isinstance(checkpoint, dict)
            else None,
            "checkpoint_date": checkpoint.get("date")
            if isinstance(checkpoint, dict)
            else None,
            "names": names,
            "class_count": len(names) if isinstance(names, dict) else None,
            "train_args": json_safe(train_args),
            "stored_attention_module_audit": json_safe(
                getattr(model, "attention_module_audit", None)
            ),
        },
        "architecture": {
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "module_count": len(module_rows),
            "module_tree": module_rows,
            "classify_modules": [name for name, _ in classify_modules],
            "head_input_shapes": captured,
            "output_shape": list(output.shape),
            "attention_wrappers": [name for name, _ in wrappers],
            "simam_dcfr_modules": simam_structures,
            "wrapper_order_valid": wrapper_order_valid,
            "intended_late_feature_recalibration_path_present": intended_path_present,
            "indistinguishable_from_ce_baseline_architecture": not intended_path_present,
            "simam_dcfr_forward_source_sha256": hashlib.sha256(
                inspect.getsource(SimAMDCFR.forward).encode("utf-8")
            ).hexdigest(),
        },
        "checks": {
            "expected_class_count": args.expected_classes,
            "class_count_matches": isinstance(names, dict)
            and len(names) == args.expected_classes,
            "output_shape_expected": [1, args.expected_classes],
            "output_shape_matches": list(output.shape) == [1, args.expected_classes],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sha256": expected_hash,
                "output": str(args.output),
                "intended_path": intended_path_present,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
