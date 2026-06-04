from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def replace_block(text: str, start_pattern: str, end_pattern: str, replacement: str, label: str) -> str:
    m1 = re.search(start_pattern, text, flags=re.M)
    if not m1:
        raise RuntimeError(f"Could not find start of {label}")
    m2 = re.search(end_pattern, text[m1.start():], flags=re.M)
    if not m2:
        raise RuntimeError(f"Could not find end of {label}")
    start = m1.start()
    end = m1.start() + m2.start()
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def replace_block_start_options(text: str, start_patterns: list[str], end_pattern: str, replacement: str, label: str) -> str:
    last_error = None
    for pattern in start_patterns:
        try:
            return replace_block(text, pattern, end_pattern, replacement, label)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not replace {label}: {last_error}")


ATTENTION_CLASS = r'''
    class AttentionBeforeClassify(_attention_nn.Module):
        """Wrap an Ultralytics Classify head and apply attention before the head.

        The wrapper preserves Ultralytics parsed-model metadata. BaseModel forward/training
        expects attributes such as `.f`, `.i`, `.type`, and `.np` on graph modules.
        """

        def __init__(self, head=None, attention=None) -> None:
            super().__init__()
            if head is None:
                raise ValueError("AttentionBeforeClassify requires an existing Classify head")
            if attention is None:
                raise ValueError("AttentionBeforeClassify requires an attention module")
            self.attention = attention
            self.head = head
            self.f = getattr(head, "f", -1)
            self.i = getattr(head, "i", -1)
            self.type = getattr(head, "type", f"{self.__class__.__module__}.{self.__class__.__name__}")
            self.np = int(sum(p.numel() for p in self.parameters()))

        def forward(self, x):
            features = tensor_from_yolo_feature_input(x)
            attended = self.attention(features)
            return self.head(attended)
'''

HELPER_FUNCS = r'''
def _first_parameter_device_dtype(model):
    torch, _nn, _F = _torch_stack()
    for p in model.parameters():
        return p.device, p.dtype
    return torch.device("cpu"), torch.float32


def reshape_classification_outputs_if_needed(model, num_classes: int) -> dict[str, Any]:
    """Force an Ultralytics classification model/head to output `num_classes` classes.

    Raw `yolo26m-cls.pt` starts with an ImageNet 1000-class head. The shrimp task
    needs 4 classes before attention injection self-tests are meaningful.
    """
    torch, nn, _F = _torch_stack()
    target_device, target_dtype = _first_parameter_device_dtype(model)
    info: dict[str, Any] = {
        "requested_num_classes": int(num_classes),
        "target_device": str(target_device),
        "target_dtype": str(target_dtype),
        "status": "not_attempted",
    }

    try:
        from ultralytics.nn.tasks import ClassificationModel
        ClassificationModel.reshape_outputs(model, int(num_classes))
        model.to(target_device)
        info["status"] = "classification_model_reshape_outputs_called"
        return info
    except Exception as exc:
        info["classification_model_reshape_error"] = repr(exc)

    try:
        _container, _index, head = find_classification_head(model)
    except Exception as exc:
        info["status"] = "failed_find_head"
        info["error"] = repr(exc)
        return info

    try:
        linear = getattr(head, "linear", None)
        if isinstance(linear, nn.Linear) and int(linear.out_features) != int(num_classes):
            old_device = linear.weight.device
            old_dtype = linear.weight.dtype
            new_linear = nn.Linear(linear.in_features, int(num_classes), bias=linear.bias is not None)
            new_linear = new_linear.to(old_device, dtype=old_dtype)
            head.linear = new_linear
            model.to(target_device)
            info["status"] = "manual_head_linear_replaced"
            info["linear_in_features"] = int(linear.in_features)
            return info
    except Exception as exc:
        info["manual_linear_replace_error"] = repr(exc)

    # Some Classify heads use `.linear` only, but keep this explicit for diagnostics.
    model.to(target_device)
    info["status"] = "no_supported_manual_head_replacement"
    return info


def copy_ultralytics_layer_metadata(dst, src) -> None:
    """Copy metadata fields used by Ultralytics model graph traversal."""
    for attr in ["f", "i", "type"]:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
        elif attr in {"f", "i"}:
            setattr(dst, attr, -1)
        else:
            setattr(dst, attr, f"{dst.__class__.__module__}.{dst.__class__.__name__}")
    try:
        dst.np = int(sum(p.numel() for p in dst.parameters()))
    except Exception:
        dst.np = int(getattr(src, "np", 0) or 0)


def wrap_head_with_attention(head, attention):
    wrapped = AttentionBeforeClassify(head=head, attention=attention)
    copy_ultralytics_layer_metadata(wrapped, head)
    return wrapped
'''

INJECT_FUNC = r'''
def inject_attention_before_classify(model, attention_key: str, img_size: int = 224, num_classes: int = 4) -> dict[str, Any]:
    torch, _nn, _F = _torch_stack()
    key = normalize_attention_key(attention_key)
    original_device, _original_dtype = _first_parameter_device_dtype(model)

    reshape_audit = reshape_classification_outputs_if_needed(model, int(num_classes))
    model.to(original_device)
    device, _dtype = _first_parameter_device_dtype(model)

    audit: dict[str, Any] = {
        "attention_key": key,
        "requested_attention_key": key,
        "status": "not_requested",
        "inserted": False,
        "classification_head_reshape_audit": reshape_audit,
        **attention_metadata(key),
    }

    if attention_is_baseline(key):
        was_training = bool(model.training)
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=device)
            output_shape = assert_valid_classification_output(model(dummy), num_classes)
        if was_training:
            model.train()
        audit.update({
            "status": "baseline_no_attention",
            "inserted": False,
            "params_added": 0,
            "verified_output_shape": output_shape,
            "img_size": int(img_size),
            "num_classes": int(num_classes),
            "device": str(device),
        })
        model.attention_module_audit = audit
        return audit

    container, index, head = find_classification_head(model)
    channels, head_input_shape, original_output_shape = infer_head_input_channels(model, head, img_size, device)
    attention = make_attention(key, channels).to(device)
    wrapped = wrap_head_with_attention(head, attention).to(device)
    copy_ultralytics_layer_metadata(wrapped, head)
    container[index] = wrapped
    model.to(device)

    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        output_shape = assert_valid_classification_output(model(dummy), num_classes)
    if was_training:
        model.train()

    audit.update({
        "status": "inserted",
        "inserted": True,
        "target_index": int(index),
        "target_module_class": head.__class__.__name__,
        "wrapper_class": wrapped.__class__.__name__,
        "attention_module_class": attention.__class__.__name__,
        "params_added": module_parameter_count(attention),
        "wrapper_total_params": module_parameter_count(wrapped),
        "inferred_channels": int(channels),
        "head_input_shape": head_input_shape,
        "original_output_shape": original_output_shape,
        "verified_output_shape": output_shape,
        "img_size": int(img_size),
        "num_classes": int(num_classes),
        "device": str(device),
        "preserved_ultralytics_attrs": {
            "f": getattr(wrapped, "f", None),
            "i": getattr(wrapped, "i", None),
            "type": getattr(wrapped, "type", None),
            "np": getattr(wrapped, "np", None),
        },
    })
    model.attention_module_audit = audit
    return audit
'''


def patch_attention_py(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "class AttentionBeforeClassify" not in text:
        raise RuntimeError("attention.py does not contain AttentionBeforeClassify")

    text = replace_block(
        text,
        r"^    class AttentionBeforeClassify\(_attention_nn\.Module\):\n",
        r"^else:\n",
        ATTENTION_CLASS,
        "AttentionBeforeClassify class",
    )

    text = replace_block_start_options(
        text,
        [
            r"^def reshape_classification_outputs_if_needed\(model, num_classes: int\).*?\n",
            r"^def wrap_head_with_attention\(head, attention\):\n",
        ],
        r"^def inject_attention_before_classify\(",
        HELPER_FUNCS,
        "reshape/wrap helper functions",
    )

    text = replace_block(
        text,
        r"^def inject_attention_before_classify\(model, attention_key: str, img_size: int = 224, num_classes: int = 4\) -> dict\[str, Any\]:\n",
        r"^def module_unit_test\(\) -> dict\[str, Any\]:\n",
        INJECT_FUNC,
        "inject_attention_before_classify",
    )

    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default="/home/drnguyenvinh/notebooks/Improving Lightweight Shrimp Disease Classification with Co-Infection-Aware Losses and RandAugment")
    parser.add_argument("--run_compileall", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    attention_py = project_dir / "shrimp_scripts" / "attention.py"
    if not attention_py.exists():
        raise FileNotFoundError(f"Missing attention.py: {attention_py}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = attention_py.with_suffix(attention_py.suffix + f".bak_attention_injection_v2_{stamp}")
    shutil.copy2(attention_py, backup)
    print(f"Backup created: {backup}")

    patch_attention_py(attention_py)
    print(f"Patched: {attention_py}")

    if args.run_compileall:
        cmd = [sys.executable, "-m", "compileall", "shrimp_scripts", "experiments"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=project_dir, check=True)

    print("Patch v2 complete. Run attention self-test before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
