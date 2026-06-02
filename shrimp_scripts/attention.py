"""Repo-local attention modules and YOLO classification-head injection helpers."""

from __future__ import annotations

from typing import Any

try:
    import torch as _attention_torch
    import torch.nn as _attention_nn
except Exception:
    _attention_torch = None
    _attention_nn = None


def _torch_stack():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


if _attention_nn is not None:
    class SEModule(_attention_nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            hidden = max(1, int(channels) // 16)
            self.pool = _attention_nn.AdaptiveAvgPool2d(1)
            self.fc = _attention_nn.Sequential(
                _attention_nn.Conv2d(channels, hidden, 1, bias=True),
                _attention_nn.ReLU(inplace=True),
                _attention_nn.Conv2d(hidden, channels, 1, bias=True),
                _attention_nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.fc(self.pool(x))


    class ECAModule(_attention_nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            k_size = 3
            self.pool = _attention_nn.AdaptiveAvgPool2d(1)
            self.conv = _attention_nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            self.sigmoid = _attention_nn.Sigmoid()

        def forward(self, x):
            y = self.pool(x).squeeze(-1).transpose(-1, -2)
            y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
            return x * self.sigmoid(y)


    class SimAM(_attention_nn.Module):
        def __init__(self, channels: int, eps: float = 1e-4) -> None:
            super().__init__()
            self.eps = eps
            self.sigmoid = _attention_nn.Sigmoid()

        def forward(self, x):
            n = max(1, x.shape[2] * x.shape[3] - 1)
            d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
            v = d.sum(dim=(2, 3), keepdim=True) / n
            e = d / (4 * (v + self.eps)) + 0.5
            return x * self.sigmoid(e)


    class CoordAtt(_attention_nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            hidden = max(8, int(channels) // 32)
            self.conv1 = _attention_nn.Conv2d(channels, hidden, 1, bias=False)
            self.bn1 = _attention_nn.BatchNorm2d(hidden)
            self.act = _attention_nn.SiLU(inplace=True)
            self.conv_h = _attention_nn.Conv2d(hidden, channels, 1, bias=True)
            self.conv_w = _attention_nn.Conv2d(hidden, channels, 1, bias=True)

        def forward(self, x):
            _b, _c, h, w = x.shape
            x_h = x.mean(dim=3, keepdim=True)
            x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
            y = _attention_torch.cat([x_h, x_w], dim=2)
            y = self.act(self.bn1(self.conv1(y)))
            y_h, y_w = _attention_torch.split(y, [h, w], dim=2)
            y_w = y_w.permute(0, 1, 3, 2)
            return x * self.conv_h(y_h).sigmoid() * self.conv_w(y_w).sigmoid()


    class CBAM(_attention_nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            hidden = max(1, int(channels) // 16)
            self.mlp = _attention_nn.Sequential(
                _attention_nn.Conv2d(channels, hidden, 1, bias=False),
                _attention_nn.ReLU(inplace=True),
                _attention_nn.Conv2d(hidden, channels, 1, bias=False),
            )
            self.spatial = _attention_nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        def forward(self, x):
            avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
            mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
            x = x * (avg + mx).sigmoid()
            spatial = self.spatial(_attention_torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)).sigmoid()
            return x * spatial


    class EMA(_attention_nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            groups = max(1, min(32, int(channels) // 32))
            while int(channels) % groups != 0 and groups > 1:
                groups -= 1
            self.groups = groups
            group_channels = int(channels) // groups
            self.conv = _attention_nn.Conv2d(group_channels, group_channels, 1, bias=True)
            self.gn = _attention_nn.GroupNorm(1, group_channels)

        def forward(self, x):
            b, c, h, w = x.shape
            gx = x.reshape(b * self.groups, c // self.groups, h, w)
            weight = self.conv(gx.mean(dim=(2, 3), keepdim=True)).sigmoid()
            gx = self.gn(gx * weight)
            return gx.reshape(b, c, h, w)


    class TripletBasicConv(_attention_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = _attention_nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        def forward(self, x):
            z = _attention_torch.cat([x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)], dim=1)
            return x * self.conv(z).sigmoid()


    class TripletAttention(_attention_nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.cw = TripletBasicConv()
            self.hc = TripletBasicConv()
            self.hw = TripletBasicConv()

        def forward(self, x):
            x_perm1 = x.permute(0, 2, 1, 3).contiguous()
            out1 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()
            x_perm2 = x.permute(0, 3, 2, 1).contiguous()
            out2 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()
            out3 = self.hw(x)
            return (out1 + out2 + out3) / 3.0


    class AttentionBeforeClassify(_attention_nn.Module):
        def __init__(self, head=None, attention=None) -> None:
            super().__init__()
            self.attention = attention
            self.head = head

        def forward(self, x):
            features = tensor_from_yolo_feature_input(x)
            return self.head(self.attention(features))
else:
    class AttentionBeforeClassify:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to construct AttentionBeforeClassify")


def normalize_attention_key(attention_key: str | None) -> str:
    return str(attention_key or "none_baseline").strip().lower()


def attention_is_baseline(attention_key: str | None) -> bool:
    return normalize_attention_key(attention_key) in {"", "none", "none_baseline", "baseline"}


ATTENTION_IMPLEMENTATION_METADATA: dict[str, dict[str, Any]] = {
    "none_baseline": {
        "implementation_source": "no_attention_baseline",
        "implementation_variant": "no_attention_baseline",
        "is_exact_official_implementation": False,
    },
    "eca": {
        "implementation_source": "shrimp_scripts.attention.ECAModule",
        "implementation_variant": "repo_local_simplified_eca_like_screening_variant",
        "is_exact_official_implementation": False,
    },
    "simam": {
        "implementation_source": "shrimp_scripts.attention.SimAM",
        "implementation_variant": "repo_local_simplified_simam_like_screening_variant",
        "is_exact_official_implementation": False,
    },
    "coordatt": {
        "implementation_source": "shrimp_scripts.attention.CoordAtt",
        "implementation_variant": "repo_local_simplified_coordatt_like_screening_variant",
        "is_exact_official_implementation": False,
    },
    "cbam": {
        "implementation_source": "shrimp_scripts.attention.CBAM",
        "implementation_variant": "repo_local_simplified_cbam_like_screening_variant",
        "is_exact_official_implementation": False,
    },
    "se": {
        "implementation_source": "shrimp_scripts.attention.SEModule",
        "implementation_variant": "repo_local_simplified_se_like_screening_variant",
        "is_exact_official_implementation": False,
    },
    "ema": {
        "implementation_source": "shrimp_scripts.attention.EMA",
        "implementation_variant": "repo_local_simplified_ema_like_screening_variant",
        "is_exact_official_implementation": False,
    },
    "triplet": {
        "implementation_source": "shrimp_scripts.attention.TripletAttention",
        "implementation_variant": "repo_local_simplified_triplet_attention_like_screening_variant",
        "is_exact_official_implementation": False,
    },
    "c2psa_or_psa": {
        "implementation_source": "installed_ultralytics.nn.modules.block.C2PSA_or_PSA",
        "implementation_variant": "installed_ultralytics_c2psa_or_psa_module",
        "is_exact_official_implementation": True,
    },
}


def attention_metadata(attention_key: str | None) -> dict[str, Any]:
    key = normalize_attention_key(attention_key)
    return dict(ATTENTION_IMPLEMENTATION_METADATA.get(key, {
        "implementation_source": "unknown",
        "implementation_variant": "unknown",
        "is_exact_official_implementation": False,
    }))


def module_parameter_count(module: Any) -> int:
    try:
        return int(sum(parameter.numel() for parameter in module.parameters()))
    except Exception:
        return 0


def tensor_from_yolo_feature_input(x):
    torch, _nn, _F = _torch_stack()
    if isinstance(x, (list, tuple)):
        tensors = [item for item in x if torch.is_tensor(item)]
        if not tensors:
            raise TypeError("attention_wrapper_received_no_tensor_features")
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=1)
    return x


def make_se(channels: int):
    _torch_stack()
    return SEModule(channels)


def make_eca(channels: int):
    _torch_stack()
    return ECAModule(channels)


def make_simam(channels: int):
    _torch_stack()
    return SimAM(channels)


def make_coordatt(channels: int):
    _torch_stack()
    return CoordAtt(channels)


def make_cbam(channels: int):
    _torch_stack()
    return CBAM(channels)


def make_ema(channels: int):
    _torch_stack()
    return EMA(channels)


def make_triplet(channels: int):
    _torch_stack()
    return TripletAttention(channels)


def make_ultralytics_psa(channels: int):
    _torch, nn, _F = _torch_stack()
    try:
        from ultralytics.nn.modules.block import C2PSA
        return C2PSA(channels, channels, n=1)
    except Exception as c2psa_exc:
        try:
            from ultralytics.nn.modules.block import PSA
            return PSA(channels, channels)
        except Exception as psa_exc:
            raise RuntimeError(f"c2psa_or_psa_unavailable:c2psa={c2psa_exc!r};psa={psa_exc!r}") from psa_exc


def make_attention(attention_key: str, channels: int):
    key = normalize_attention_key(attention_key)
    if key == "eca":
        return make_eca(channels)
    if key == "simam":
        return make_simam(channels)
    if key == "coordatt":
        return make_coordatt(channels)
    if key == "cbam":
        return make_cbam(channels)
    if key == "se":
        return make_se(channels)
    if key == "ema":
        return make_ema(channels)
    if key == "triplet":
        return make_triplet(channels)
    if key == "c2psa_or_psa":
        return make_ultralytics_psa(channels)
    raise ValueError(f"unknown_attention_key:{attention_key}")


def attention_keys() -> list[str]:
    return ["none_baseline", "eca", "simam", "coordatt", "cbam", "se", "ema", "triplet", "c2psa_or_psa"]


def checkpoint_safe_attention_classes() -> list[type]:
    names = [
        "AttentionBeforeClassify",
        "SEModule",
        "ECAModule",
        "SimAM",
        "CoordAtt",
        "CBAM",
        "EMA",
        "TripletBasicConv",
        "TripletAttention",
    ]
    return [globals()[name] for name in names if isinstance(globals().get(name), type)]


def find_classification_head(model) -> tuple[Any, int, Any]:
    container = getattr(model, "model", None)
    if container is None:
        raise RuntimeError("model_has_no_model_container")
    indexed = list(enumerate(container)) if hasattr(container, "__iter__") else []
    if not indexed:
        raise RuntimeError("model_container_not_iterable")
    for index, module in reversed(indexed):
        if module.__class__.__name__.lower() == "classify":
            return container, index, module
    raise RuntimeError("ultralytics_classify_head_not_found")


def infer_head_input_channels(model, head, img_size: int, device) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    torch, _nn, _F = _torch_stack()
    captured: dict[str, Any] = {}

    def pre_hook(_module, inputs):
        x = tensor_from_yolo_feature_input(inputs[0])
        captured["shape"] = tuple(int(v) for v in x.shape)

    handle = head.register_forward_pre_hook(pre_hook)
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        output = model(dummy)
    handle.remove()
    if was_training:
        model.train()
    if "shape" not in captured:
        raise RuntimeError("classification_head_input_not_captured")
    out_tensor = classification_output_tensor(output)
    return int(captured["shape"][1]), captured["shape"], tuple(int(v) for v in out_tensor.shape)


def classification_output_tensor(output):
    torch, _nn, _F = _torch_stack()
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError("unable_to_extract_classification_output_tensor")


def assert_valid_classification_output(output, num_classes: int) -> tuple[int, ...]:
    torch, _nn, _F = _torch_stack()
    tensor = classification_output_tensor(output)
    if tensor.ndim != 2 or int(tensor.shape[1]) != int(num_classes):
        raise RuntimeError(f"attention_injection_bad_output_shape:{tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError("attention_injection_nonfinite_output")
    return tuple(int(v) for v in tensor.shape)


def wrap_head_with_attention(head, attention):
    return AttentionBeforeClassify(head=head, attention=attention)


def inject_attention_before_classify(model, attention_key: str, img_size: int = 224, num_classes: int = 4) -> dict[str, Any]:
    torch, _nn, _F = _torch_stack()
    key = normalize_attention_key(attention_key)
    audit: dict[str, Any] = {
        "attention_key": key,
        "status": "not_requested",
        "inserted": False,
        **attention_metadata(key),
    }
    if attention_is_baseline(key):
        audit.update({"status": "baseline_no_attention", "inserted": False, "params_added": 0})
        model.attention_module_audit = audit
        return audit
    device = next(model.parameters()).device
    container, index, head = find_classification_head(model)
    channels, head_input_shape, original_output_shape = infer_head_input_channels(model, head, img_size, device)
    attention = make_attention(key, channels).to(device)
    wrapped = wrap_head_with_attention(head, attention).to(device)
    container[index] = wrapped
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
        "inferred_channels": int(channels),
        "head_input_shape": head_input_shape,
        "original_output_shape": original_output_shape,
        "verified_output_shape": output_shape,
        "img_size": int(img_size),
        "num_classes": int(num_classes),
    })
    model.attention_module_audit = audit
    return audit


def module_unit_test() -> dict[str, Any]:
    torch, _nn, _F = _torch_stack()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in attention_keys():
        if attention_is_baseline(key):
            continue
        for channels in [128, 256, 512, 1024]:
            row: dict[str, Any] = {"attention_key": key, "channels": channels, "status": "failed", **attention_metadata(key)}
            try:
                module = make_attention(key, channels).to(device)
                module.eval()
                dtype = torch.float16 if device.type == "cuda" else torch.float32
                x = torch.randn(2, channels, 14, 14, device=device, dtype=dtype)
                with torch.no_grad():
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda", enabled=True):
                            y = module(x)
                    else:
                        y = module(x.float())
                if tuple(y.shape) != tuple(x.shape):
                    raise RuntimeError(f"shape_changed:{tuple(x.shape)}->{tuple(y.shape)}")
                if not bool(torch.isfinite(y.float()).all().item()):
                    raise RuntimeError("nonfinite_output")
                row["status"] = "passed"
                row["shape"] = tuple(int(v) for v in y.shape)
                row["params_added"] = module_parameter_count(module)
                row.update(attention_metadata(key))
            except Exception as exc:
                if key == "c2psa_or_psa" and "c2psa_or_psa_unavailable" in repr(exc):
                    row["status"] = "skipped_not_compatible"
                    row["skip_reason"] = repr(exc)
                    row["params_added"] = 0
                else:
                    row["error"] = repr(exc)
                    errors.append(f"{key}:{channels}:{repr(exc)}")
            rows.append(row)
    return {
        "status": "passed" if not errors else "failed",
        "device": str(device),
        "rows": rows,
        "errors": errors,
    }
