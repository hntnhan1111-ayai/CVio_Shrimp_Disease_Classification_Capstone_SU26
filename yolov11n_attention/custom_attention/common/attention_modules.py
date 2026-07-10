"""Lightweight custom attention modules for YOLO11n instance segmentation.

Each module accepts a tensor shaped [B, C, H, W] and returns the same shape.
AttentionGate is the only multi-input module; it accepts [x, gate] and returns
the shape of x.
"""

from __future__ import annotations

import inspect
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAMCore(nn.Module):
    """Parameter-free neuron attention used inside CESA-Lite."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        if n <= 0:
            return x
        x_minus_mu_square = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        denom = 4 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / n + self.eps)
        attention = x_minus_mu_square / denom + 0.5
        return x * torch.sigmoid(attention)


class CoordGate(nn.Module):
    """Coordinate attention gate that preserves spatial coordinates."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return a_h * a_w


class ECAGate(nn.Module):
    """Efficient channel attention gate."""

    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        if k_size % 2 == 0:
            k_size += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return torch.sigmoid(y)


class CESALite(nn.Module):
    """Coordinate + ECA + SimAM attention with residual scaling."""

    def __init__(self, channels: int, reduction: int = 32, eca_k: int = 3, gamma_init: float = 0.1):
        super().__init__()
        self.coord = CoordGate(channels, reduction)
        self.eca = ECAGate(channels, eca_k)
        self.simam = SimAMCore()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.coord(x) * self.eca(x)
        y = self.simam(y)
        return x + self.gamma * y


class AttentionGate(nn.Module):
    """U-Net style attention gate for [shallow_feature, semantic_gate]."""

    def __init__(self, x_channels: int, gate_channels: int, inter_ratio: int = 2, gamma_init: float = 0.1):
        super().__init__()
        inter_channels = max(8, x_channels // max(1, inter_ratio))
        self.theta_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise TypeError("AttentionGate expects [x, gate] input.")
        x, gate = inputs
        if gate.shape[-2:] != x.shape[-2:]:
            gate = F.interpolate(gate, size=x.shape[-2:], mode="nearest")
        alpha = torch.sigmoid(self.psi(self.act(self.theta_x(x) + self.phi_g(gate))))
        return x + self.gamma * x * alpha


class SpatialGate(nn.Module):
    """Small spatial gate used by TripletAttention and CBAM-Lite."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class TripletAttention(nn.Module):
    """Triplet attention over C-H, C-W, and H-W views with residual scaling."""

    def __init__(self, channels: int | None = None, kernel_size: int = 7, gamma_init: float = 0.1):
        super().__init__()
        self.cw_gate = SpatialGate(kernel_size)
        self.hc_gate = SpatialGate(kernel_size)
        self.hw_gate = SpatialGate(kernel_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = (x_perm1 * self.cw_gate(x_perm1)).permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = (x_perm2 * self.hc_gate(x_perm2)).permute(0, 3, 2, 1).contiguous()

        x_out3 = x * self.hw_gate(x)
        y = (x_out1 + x_out2 + x_out3) / 3.0
        return x + self.gamma * y


class CALiteSpatialGate(nn.Module):
    """Coordinate attention plus a lightweight spatial gate."""

    def __init__(self, channels: int, reduction: int = 32, kernel_size: int = 7, gamma_init: float = 0.1):
        super().__init__()
        self.coord = CoordGate(channels, reduction)
        self.spatial = SpatialGate(kernel_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.coord(x) * self.spatial(x)
        return x + self.gamma * x * gate


class NAMAttention(nn.Module):
    """Normalization-based attention using BatchNorm scale information."""

    def __init__(self, channels: int, gamma_init: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(x)
        weight = self.bn.weight.abs()
        weight = weight / (weight.sum() + self.eps)
        gate = torch.sigmoid(y * weight.view(1, -1, 1, 1))
        return x + self.gamma * x * gate


class LowFPCBAMLite(nn.Module):
    """Spatial-only CBAM-lite gate for low false-positive experiments."""

    def __init__(self, channels: int, kernel_size: int = 7, gamma_init: float = 0.1):
        super().__init__()
        self.spatial = SpatialGate(kernel_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.spatial(x)
        return x + self.gamma * x * gate


CUSTOM_ATTENTION_MODULES = (
    CESALite,
    TripletAttention,
    CALiteSpatialGate,
    NAMAttention,
    LowFPCBAMLite,
)


def _module_names(modules: Iterable[type[nn.Module]]) -> str:
    return ", ".join(module.__name__ for module in modules)


def register_custom_attention_modules():
    """Register modules and patch Ultralytics parse_model in memory."""

    import ultralytics.nn.tasks as tasks

    for module in (*CUSTOM_ATTENTION_MODULES, AttentionGate):
        setattr(tasks, module.__name__, module)

    if getattr(tasks, "_custom_attention_parse_patched", False):
        return

    source = inspect.getsource(tasks.parse_model)
    needle = "        elif m in frozenset(\n            {\n                Detect,"
    insert = """        elif m in frozenset({CESALite, TripletAttention, CALiteSpatialGate, NAMAttention, LowFPCBAMLite}):
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m is AttentionGate:
            c1 = ch[f[0]]
            cg = ch[f[1]]
            c2 = c1
            args = [c1, cg, *args]
"""
    if needle not in source:
        raise RuntimeError(
            "Could not patch ultralytics.nn.tasks.parse_model: Detect branch marker not found. "
            f"Custom modules: {_module_names((*CUSTOM_ATTENTION_MODULES, AttentionGate))}"
        )

    patched = source.replace(needle, insert + needle, 1)
    exec(compile(patched, "<custom_attention_parse_model>", "exec"), tasks.__dict__)
    tasks._custom_attention_parse_patched = True


__all__ = [
    "AttentionGate",
    "CALiteSpatialGate",
    "CESALite",
    "LowFPCBAMLite",
    "NAMAttention",
    "TripletAttention",
    "register_custom_attention_modules",
]
