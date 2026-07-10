"""Shared lightweight attention modules for YOLO11n-seg experiments.

All public modules in this file accept a single tensor shaped [B, C, H, W]
and return the same shape. Multi-input ideas from the research proposal are
implemented here as single-input safe variants first, so Ultralytics YAML
parsing and shape checks stay robust.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    return kernel_size if kernel_size % 2 else kernel_size + 1


def _safe_hidden(channels: int, reduction: int, minimum: int = 8) -> int:
    return max(minimum, int(channels) // max(1, int(reduction)))


def _safe_groups(channels: int, preferred_groups: int) -> int:
    preferred_groups = max(1, int(preferred_groups))
    channels = int(channels)
    if channels % preferred_groups == 0:
        return preferred_groups
    group = math.gcd(channels, preferred_groups)
    return max(1, group)


class ECAChannelGate(nn.Module):
    """Efficient channel attention gate returning [B, C, 1, 1]."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(1, 2).contiguous()
        y = self.conv(y).transpose(1, 2).unsqueeze(-1).contiguous()
        return torch.sigmoid(y)


class SimAMGate(nn.Module):
    """Parameter-free neuron saliency gate returning [B, C, H, W]."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        if n <= 0:
            return torch.ones_like(x)
        centered = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        denom = 4.0 * (centered.sum(dim=(2, 3), keepdim=True) / n + self.eps)
        return torch.sigmoid(centered / denom + 0.5)


class CoordinateAttentionLite(nn.Module):
    """Coordinate attention gate preserving height/width location cues."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = _safe_hidden(channels, reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2).contiguous()
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2).contiguous()
        return torch.sigmoid(self.conv_h(y_h)) * torch.sigmoid(self.conv_w(y_w))


class SpatialDescriptorGate(nn.Module):
    """Small CBAM-style spatial descriptor gate returning [B, 1, H, W]."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        maxv = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((avg, maxv), dim=1)))


class SpatialGroupEnhanceGate(nn.Module):
    """Spatial group-wise enhance gate with safe group fallback."""

    def __init__(self, channels: int, groups: int = 8, eps: float = 1e-5):
        super().__init__()
        self.groups = _safe_groups(channels, groups)
        self.channels_per_group = channels // self.groups
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(1, self.groups, 1, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, self.groups, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        b, c, h, w = x.shape
        grouped = x.view(b, self.groups, self.channels_per_group, h, w)
        context = grouped.mean(dim=(3, 4), keepdim=True)
        score = (grouped * context).sum(dim=2, keepdim=True)
        score = score - score.mean(dim=(3, 4), keepdim=True)
        score = score / (score.std(dim=(3, 4), keepdim=True, unbiased=False) + self.eps)
        gate = torch.sigmoid(score * self.weight + self.bias)
        return gate.expand_as(grouped).reshape(b, c, h, w)


class CESLite(nn.Module):
    """Coordinate-ECA-SimAM Lite attention with residual scaling."""

    def __init__(
        self,
        channels: int,
        reduction: int = 32,
        eca_kernel: int = 3,
        simam_eps: float = 1e-4,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        self.coord = CoordinateAttentionLite(channels, reduction)
        self.eca = ECAChannelGate(channels, eca_kernel)
        self.simam = SimAMGate(simam_eps)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.coord(x) * self.eca(x) * self.simam(x)
        return x + self.gamma * x * (gate - 0.5)


class LowFPResidualSpatialGate(nn.Module):
    """Centered residual spatial gate for low false-positive experiments."""

    def __init__(self, channels: int, kernel_size: int = 3, gamma_init: float = 0.0):
        super().__init__()
        self.spatial = SpatialDescriptorGate(kernel_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.spatial(x)
        gate = gate - gate.mean(dim=(2, 3), keepdim=True)
        return x + self.gamma * x * gate


class P3P4SemanticAttentionGate(nn.Module):
    """Single-input safe semantic-style gate for P3/P4 features.

    The proposal describes a semantic-guided two-input gate. This first-pass
    implementation uses pooled self-context plus a spatial descriptor so it can
    be inserted safely with standard Ultralytics YAML.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 3, gamma_init: float = 0.0):
        super().__init__()
        hidden = _safe_hidden(channels, reduction)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial = SpatialDescriptorGate(kernel_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.context(x) * self.spatial(x)
        return x + self.gamma * x * (gate - 0.5)


class BoundaryAwareLiteAttention(nn.Module):
    """Light boundary/detail attention using depthwise local contrast."""

    def __init__(self, channels: int, kernel_size: int = 3, groups: int = 8, gamma_init: float = 0.0):
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        groups = _safe_groups(channels, groups)
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contrast = self.dw(x) - self.pool(x)
        gate = torch.sigmoid(self.mix(contrast))
        return x + self.gamma * x * (gate - 0.5)


class SGEECAHeadGate(nn.Module):
    """Spatial Group Enhance + ECA gate for pre-Segment features."""

    def __init__(self, channels: int, groups: int = 8, eca_kernel: int = 3, gamma_init: float = 0.0):
        super().__init__()
        self.sge = SpatialGroupEnhanceGate(channels, groups)
        self.eca = ECAChannelGate(channels, eca_kernel)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.sge(x) * self.eca(x)
        return x + self.gamma * x * (gate - 0.5)


class ContextSuppressionGateLite(nn.Module):
    """Light single-input context suppression gate.

    The proposed P5-guided version is multi-input. This safe variant derives a
    global channel context from the same feature map and can be inserted with a
    standard single-input YAML layer.
    """

    def __init__(self, channels: int, reduction: int = 16, gamma_init: float = 0.0):
        super().__init__()
        hidden = _safe_hidden(channels, reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        return x + self.gamma * x * (gate - 0.5)


class CASpatialLowFPGate(nn.Module):
    """Coordinate attention plus centered low-FP spatial gate."""

    def __init__(self, channels: int, reduction: int = 32, kernel_size: int = 3, gamma_init: float = 0.0):
        super().__init__()
        self.coord = CoordinateAttentionLite(channels, reduction)
        self.spatial = SpatialDescriptorGate(kernel_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x)
        spatial = spatial - spatial.mean(dim=(2, 3), keepdim=True)
        gate = self.coord(x) * spatial
        return x + self.gamma * x * gate


class PrototypeAwareMaskGateLite(nn.Module):
    """P3-oriented mask/prototype-aware gate used before Segment input."""

    def __init__(self, channels: int, kernel_size: int = 3, eca_kernel: int = 3, gamma_init: float = 0.0):
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        self.mask_hint = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=False,
        )
        self.eca = ECAChannelGate(channels, eca_kernel)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.mask_hint(x)) * self.eca(x)
        return x + self.gamma * x * (gate - 0.5)


class TripletAttention(nn.Module):
    """Triplet attention ablation module kept for compatibility."""

    def __init__(self, channels: int | None = None, kernel_size: int = 7, gamma_init: float = 0.0):
        super().__init__()
        self.cw_gate = SpatialDescriptorGate(kernel_size)
        self.hc_gate = SpatialDescriptorGate(kernel_size)
        self.hw_gate = SpatialDescriptorGate(kernel_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cw = x.permute(0, 2, 1, 3).contiguous()
        y_cw = (x_cw * self.cw_gate(x_cw)).permute(0, 2, 1, 3).contiguous()
        x_hc = x.permute(0, 3, 2, 1).contiguous()
        y_hc = (x_hc * self.hc_gate(x_hc)).permute(0, 3, 2, 1).contiguous()
        y_hw = x * self.hw_gate(x)
        return x + self.gamma * ((y_cw + y_hc + y_hw) / 3.0 - x)


class NAMAttention(nn.Module):
    """Normalization-based attention ablation module."""

    def __init__(self, channels: int, gamma_init: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(x)
        weight = self.bn.weight.abs()
        weight = weight / (weight.sum() + self.eps)
        gate = torch.sigmoid(y * weight.view(1, -1, 1, 1))
        return x + self.gamma * x * (gate - 0.5)


CUSTOM_ATTENTION_MODULES: tuple[type[nn.Module], ...] = (
    SGEECAHeadGate,
    CESLite,
    LowFPResidualSpatialGate,
    P3P4SemanticAttentionGate,
    BoundaryAwareLiteAttention,
    ContextSuppressionGateLite,
    CASpatialLowFPGate,
    PrototypeAwareMaskGateLite,
    TripletAttention,
    NAMAttention,
)


def module_name_list(modules: Iterable[type[nn.Module]] = CUSTOM_ATTENTION_MODULES) -> list[str]:
    return [module.__name__ for module in modules]


__all__ = [
    "BoundaryAwareLiteAttention",
    "CASpatialLowFPGate",
    "CESLite",
    "ContextSuppressionGateLite",
    "CUSTOM_ATTENTION_MODULES",
    "ECAChannelGate",
    "LowFPResidualSpatialGate",
    "NAMAttention",
    "P3P4SemanticAttentionGate",
    "PrototypeAwareMaskGateLite",
    "SGEECAHeadGate",
    "SpatialDescriptorGate",
    "SpatialGroupEnhanceGate",
    "TripletAttention",
    "module_name_list",
]
