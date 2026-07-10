"""Lightweight research attention modules for YOLO11n-seg.

Every public module accepts a tensor shaped [B, C, H, W] and returns the same
shape. The modules are intentionally single-input blocks so they can be placed
before the Ultralytics Segment head without changing P3/P4/P5 channels.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


def _resolve_channels(c1: int | None = None, channels: int | None = None) -> int:
    value = channels if channels is not None else c1
    if value is None:
        raise ValueError("A channel count must be provided via c1 or channels.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"channels must be positive, got {value}")
    return value


def _odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    return kernel_size if kernel_size % 2 else kernel_size + 1


def _safe_hidden(channels: int, reduction: int, minimum: int = 8) -> int:
    return max(1, min(int(channels), max(int(minimum), int(channels) // max(1, int(reduction)))))


def _safe_groups(channels: int, preferred_groups: int) -> int:
    channels = int(channels)
    preferred_groups = max(1, int(preferred_groups))
    if channels % preferred_groups == 0:
        return preferred_groups
    return max(1, math.gcd(channels, preferred_groups))


class ECAGate(nn.Module):
    """Efficient Channel Attention gate returning [B, C, 1, 1]."""

    def __init__(self, c1: int | None = None, k_size: int = 3, channels: int | None = None):
        super().__init__()
        channels = _resolve_channels(c1, channels)
        k_size = _odd_kernel(k_size)
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(1, 2).contiguous()  # [B, 1, C]
        y = self.conv(y)
        y = y.transpose(1, 2).unsqueeze(-1).contiguous()  # [B, C, 1, 1]
        return torch.sigmoid(y)


class _SpatialDescriptorGate(nn.Module):
    """Small spatial descriptor used by Triplet-Lite axis branches."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        maxv = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((avg, maxv), dim=1)))


class TripletLiteGate(nn.Module):
    """Safe Triplet-style gate returning [B, C, H, W].

    This is a Triplet-Lite implementation: one depthwise-pointwise HW branch
    plus two descriptor branches on channel-width and height-channel views.
    The permuted tensors are made contiguous before convolution-like ops.
    """

    def __init__(self, c1: int | None = None, k: int = 5, channels: int | None = None):
        super().__init__()
        channels = _resolve_channels(c1, channels)
        k = _odd_kernel(k)
        self.hw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=k, padding=(k - 1) // 2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.cw = _SpatialDescriptorGate(kernel_size=k)
        self.hc = _SpatialDescriptorGate(kernel_size=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gate_hw = self.hw(x)  # [B, C, H, W]

        x_cw = x.permute(0, 2, 1, 3).contiguous()  # [B, H, C, W]
        gate_cw = self.cw(x_cw).permute(0, 2, 1, 3).contiguous()  # [B, C, 1, W]
        gate_cw = gate_cw.expand(b, c, h, w)

        x_hc = x.permute(0, 3, 2, 1).contiguous()  # [B, W, H, C]
        gate_hc = self.hc(x_hc).permute(0, 3, 2, 1).contiguous()  # [B, C, H, 1]
        gate_hc = gate_hc.expand(b, c, h, w)

        return (gate_hw + gate_cw + gate_hc) / 3.0


class CoordinateAttentionLite(nn.Module):
    """Coordinate Attention gate preserving height/width position cues."""

    def __init__(self, c1: int | None = None, reduction: int = 32, channels: int | None = None):
        super().__init__()
        channels = _resolve_channels(c1, channels)
        hidden = _safe_hidden(channels, reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)  # [B, C, H, 1]
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2).contiguous()  # [B, C, W, 1]
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2).contiguous()
        return torch.sigmoid(self.conv_h(y_h)) * torch.sigmoid(self.conv_w(y_w))


class SGEGate(nn.Module):
    """Spatial Group Enhance gate returning [B, C, H, W]."""

    def __init__(self, c1: int | None = None, groups: int = 8, eps: float = 1e-5, channels: int | None = None):
        super().__init__()
        channels = _resolve_channels(c1, channels)
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
        var = score.pow(2).mean(dim=(3, 4), keepdim=True)
        score = score / torch.sqrt(var + self.eps)
        gate = torch.sigmoid(score * self.weight + self.bias)
        return gate.expand_as(grouped).reshape(b, c, h, w)


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
    """Parameter-free SimAM neuron saliency gate returning [B, C, H, W]."""

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


class CoTEGate(nn.Module):
    """Consensus-Regularized Triplet-ECA Gate."""

    def __init__(
        self,
        c1: int,
        eca_k: int = 3,
        triplet_k: int = 5,
        alpha_init: float = 0.20,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.triplet = TripletLiteGate(channels, k=triplet_k)
        self.eca = ECAGate(channels, k_size=eca_k)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.lambda_t = nn.Parameter(torch.tensor(1.0))
        self.lambda_c = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        a_t = self.triplet(x)  # [B, C, H, W]
        a_c = self.eca(x).expand_as(x)  # [B, C, H, W]
        disagreement = self.smooth(torch.abs(a_t - a_c))
        gate = torch.sigmoid(self.lambda_t * a_t + self.lambda_c * a_c - self.alpha * disagreement)
        return identity + self.gamma * identity * gate


class SCSGGate(nn.Module):
    """Scale-Aware Coordinate-SGE Gate."""

    def __init__(
        self,
        c1: int,
        reduction: int = 32,
        groups: int = 8,
        beta_init: float = 1.0,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.coordinate = CoordinateAttentionLite(channels, reduction=reduction)
        self.sge = SGEGate(channels, groups=groups)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        a_ca = self.coordinate(x)  # [B, C, H, W]
        a_sge = self.sge(x)  # [B, C, H, W]
        gate = torch.sigmoid(a_ca + self.beta * a_sge)
        return identity + self.gamma * identity * gate


class LPSCGate(nn.Module):
    """Lesion-Preserving NAM-SimAM Contrast Gate."""

    def __init__(
        self,
        c1: int,
        simam_eps: float = 1e-4,
        eta_init: float = 1.0,
        delta_init: float = 0.5,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.nam = NAMGate(channels)
        self.simam = SimAMGate(channels, eps=simam_eps)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        a_n = self.nam(x)  # [B, C, H, W]
        a_s = self.simam(x)  # [B, C, H, W]
        prior = self.dw(x)
        prior = torch.sigmoid(prior - self.smooth(prior))
        gate = torch.sigmoid(a_n + self.eta * a_s + self.delta * prior)
        return identity + self.gamma * identity * gate


RESEARCH_ATTENTION_MODULES: tuple[type[nn.Module], ...] = (
    ECAGate,
    TripletLiteGate,
    CoordinateAttentionLite,
    SGEGate,
    NAMGate,
    SimAMGate,
    CoTEGate,
    SCSGGate,
    LPSCGate,
)


def module_name_list(modules: Iterable[type[nn.Module]] = RESEARCH_ATTENTION_MODULES) -> list[str]:
    return [module.__name__ for module in modules]


__all__ = [
    "CoTEGate",
    "CoordinateAttentionLite",
    "ECAGate",
    "LPSCGate",
    "NAMGate",
    "RESEARCH_ATTENTION_MODULES",
    "SCSGGate",
    "SGEGate",
    "SimAMGate",
    "TripletLiteGate",
    "module_name_list",
]
