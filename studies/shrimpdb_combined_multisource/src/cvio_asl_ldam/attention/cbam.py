"""Convolutional block attention module used by comparison experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        maximum = F.adaptive_max_pool2d(x, 1).view(batch, channels)
        channel_scale = torch.sigmoid(self.mlp(avg) + self.mlp(maximum))
        x = x * channel_scale.view(batch, channels, 1, 1)
        pooled = torch.cat(
            [x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True).values],
            dim=1,
        )
        return x * torch.sigmoid(self.spatial(pooled))
