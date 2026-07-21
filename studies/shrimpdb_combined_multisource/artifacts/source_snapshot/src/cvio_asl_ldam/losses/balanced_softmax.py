"""Balanced Softmax reference implementation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: Sequence[int | float]) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if torch.any(counts <= 0):
            raise ValueError("class_counts must be positive")
        self.register_buffer("log_prior", counts.log())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prior = self.log_prior.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits + prior.view(1, -1), targets.long().view(-1))
