"""Loss functions for the controlled shrimp disease experiments."""

from __future__ import annotations

from typing import Any

from . import config

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None


LOSS_CONFIG: dict[str, dict[str, Any]] = {
    "baseline_ce": {"name": "CE", "type": "baseline", "custom": False},
    "asl_single_label": {
        "name": "ASLSingleLabel",
        "type": "existing_asl_baseline",
        "custom": False,
        "gamma_pos": 0.0,
        "gamma_neg": 4.0,
        "eps": 0.1,
    },
    "pairwise_coinfection_ranking_asl": {
        "name": "PairwiseCoInfectionRankingASL",
        "type": "dataset_specific_asl_variant",
        "custom": True,
        "margin": 0.10,
        "lambda_rank": 0.10,
    },
}


def reduce_values(values, reduction: str):
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(reduction)


def _torch_modules():
    if torch is None or nn is None or F is None:
        raise ImportError("torch is required to construct shrimp loss functions")
    return torch, nn, F


_BaseLossModule = nn.Module if nn is not None else object


class ASLSingleLabel(_BaseLossModule):
    """Official-style single-label ASL baseline for logits shaped [B,4]."""

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, eps: float = 0.1, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F = _torch_modules()
        targets = targets.long().view(-1)
        if logits.ndim != 2 or logits.shape[1] != config.NUM_CLASSES:
            raise ValueError(f"expected logits [B,{config.NUM_CLASSES}], got {tuple(logits.shape)}")
        log_probs = _F.log_softmax(logits.float(), dim=1)
        probs = log_probs.exp()
        one_hot = _torch.zeros_like(logits.float()).scatter_(1, targets.unsqueeze(1), 1.0)
        anti_targets = 1.0 - one_hot
        xs_pos = probs * one_hot
        xs_neg = (1.0 - probs) * anti_targets
        gamma = self.gamma_pos * one_hot + self.gamma_neg * anti_targets
        weights = _torch.pow(_torch.clamp(1.0 - xs_pos - xs_neg, min=1e-8), gamma)
        weighted_log_probs = log_probs * weights
        if self.eps > 0:
            one_hot = one_hot * (1.0 - self.eps) + self.eps / logits.size(1)
        loss = -(one_hot * weighted_log_probs).sum(dim=1)
        return reduce_values(loss, self.reduction)


class PairwiseCoInfectionRankingASL(_BaseLossModule):
    """Dataset-specific ASL variant that discourages BG/WSSV versus WSSV_BG confusion."""

    def __init__(self, margin: float = 0.10, lambda_rank: float = 0.10, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction="none")
        self.margin = float(margin)
        self.lambda_rank = float(lambda_rank)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F = _torch_modules()
        targets = targets.long().view(-1)
        if logits.ndim != 2 or logits.shape[1] != config.NUM_CLASSES:
            raise ValueError(f"expected logits [B,{config.NUM_CLASSES}], got {tuple(logits.shape)}")
        probs = _F.softmax(logits.float(), dim=1)
        p_bg = probs[:, 1]
        p_wssv = probs[:, 2]
        p_mix = probs[:, 3]
        penalty = _torch.zeros_like(p_mix)
        bg_mask = targets == 1
        wssv_mask = targets == 2
        mix_mask = targets == 3
        if bg_mask.any():
            penalty[bg_mask] = _F.relu(self.margin + p_mix[bg_mask] - p_bg[bg_mask]).pow(2)
        if wssv_mask.any():
            penalty[wssv_mask] = _F.relu(self.margin + p_mix[wssv_mask] - p_wssv[wssv_mask]).pow(2)
        if mix_mask.any():
            strongest_single = _torch.maximum(p_bg[mix_mask], p_wssv[mix_mask])
            penalty[mix_mask] = 0.5 * _F.relu(self.margin + strongest_single - p_mix[mix_mask]).pow(2)
        loss = self.base(logits.float(), targets) + self.lambda_rank * penalty
        return reduce_values(loss, self.reduction)


def make_loss(loss_key: str):
    _torch, nn, _F = _torch_modules()
    if loss_key == "baseline_ce":
        return nn.CrossEntropyLoss()
    if loss_key == "asl_single_label":
        return ASLSingleLabel()
    if loss_key == "pairwise_coinfection_ranking_asl":
        return PairwiseCoInfectionRankingASL()
    raise KeyError(loss_key)
