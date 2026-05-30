"""Loss functions for the controlled shrimp disease experiments."""

from __future__ import annotations

from typing import Any

from . import config


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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


class ASLSingleLabel:
    """Official-style single-label ASL baseline for logits shaped [B,4]."""

    def __new__(cls, gamma_pos: float = 0.0, gamma_neg: float = 4.0, eps: float = 0.1, reduction: str = "mean"):
        torch, nn, _F = _torch_modules()

        class _ASLSingleLabel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gamma_pos = float(gamma_pos)
                self.gamma_neg = float(gamma_neg)
                self.eps = float(eps)
                self.reduction = reduction

            def forward(self, logits, targets):
                targets = targets.long().view(-1)
                if logits.ndim != 2 or logits.shape[1] != config.NUM_CLASSES:
                    raise ValueError(f"expected logits [B,{config.NUM_CLASSES}], got {tuple(logits.shape)}")
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=1)
                probs = log_probs.exp()
                one_hot = torch.zeros_like(logits.float()).scatter_(1, targets.unsqueeze(1), 1.0)
                anti_targets = 1.0 - one_hot
                xs_pos = probs * one_hot
                xs_neg = (1.0 - probs) * anti_targets
                gamma = self.gamma_pos * one_hot + self.gamma_neg * anti_targets
                weights = torch.pow(torch.clamp(1.0 - xs_pos - xs_neg, min=1e-8), gamma)
                weighted_log_probs = log_probs * weights
                if self.eps > 0:
                    one_hot = one_hot * (1.0 - self.eps) + self.eps / logits.size(1)
                loss = -(one_hot * weighted_log_probs).sum(dim=1)
                return reduce_values(loss, self.reduction)

        return _ASLSingleLabel()


class PairwiseCoInfectionRankingASL:
    """Dataset-specific ASL variant that discourages BG/WSSV versus WSSV_BG confusion."""

    def __new__(cls, margin: float = 0.10, lambda_rank: float = 0.10, reduction: str = "mean"):
        torch, nn, F = _torch_modules()

        class _PairwiseCoInfectionRankingASL(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = ASLSingleLabel(reduction="none")
                self.margin = float(margin)
                self.lambda_rank = float(lambda_rank)
                self.reduction = reduction

            def forward(self, logits, targets):
                targets = targets.long().view(-1)
                if logits.ndim != 2 or logits.shape[1] != config.NUM_CLASSES:
                    raise ValueError(f"expected logits [B,{config.NUM_CLASSES}], got {tuple(logits.shape)}")
                probs = F.softmax(logits.float(), dim=1)
                p_bg = probs[:, 1]
                p_wssv = probs[:, 2]
                p_mix = probs[:, 3]
                penalty = torch.zeros_like(p_mix)
                bg_mask = targets == 1
                wssv_mask = targets == 2
                mix_mask = targets == 3
                if bg_mask.any():
                    penalty[bg_mask] = F.relu(self.margin + p_mix[bg_mask] - p_bg[bg_mask]).pow(2)
                if wssv_mask.any():
                    penalty[wssv_mask] = F.relu(self.margin + p_mix[wssv_mask] - p_wssv[wssv_mask]).pow(2)
                if mix_mask.any():
                    strongest_single = torch.maximum(p_bg[mix_mask], p_wssv[mix_mask])
                    penalty[mix_mask] = 0.5 * F.relu(self.margin + strongest_single - p_mix[mix_mask]).pow(2)
                loss = self.base(logits.float(), targets) + self.lambda_rank * penalty
                return reduce_values(loss, self.reduction)

        return _PairwiseCoInfectionRankingASL()


def make_loss(loss_key: str):
    _torch, nn, _F = _torch_modules()
    if loss_key == "baseline_ce":
        return nn.CrossEntropyLoss()
    if loss_key == "asl_single_label":
        return ASLSingleLabel()
    if loss_key == "pairwise_coinfection_ranking_asl":
        return PairwiseCoInfectionRankingASL()
    raise KeyError(loss_key)

