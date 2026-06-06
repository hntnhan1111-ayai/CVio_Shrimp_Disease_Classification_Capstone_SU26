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
    "class_weighted_asl": {
        "name": "ClassWeightedASL",
        "type": "screening_asl_variant",
        "custom": True,
        "strength": 0.35,
    },
    "coinfection_weighted_asl": {
        "name": "CoInfectionWeightedASL",
        "type": "screening_asl_variant",
        "custom": True,
        "lambda_coinfection": 0.20,
    },
    "boundary_weighted_asl": {
        "name": "BoundaryWeightedASL",
        "type": "screening_asl_variant",
        "custom": True,
        "lambda_boundary": 0.12,
    },
    "confusion_aware_negative_asl": {
        "name": "ConfusionAwareNegativeASL",
        "type": "screening_asl_variant",
        "custom": True,
        "lambda_confusion": 0.06,
        "danger_gamma": 2.0,
    },
    "soft_target_coinfection_asl": {
        "name": "SoftTargetCoInfectionASL",
        "type": "screening_asl_variant",
        "custom": True,
        "alpha": 0.12,
    },
    "attribute_projection_asl": {
        "name": "AttributeProjectionASL",
        "type": "screening_asl_variant",
        "custom": True,
        "lambda_attr": 0.10,
    },
    "adaptive_gamma_asl": {
        "name": "AdaptiveGammaASL",
        "type": "screening_asl_variant",
        "custom": True,
        "gamma_neg": [4.0, 3.5, 3.5, 2.5],
    },
    "asl_ldam_margin": {
        "name": "ASLLDAMMargin",
        "type": "screening_asl_variant",
        "custom": True,
        "max_margin": 0.12,
    },
    "dangerous_confidence_penalty_asl": {
        "name": "DangerousConfidencePenaltyASL",
        "type": "screening_asl_variant",
        "custom": True,
        "tau": 0.35,
        "lambda_penalty": 0.08,
    },
    "coinfection_logit_adjusted_asl": {
        "name": "CoInfectionLogitAdjustedASL",
        "type": "screening_asl_variant",
        "custom": True,
        "strength": 0.15,
        "coinfection_cap": 0.18,
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


def _validate_logits_targets(logits, targets):
    _torch, _nn, _F = _torch_modules()
    targets = targets.long().view(-1)
    logits = logits.float()
    if logits.ndim != 2 or logits.shape[1] != config.NUM_CLASSES:
        raise ValueError(f"expected logits [B,{config.NUM_CLASSES}], got {tuple(logits.shape)}")
    return _torch, _nn, _F, logits, targets


def _class_counts_tensor(logits):
    _torch, _nn, _F = _torch_modules()
    counts = [config.EXPECTED_NAME_COUNTS[name] for name in config.CLASS_NAMES]
    return _torch.tensor(counts, device=logits.device, dtype=logits.dtype)


def _mild_inverse_frequency_weights(logits, strength: float = 0.35, min_weight: float = 0.85, max_weight: float = 1.25):
    _torch, _nn, _F = _torch_modules()
    counts = _class_counts_tensor(logits)
    raw = _torch.sqrt(counts.mean() / counts.clamp_min(1.0))
    normalized = raw / raw.mean().clamp_min(1e-8)
    return _torch.clamp(1.0 + float(strength) * (normalized - 1.0), min=float(min_weight), max=float(max_weight))


def _target_one_hot(logits, targets):
    _torch, _nn, _F = _torch_modules()
    return _torch.zeros_like(logits.float()).scatter_(1, targets.unsqueeze(1), 1.0)


def _soft_asl_loss(logits, target_probs, gamma_pos=0.0, gamma_neg=4.0, eps=0.0, reduction: str = "none", gamma_neg_vector=None):
    _torch, _nn, _F = _torch_modules()
    logits = logits.float()
    target_probs = target_probs.to(device=logits.device, dtype=logits.dtype)
    log_probs = _F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    anti_targets = 1.0 - target_probs
    if gamma_neg_vector is None:
        gamma_neg_value = float(gamma_neg)
    else:
        gamma_neg_value = gamma_neg_vector.to(device=logits.device, dtype=logits.dtype).view(1, -1)
    gamma = float(gamma_pos) * target_probs + gamma_neg_value * anti_targets
    xs_pos = probs * target_probs
    xs_neg = (1.0 - probs) * anti_targets
    weights = _torch.pow(_torch.clamp(1.0 - xs_pos - xs_neg, min=1e-8), gamma)
    if eps > 0:
        target_probs = target_probs * (1.0 - float(eps)) + float(eps) / logits.size(1)
    loss = -(target_probs * log_probs * weights).sum(dim=1)
    return reduce_values(loss, reduction)


def _dangerous_negative_penalty(probs, targets, *, tau: float | None = None, gamma: float = 2.0):
    _torch, _nn, _F = _torch_modules()
    penalty = _torch.zeros_like(probs[:, 0])
    for true_label, wrong_label in [(1, 3), (2, 3), (3, 1), (3, 2)]:
        mask = targets == true_label
        if mask.any():
            p_wrong = probs[mask, wrong_label].float().clamp(1e-6, 1.0 - 1e-6)
            if tau is None:
                term = -_torch.log1p(-p_wrong) * p_wrong.pow(float(gamma))
            else:
                term = _F.relu(p_wrong - float(tau)).pow(2)
            penalty[mask] = penalty[mask] + term
    return penalty


def _weighted_mean(loss, sample_weights):
    return (loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


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


class ClassWeightedASL(_BaseLossModule):
    """ASL with mild class-count weights normalized around one."""

    def __init__(self, strength: float = 0.35, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction="none")
        self.strength = float(strength)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        loss = self.base(logits, targets)
        class_weights = _mild_inverse_frequency_weights(logits, strength=self.strength)
        sample_weights = class_weights[targets]
        if self.reduction == "mean":
            return _weighted_mean(loss, sample_weights)
        return reduce_values(loss * sample_weights, self.reduction)


class CoInfectionWeightedASL(_BaseLossModule):
    """ASL with a modest extra sample weight for WSSV_BG."""

    def __init__(self, lambda_coinfection: float = 0.20, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction="none")
        self.lambda_coinfection = float(lambda_coinfection)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        loss = self.base(logits, targets)
        sample_weights = _torch.ones_like(loss)
        sample_weights = sample_weights + self.lambda_coinfection * (targets == 3).to(loss.dtype)
        if self.reduction == "mean":
            return _weighted_mean(loss, sample_weights)
        return reduce_values(loss * sample_weights, self.reduction)


class BoundaryWeightedASL(_BaseLossModule):
    """ASL with modest extra weight for BG/WSSV/WSSV_BG boundary classes."""

    def __init__(self, lambda_boundary: float = 0.12, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction="none")
        self.lambda_boundary = float(lambda_boundary)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        loss = self.base(logits, targets)
        sample_weights = _torch.ones_like(loss)
        sample_weights = sample_weights + self.lambda_boundary * (targets != 0).to(loss.dtype)
        if self.reduction == "mean":
            return _weighted_mean(loss, sample_weights)
        return reduce_values(loss * sample_weights, self.reduction)


class ConfusionAwareNegativeASL(_BaseLossModule):
    """ASL plus focused penalties on known dangerous negative classes."""

    def __init__(self, lambda_confusion: float = 0.06, danger_gamma: float = 2.0, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction="none")
        self.lambda_confusion = float(lambda_confusion)
        self.danger_gamma = float(danger_gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        probs = _F.softmax(logits, dim=1)
        loss = self.base(logits, targets) + self.lambda_confusion * _dangerous_negative_penalty(probs, targets, gamma=self.danger_gamma)
        return reduce_values(loss, self.reduction)


class SoftTargetCoInfectionASL(_BaseLossModule):
    """Soft-target ASL for WSSV_BG while keeping WSSV_BG dominant."""

    def __init__(self, alpha: float = 0.12, gamma_pos: float = 0.0, gamma_neg: float = 4.0, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.alpha = float(alpha)
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        target_probs = _target_one_hot(logits, targets)
        mix_mask = targets == 3
        if mix_mask.any():
            target_probs[mix_mask] = logits.new_tensor([0.0, self.alpha / 2.0, self.alpha / 2.0, 1.0 - self.alpha])
        return _soft_asl_loss(logits, target_probs, gamma_pos=self.gamma_pos, gamma_neg=self.gamma_neg, reduction=self.reduction)


class AttributeProjectionASL(_BaseLossModule):
    """ASL plus manual float32 BCE on BG/WSSV attribute projections."""

    def __init__(self, lambda_attr: float = 0.10, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction="none")
        self.lambda_attr = float(lambda_attr)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        probs = _F.softmax(logits, dim=1).float()
        attr_probs = _torch.stack([probs[:, 1] + probs[:, 3], probs[:, 2] + probs[:, 3]], dim=1).clamp(1e-6, 1.0 - 1e-6)
        attr_map = logits.new_tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        attr_targets = attr_map[targets]
        attr_loss = -(attr_targets * attr_probs.log() + (1.0 - attr_targets) * (1.0 - attr_probs).log()).mean(dim=1)
        loss = self.base(logits, targets) + self.lambda_attr * attr_loss
        return reduce_values(loss, self.reduction)


class AdaptiveGammaASL(_BaseLossModule):
    """ASL with class-dependent negative focusing values."""

    def __init__(self, gamma_neg: list[float] | tuple[float, ...] = (4.0, 3.5, 3.5, 2.5), gamma_pos: float = 0.0, eps: float = 0.1, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.gamma_neg = [float(value) for value in gamma_neg]
        self.gamma_pos = float(gamma_pos)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        gamma_vector = logits.new_tensor(self.gamma_neg)
        return _soft_asl_loss(logits, _target_one_hot(logits, targets), gamma_pos=self.gamma_pos, eps=self.eps, reduction=self.reduction, gamma_neg_vector=gamma_vector)


class ASLLDAMMargin(_BaseLossModule):
    """LDAM-style small class-dependent true-class margin before ASL."""

    def __init__(self, max_margin: float = 0.12, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction=reduction)
        self.max_margin = float(max_margin)

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        counts = _class_counts_tensor(logits).clamp_min(1.0)
        margins = 1.0 / _torch.sqrt(_torch.sqrt(counts))
        margins = margins * (self.max_margin / margins.max().clamp_min(1e-8))
        adjusted = logits - _target_one_hot(logits, targets) * margins[targets].unsqueeze(1)
        return self.base(adjusted, targets)


class DangerousConfidencePenaltyASL(_BaseLossModule):
    """ASL plus penalty when dangerous wrong-class confidence exceeds tau."""

    def __init__(self, tau: float = 0.35, lambda_penalty: float = 0.08, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction="none")
        self.tau = float(tau)
        self.lambda_penalty = float(lambda_penalty)
        self.reduction = reduction

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        probs = _F.softmax(logits, dim=1)
        loss = self.base(logits, targets) + self.lambda_penalty * _dangerous_negative_penalty(probs, targets, tau=self.tau)
        return reduce_values(loss, self.reduction)


class CoInfectionLogitAdjustedASL(_BaseLossModule):
    """ASL with conservative prior-based logit adjustment."""

    def __init__(self, strength: float = 0.15, coinfection_cap: float = 0.18, reduction: str = "mean") -> None:
        _torch, _nn, _F = _torch_modules()
        super().__init__()
        self.base = ASLSingleLabel(reduction=reduction)
        self.strength = float(strength)
        self.coinfection_cap = float(coinfection_cap)

    def forward(self, logits, targets):
        _torch, _nn, _F, logits, targets = _validate_logits_targets(logits, targets)
        priors = _class_counts_tensor(logits)
        priors = priors / priors.sum().clamp_min(1.0)
        adjustment = -_torch.log(priors.clamp_min(1e-8))
        adjustment = adjustment - adjustment.mean()
        adjustment[3] = _torch.clamp(adjustment[3], max=self.coinfection_cap / max(self.strength, 1e-8))
        adjusted = logits + self.strength * adjustment.view(1, -1)
        return self.base(adjusted, targets)


LOSS_CHECKPOINT_SAFE_CLASSES = [
    ASLSingleLabel,
    PairwiseCoInfectionRankingASL,
    ClassWeightedASL,
    CoInfectionWeightedASL,
    BoundaryWeightedASL,
    ConfusionAwareNegativeASL,
    SoftTargetCoInfectionASL,
    AttributeProjectionASL,
    AdaptiveGammaASL,
    ASLLDAMMargin,
    DangerousConfidencePenaltyASL,
    CoInfectionLogitAdjustedASL,
]


def make_loss(loss_key: str):
    _torch, nn, _F = _torch_modules()
    if loss_key == "baseline_ce":
        return nn.CrossEntropyLoss()
    if loss_key == "asl_single_label":
        return ASLSingleLabel()
    if loss_key == "pairwise_coinfection_ranking_asl":
        return PairwiseCoInfectionRankingASL()
    if loss_key == "class_weighted_asl":
        return ClassWeightedASL()
    if loss_key == "coinfection_weighted_asl":
        return CoInfectionWeightedASL()
    if loss_key == "boundary_weighted_asl":
        return BoundaryWeightedASL()
    if loss_key == "confusion_aware_negative_asl":
        return ConfusionAwareNegativeASL()
    if loss_key == "soft_target_coinfection_asl":
        return SoftTargetCoInfectionASL()
    if loss_key == "attribute_projection_asl":
        return AttributeProjectionASL()
    if loss_key == "adaptive_gamma_asl":
        return AdaptiveGammaASL()
    if loss_key == "asl_ldam_margin":
        return ASLLDAMMargin()
    if loss_key == "dangerous_confidence_penalty_asl":
        return DangerousConfidencePenaltyASL()
    if loss_key == "coinfection_logit_adjusted_asl":
        return CoInfectionLogitAdjustedASL()
    raise KeyError(loss_key)
