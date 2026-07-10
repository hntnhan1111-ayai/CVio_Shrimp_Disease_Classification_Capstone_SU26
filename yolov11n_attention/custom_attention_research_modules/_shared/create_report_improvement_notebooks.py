"""Create Kaggle-standalone improvement notebooks from the strong-augmentation template."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import yaml
from nbformat.validator import validate


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cote_gate" / "cote_gate_strong_augmentation.ipynb"


BASE_MODEL = {
    "nc": 2,
    "depth_multiple": 1.0,
    "width_multiple": 1.0,
    "backbone": [
        [-1, 1, "Conv", [16, 3, 2]],
        [-1, 1, "Conv", [32, 3, 2]],
        [-1, 1, "C3k2", [64, False, 0.25]],
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "C3k2", [64, False, 0.25]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 1, "C3k2", [128, False, 0.25]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 1, "C3k2", [256, True]],
        [-1, 1, "SPPF", [256, 5]],
        [-1, 1, "C2PSA", [256]],
    ],
    "head": [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 1, "C3k2", [128, False]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 1, "C3k2", [64, False]],
        [-1, 1, "Conv", [64, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 1, "C3k2", [128, False]],
        [-1, 1, "Conv", [128, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 1, "C3k2", [256, True]],
        [19, 1, "PLACEHOLDER_ATTENTION", []],
        [[16, 23, 22], 1, "Segment", ["nc", 32, 64]],
    ],
}


VARIANTS = [
    {
        "folder": "cote_sr_gate",
        "notebook": "cote_sr_p4_strong.ipynb",
        "experiment": "cote_sr_p4_strong",
        "class_name": "CoTESimAMRescueGate",
        "args": [3, 5, 0.10, 0.25, 0.0],
        "title": "CoTE-SR P4 Strong - CoTE-SimAM Rescue Gate",
        "priority": "Priority 1 from the report.",
        "idea": "Triplet-Lite and ECA evidence with a light SimAM rescue term to recover weak lesion and boundary features.",
        "risk": "SimAM rescue can reopen healthy texture false positives if the rescue term becomes too strong.",
        "train_name": "YOLO11n-seg + CoTE-SR - CoTE-SimAM Rescue Gate + Strong Augmentation",
    },
    {
        "folder": "lpsc_er_gate",
        "notebook": "lpsc_er_p4_strong.ipynb",
        "experiment": "lpsc_er_p4_strong",
        "class_name": "LPSCECARescueGate",
        "args": [3, 0.25, 1.25, 0.25, 0.50, 0.0],
        "title": "LPSC-ER P4 Strong - LPSC-ECA Rescue Gate",
        "priority": "Priority 2 from the report.",
        "idea": "Soft LPSC suppression with ECA channel rescue to reduce disease miss while keeping the low-FP lineage.",
        "risk": "ECA rescue can increase healthy false positives if the channel rescue is too open.",
        "train_name": "YOLO11n-seg + LPSC-ER - LPSC-ECA Rescue Gate + Strong Augmentation",
    },
    {
        "folder": "cote_sd_gate",
        "notebook": "cote_sd_p4_strong.ipynb",
        "experiment": "cote_sd_p4_strong",
        "class_name": "CoTESoftDisagreementGate",
        "args": [3, 5, 0.50, 0.05, 0.0],
        "title": "CoTE-SD P4 Strong - CoTE-Soft Disagreement Gate",
        "priority": "High-priority CoTE ablation from the report.",
        "idea": "Keep the original CoTE branches but replace hard disagreement with tanh-softened disagreement.",
        "risk": "Softening disagreement alone may not restore mAP if CoTE mainly needs an explicit rescue signal.",
        "train_name": "YOLO11n-seg + CoTE-SD - CoTE-Soft Disagreement Gate + Strong Augmentation",
    },
    {
        "folder": "lpsc_sg_gate",
        "notebook": "lpsc_sg_p4_strong.ipynb",
        "experiment": "lpsc_sg_p4_strong",
        "class_name": "LPSCSoftGate",
        "args": [0.25, 1.25, 0.25, 0.0],
        "title": "LPSC-SG P4 Strong - LPSC-Soft Gate",
        "priority": "High-priority LPSC softening ablation from the report.",
        "idea": "Reduce NAM dominance, strengthen SimAM rescue, and keep local contrast prior mild.",
        "risk": "If softening is too strong, LPSC may lose its low healthy false-positive advantage.",
        "train_name": "YOLO11n-seg + LPSC-SG - LPSC-Soft Gate + Strong Augmentation",
    },
    {
        "folder": "lpsc_ua_gate",
        "notebook": "lpsc_ua_p4_strong.ipynb",
        "experiment": "lpsc_ua_p4_strong",
        "class_name": "LPSCUncertaintyAttenuatedGate",
        "args": [0.10, 0.25, 1.25, 0.25, 0.0],
        "title": "LPSC-UA P4 Strong - LPSC-Uncertainty Attenuated Gate",
        "priority": "High-priority LPSC uncertainty ablation from the report.",
        "idea": "When NAM and SimAM disagree, attenuate NAM instead of trusting suppressive saliency too strongly.",
        "risk": "Too much attenuation can remove the conservative low-FP behavior that made LPSC useful.",
        "train_name": "YOLO11n-seg + LPSC-UA - LPSC-Uncertainty Attenuated Gate + Strong Augmentation",
    },
    {
        "folder": "cote_bl_gate",
        "notebook": "cote_bl_p4_strong.ipynb",
        "experiment": "cote_bl_p4_strong",
        "class_name": "CoTEBoundaryLiteGate",
        "args": [3, 5, 0.10, 0.10, 0.0],
        "title": "CoTE-BL P4 Strong - CoTE-BoundaryLite Residual Gate",
        "priority": "Medium-priority CoTE boundary ablation from the report.",
        "idea": "Add a very light depthwise boundary prior to CoTE to test whether mask high-IoU quality improves.",
        "risk": "Boundary priors can amplify shell edges, gill texture, or lighting boundaries in healthy images.",
        "train_name": "YOLO11n-seg + CoTE-BL - CoTE-BoundaryLite Residual Gate + Strong Augmentation",
    },
]


ATTENTION_CELL_TEMPLATE = r'''# -- Patch research attention modules into ultralytics namespace -----------------
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

VENDORED_ULTRALYTICS = Path.cwd() / 'ultralytics'
if (VENDORED_ULTRALYTICS / 'ultralytics' / '__init__.py').exists() and str(VENDORED_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(VENDORED_ULTRALYTICS))

loaded_ultralytics = sys.modules.get('ultralytics')
if loaded_ultralytics is not None and not hasattr(loaded_ultralytics, '__version__'):
    for module_name in list(sys.modules):
        if module_name == 'ultralytics' or module_name.startswith('ultralytics.'):
            del sys.modules[module_name]

if importlib.util.find_spec('ultralytics') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'])

"""Kaggle-standalone research attention modules for YOLO11n-seg.

Every module accepts [B, C, H, W] and returns the same shape. The notebook
patches Ultralytics inline and creates the YAML inline, so no external custom
Python or YAML file is required after uploading this notebook to Kaggle.
"""

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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = y.squeeze(-1).transpose(1, 2).contiguous()
        y = self.conv(y)
        y = y.transpose(1, 2).unsqueeze(-1).contiguous()
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
    """Triplet-style shape-preserving gate with safe contiguous permutes."""

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
        gate_hw = self.hw(x)

        x_cw = x.permute(0, 2, 1, 3).contiguous()
        gate_cw = self.cw(x_cw).permute(0, 2, 1, 3).contiguous()
        gate_cw = gate_cw.expand(b, c, h, w)

        x_hc = x.permute(0, 3, 2, 1).contiguous()
        gate_hc = self.hc(x_hc).permute(0, 3, 2, 1).contiguous()
        gate_hc = gate_hc.expand(b, c, h, w)

        return (gate_hw + gate_cw + gate_hc) / 3.0


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


class LocalContrastPriorLite(nn.Module):
    """Export-friendly local contrast prior returning [B, C, H, W]."""

    def __init__(self, c1: int):
        super().__init__()
        channels = _resolve_channels(c1)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prior = self.dw(x)
        return torch.sigmoid(prior - self.smooth(prior))


class CoTEGate(nn.Module):
    """Original Consensus-Regularized Triplet-ECA Gate."""

    def __init__(self, c1: int, eca_k: int = 3, triplet_k: int = 5, alpha_init: float = 0.20, gamma_init: float = 0.0):
        super().__init__()
        channels = _resolve_channels(c1)
        self.triplet = TripletLiteGate(channels, k=triplet_k)
        self.eca = ECAGate(channels, k_size=eca_k)
        self.lambda_t = nn.Parameter(torch.tensor(1.0))
        self.lambda_c = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        t = self.triplet(x)
        c = self.eca(x).expand_as(x)
        d = torch.abs(t - c)
        gate = torch.sigmoid(self.lambda_t * t + self.lambda_c * c - self.alpha * d)
        return identity + self.gamma * identity * gate


class CoTESimAMRescueGate(nn.Module):
    """CoTE-SR: CoTE plus SimAM rescue for weak lesions and boundaries."""

    def __init__(
        self,
        c1: int,
        eca_k: int = 3,
        triplet_k: int = 5,
        alpha_init: float = 0.10,
        eta_init: float = 0.25,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.triplet = TripletLiteGate(channels, k=triplet_k)
        self.eca = ECAGate(channels, k_size=eca_k)
        self.simam = SimAMGate(channels)
        self.lambda_t = nn.Parameter(torch.tensor(1.0))
        self.lambda_c = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        t = self.triplet(x)
        c = self.eca(x).expand_as(x)
        s = self.simam(x)
        d = torch.abs(t - c)
        gate = torch.sigmoid(self.lambda_t * t + self.lambda_c * c + self.eta * s - self.alpha * d)
        return identity + self.gamma * identity * gate


class CoTESoftDisagreementGate(nn.Module):
    """CoTE-SD: CoTE with tanh-softened disagreement penalty."""

    def __init__(
        self,
        c1: int,
        eca_k: int = 3,
        triplet_k: int = 5,
        tau_init: float = 0.50,
        alpha_init: float = 0.05,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.triplet = TripletLiteGate(channels, k=triplet_k)
        self.eca = ECAGate(channels, k_size=eca_k)
        self.lambda_t = nn.Parameter(torch.tensor(1.0))
        self.lambda_c = nn.Parameter(torch.tensor(1.0))
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        t = self.triplet(x)
        c = self.eca(x).expand_as(x)
        soft_d = torch.tanh(self.tau * torch.abs(t - c))
        gate = torch.sigmoid(self.lambda_t * t + self.lambda_c * c - self.alpha * soft_d)
        return identity + self.gamma * identity * gate


class CoTEBoundaryLiteGate(nn.Module):
    """CoTE-BL: CoTE with a very light depthwise boundary prior."""

    def __init__(
        self,
        c1: int,
        eca_k: int = 3,
        triplet_k: int = 5,
        rho_init: float = 0.10,
        alpha_init: float = 0.10,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.triplet = TripletLiteGate(channels, k=triplet_k)
        self.eca = ECAGate(channels, k_size=eca_k)
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.lambda_t = nn.Parameter(torch.tensor(1.0))
        self.lambda_c = nn.Parameter(torch.tensor(1.0))
        self.rho = nn.Parameter(torch.tensor(float(rho_init)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        t = self.triplet(x)
        c = self.eca(x).expand_as(x)
        d = torch.abs(t - c)
        b = torch.sigmoid(self.dw3(x) + self.dw5(x))
        gate = torch.sigmoid(self.lambda_t * t + self.lambda_c * c + self.rho * b - self.alpha * d)
        return identity + self.gamma * identity * gate


class LPSCGate(nn.Module):
    """Original Lesion-Preserving NAM-SimAM Contrast Gate."""

    def __init__(self, c1: int, eta_init: float = 1.0, delta_init: float = 0.5, gamma_init: float = 0.0):
        super().__init__()
        channels = _resolve_channels(c1)
        self.nam = NAMGate(channels)
        self.simam = SimAMGate(channels)
        self.prior = LocalContrastPriorLite(channels)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n = self.nam(x)
        s = self.simam(x)
        p = self.prior(x)
        gate = torch.sigmoid(n + self.eta * s + self.delta * p)
        return identity + self.gamma * identity * gate


class LPSCSoftGate(nn.Module):
    """LPSC-SG: weaker NAM, stronger SimAM rescue, and mild local prior."""

    def __init__(
        self,
        c1: int,
        beta_init: float = 0.25,
        eta_init: float = 1.25,
        delta_init: float = 0.25,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.nam = NAMGate(channels)
        self.simam = SimAMGate(channels)
        self.prior = LocalContrastPriorLite(channels)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n = self.nam(x)
        s = self.simam(x)
        p = self.prior(x)
        gate = torch.sigmoid(self.beta * n + self.eta * s + self.delta * p)
        return identity + self.gamma * identity * gate


class LPSCECARescueGate(nn.Module):
    """LPSC-ER: softened LPSC with ECA channel rescue."""

    def __init__(
        self,
        c1: int,
        eca_k: int = 3,
        beta_init: float = 0.25,
        eta_init: float = 1.25,
        delta_init: float = 0.25,
        lambda_init: float = 0.50,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.nam = NAMGate(channels)
        self.simam = SimAMGate(channels)
        self.prior = LocalContrastPriorLite(channels)
        self.eca = ECAGate(channels, k_size=eca_k)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))
        self.lambda_c = nn.Parameter(torch.tensor(float(lambda_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n = self.nam(x)
        s = self.simam(x)
        p = self.prior(x)
        c = self.eca(x).expand_as(x)
        gate = torch.sigmoid(self.beta * n + self.eta * s + self.delta * p + self.lambda_c * c)
        return identity + self.gamma * identity * gate


class LPSCUncertaintyAttenuatedGate(nn.Module):
    """LPSC-UA: attenuate NAM when NAM and SimAM disagree."""

    def __init__(
        self,
        c1: int,
        alpha_init: float = 0.10,
        beta_init: float = 0.25,
        eta_init: float = 1.25,
        delta_init: float = 0.25,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.nam = NAMGate(channels)
        self.simam = SimAMGate(channels)
        self.prior = LocalContrastPriorLite(channels)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n = self.nam(x)
        s = self.simam(x)
        p = self.prior(x)
        u = torch.abs(n - s)
        n_soft = (1.0 - self.alpha * u) * n
        gate = torch.sigmoid(self.beta * n_soft + self.eta * s + self.delta * p)
        return identity + self.gamma * identity * gate


RESEARCH_ATTENTION_MODULES: tuple[type[nn.Module], ...] = (
    ECAGate,
    TripletLiteGate,
    NAMGate,
    SimAMGate,
    LocalContrastPriorLite,
    CoTEGate,
    CoTESimAMRescueGate,
    CoTESoftDisagreementGate,
    CoTEBoundaryLiteGate,
    LPSCGate,
    LPSCSoftGate,
    LPSCECARescueGate,
    LPSCUncertaintyAttenuatedGate,
)


def module_name_list(modules: Iterable[type[nn.Module]] = RESEARCH_ATTENTION_MODULES) -> list[str]:
    return [module.__name__ for module in modules]


MODULE_FOLDER_NAME = "__MODULE_FOLDER_NAME__"
EXPERIMENT_NAME = "__EXPERIMENT_NAME__"


def register_research_attention_inline():
    import ultralytics.nn.modules as nn_modules
    import ultralytics.nn.modules.conv as conv_module
    import ultralytics.nn.tasks as tasks_module

    for module_cls in RESEARCH_ATTENTION_MODULES:
        setattr(tasks_module, module_cls.__name__, module_cls)
        setattr(nn_modules, module_cls.__name__, module_cls)
        setattr(conv_module, module_cls.__name__, module_cls)

    nn_modules.__all__ = list(
        set(list(getattr(nn_modules, '__all__', [])) + module_name_list(RESEARCH_ATTENTION_MODULES))
    )
    tasks_module.RESEARCH_ATTENTION_MODULES = RESEARCH_ATTENTION_MODULES

    if getattr(tasks_module, '_shrimp_research_attention_inline_patched', False):
        return

    source = inspect.getsource(tasks_module.parse_model)
    insert = """        elif m in RESEARCH_ATTENTION_MODULES:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
"""
    if 'elif m in RESEARCH_ATTENTION_MODULES:' not in source:
        markers = (
            "        elif m in frozenset(\n            {\n                Detect,",
            "        elif m is SemanticSegment:",
            "        elif m in frozenset({TorchVision, Index}):",
        )
        for marker in markers:
            if marker in source:
                source = source.replace(marker, insert + marker, 1)
                break
        else:
            raise RuntimeError('Could not patch ultralytics.nn.tasks.parse_model for research attention modules.')

    exec(compile(source, '<research_attention_inline_parse_model>', 'exec'), tasks_module.__dict__)
    tasks_module._shrimp_research_attention_inline_patched = True


register_research_attention_inline()
print('Research attention modules registered inline:', ', '.join(module_name_list()))
print('Experiment name:', EXPERIMENT_NAME)
'''


def build_yaml_text(class_name: str, args: list[object]) -> str:
    model = {
        "nc": BASE_MODEL["nc"],
        "depth_multiple": BASE_MODEL["depth_multiple"],
        "width_multiple": BASE_MODEL["width_multiple"],
        "backbone": BASE_MODEL["backbone"],
        "head": [list(item) for item in BASE_MODEL["head"]],
    }
    model["head"][-2] = [19, 1, class_name, args]
    return yaml.safe_dump(model, sort_keys=False)


def build_yaml_cell(variant: dict[str, object]) -> str:
    yaml_text = build_yaml_text(str(variant["class_name"]), list(variant["args"]))
    return f'''# -- Create model YAML inside the notebook and run a build/shape sanity check ----
from pathlib import Path

import torch
from ultralytics import YOLO

YAML_CONTENT = {yaml_text!r}

yaml_dir = Path.cwd() / 'custom_attention_research_modules' / MODULE_FOLDER_NAME / 'generated_yamls'
yaml_dir.mkdir(parents=True, exist_ok=True)
yaml_path = yaml_dir / f'{{EXPERIMENT_NAME}}.yaml'
yaml_path.write_text(YAML_CONTENT, encoding='utf-8')

MODEL_YAML = str(yaml_path)
print('YAML written to:', yaml_path)


def _flatten_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _flatten_tensors(item)


def sanity_check_yolo_seg_model_inline(model_yaml, imgsz=640, device=None):
    register_research_attention_inline()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(str(model_yaml))
    yolo.model.to(device).eval()
    print('Model summary:')
    try:
        yolo.model.info(verbose=True)
    except TypeError:
        yolo.model.info()

    dummy = torch.zeros(1, 3, int(imgsz), int(imgsz), device=device)
    with torch.no_grad():
        output = yolo.model(dummy)

    tensors = list(_flatten_tensors(output))
    for idx, tensor in enumerate(tensors):
        if torch.isnan(tensor.detach()).any():
            raise RuntimeError(f'NaN detected in output tensor #{{idx}} with shape {{tuple(tensor.shape)}}')

    custom_names = {{cls.__name__ for cls in RESEARCH_ATTENTION_MODULES}}
    found = []
    for module in yolo.model.modules():
        name = module.__class__.__name__
        if name in custom_names and name not in found:
            found.append(name)
    if not found:
        raise RuntimeError('No research attention module found in the built model.')

    params = sum(p.numel() for p in yolo.model.parameters())
    print('Custom attention modules found:', ', '.join(found))
    print('Checked output tensors:', len(tensors))
    print(f'Total parameters: {{params:,}}')
    print('PASS:', model_yaml)
    return {{'status': 'PASS', 'custom_modules': found, 'params': params}}


sanity_check_yolo_seg_model_inline(MODEL_YAML, imgsz=640)
'''


def build_header(variant: dict[str, object]) -> str:
    return f'''# {variant["title"]}

This Kaggle-standalone notebook implements the P4-only strong-augmentation variant recommended in `deep-research-report (1).md`.

- Module folder: `{variant["folder"]}`
- Experiment name: `{variant["experiment"]}`
- Python class: `{variant["class_name"]}`
- Placement: P4-only before the YOLO11n-seg Segment head
- Attention/YAML setup: defined inline in this notebook; no external custom module or YAML file is required on Kaggle
- Strong augmentation: matched to `yolov11n_simam_NhomA/yolov11n_simam_augmentation.ipynb`
- Dataset protocol: grouped/no-leakage split, same `nc`, class names, validation/test logic, and healthy-negative evaluation

Priority: {variant["priority"]}

Idea: {variant["idea"]}

Risk: {variant["risk"]}

Track: full/labeled-only mask mAP50 and mAP50-95, healthy FP-rate, FP masks/image, disease miss rate, count MAE, healthy-aware score, and segmentation loss gap.
'''


def replace_train_name(source: str, train_name: str) -> str:
    return re.sub(
        r"'name': 'YOLO11n-seg \+ .*? \+ Strong Augmentation',",
        f"'name': '{train_name}',",
        source,
        count=1,
    )


def write_variant(variant: dict[str, object]) -> Path:
    nb = nbformat.read(TEMPLATE, as_version=4)
    nb.nbformat_minor = 5

    nb.cells[0].source = build_header(variant)
    nb.cells[5].source = (
        ATTENTION_CELL_TEMPLATE
        .replace("__MODULE_FOLDER_NAME__", str(variant["folder"]))
        .replace("__EXPERIMENT_NAME__", str(variant["experiment"]))
    )
    nb.cells[6].source = build_yaml_cell(variant)
    nb.cells[17].source = replace_train_name(nb.cells[17].source, str(variant["train_name"]))

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    validate(nb)

    out_dir = ROOT / str(variant["folder"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(variant["notebook"])
    nbformat.write(nb, out_path)
    return out_path


def main() -> None:
    written = [write_variant(variant) for variant in VARIANTS]
    for path in written:
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
