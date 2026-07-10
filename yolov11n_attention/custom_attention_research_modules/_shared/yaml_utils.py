"""YAML metadata for the three research attention modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResearchAttentionExperiment:
    folder: str
    class_name: str
    display_name: str
    yaml_name: str
    notebook_name: str
    yaml_args: tuple[Any, ...]
    idea: str
    expected: str
    risk: str
    metrics: str


BASE_MODEL: dict[str, Any] = {
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
    ],
}

P3_LAYER = 16
P4_LAYER = 19
P5_LAYER = 22
P4_ATTENTION_LAYER = 23
SEGMENT_FROM_P4_ONLY = (P3_LAYER, P4_ATTENTION_LAYER, P5_LAYER)


EXPERIMENTS: tuple[ResearchAttentionExperiment, ...] = (
    ResearchAttentionExperiment(
        folder="cote_gate",
        class_name="CoTEGate",
        display_name="CoTE-Gate - Consensus-Regularized Triplet-ECA Gate",
        yaml_name="cote_gate.yaml",
        notebook_name="cote_gate.ipynb",
        yaml_args=(3, 5, 0.20, 0.0),
        idea="Triplet-Lite localization plus ECA channel evidence with disagreement-based soft suppression.",
        expected="Keep mask localization close to Triplet while reducing healthy FP and seg_loss_gap.",
        risk="Disagreement can increase miss rate if alpha grows too strong for weak lesions.",
        metrics="test mask mAP50, full test mAP50, healthy FP, miss rate, seg_loss_gap, latency/export",
    ),
    ResearchAttentionExperiment(
        folder="scsg_gate",
        class_name="SCSGGate",
        display_name="SCSG-Gate - Scale-Aware Coordinate-SGE Gate",
        yaml_name="scsg_gate.yaml",
        notebook_name="scsg_gate.ipynb",
        yaml_args=(32, 8, 1.0, 0.0),
        idea="Coordinate Attention positional prior plus SGE group semantic denoising.",
        expected="Stable low-risk model with better localization than SGE-ECA alone.",
        risk="May be too conservative to beat the highest-mAP Triplet-style runs.",
        metrics="mean/std across seeds later, mask mAP50, healthy FP, miss rate, seg_loss_gap",
    ),
    ResearchAttentionExperiment(
        folder="lpsc_gate",
        class_name="LPSCGate",
        display_name="LPSC-Gate - Lesion-Preserving NAM-SimAM Contrast Gate",
        yaml_name="lpsc_gate.yaml",
        notebook_name="lpsc_gate.ipynb",
        yaml_args=(0.0001, 1.0, 0.5, 0.0),
        idea="NAM low-FP bias with SimAM saliency rescue and a light depthwise local contrast prior.",
        expected="Reduce healthy false positives while preserving weak lesion/boundary evidence better than NAM alone.",
        risk="The contrast prior can react to high-contrast healthy texture, so this starts as P4-only.",
        metrics="healthy FP, precision, miss rate, mask mAP50-95, boundary quality, healthy-aware score",
    ),
)

EXPERIMENT_BY_FOLDER = {exp.folder: exp for exp in EXPERIMENTS}


def build_model_dict(exp: ResearchAttentionExperiment) -> dict[str, Any]:
    """Return a YOLO11n-seg model dictionary with one P4-only attention layer."""

    model = {
        "nc": BASE_MODEL["nc"],
        "depth_multiple": BASE_MODEL["depth_multiple"],
        "width_multiple": BASE_MODEL["width_multiple"],
        "backbone": [list(layer) for layer in BASE_MODEL["backbone"]],
        "head": [list(layer) for layer in BASE_MODEL["head"]],
    }
    model["head"].append([P4_LAYER, 1, exp.class_name, list(exp.yaml_args)])
    model["head"].append([list(SEGMENT_FROM_P4_ONLY), 1, "Segment", ["nc", 32, 64]])
    return model


def write_model_yaml(exp: ResearchAttentionExperiment, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(build_model_dict(exp), f, sort_keys=False)


def all_experiments() -> tuple[ResearchAttentionExperiment, ...]:
    return EXPERIMENTS


__all__ = [
    "BASE_MODEL",
    "EXPERIMENTS",
    "EXPERIMENT_BY_FOLDER",
    "P3_LAYER",
    "P4_ATTENTION_LAYER",
    "P4_LAYER",
    "P5_LAYER",
    "ResearchAttentionExperiment",
    "SEGMENT_FROM_P4_ONLY",
    "all_experiments",
    "build_model_dict",
    "write_model_yaml",
]
