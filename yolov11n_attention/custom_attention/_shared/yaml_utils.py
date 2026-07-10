"""Metadata and YAML helpers for custom attention model variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AttentionExperiment:
    folder: str
    class_name: str
    display_name: str
    priority: int
    placement: str
    yaml_args: tuple[Any, ...]
    segment_from: tuple[int, int, int]
    source_layers: tuple[int, ...]
    idea: str
    expected: str
    risk: str
    metrics: str
    notebook_name: str


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


EXPERIMENTS: tuple[AttentionExperiment, ...] = (
    AttentionExperiment(
        folder="sge_eca_head_gate",
        class_name="SGEECAHeadGate",
        display_name="SGE-ECA Head Gate",
        priority=1,
        placement="P3/P4 before Segment",
        yaml_args=(8, 3, 0.0),
        segment_from=(23, 24, 22),
        source_layers=(16, 19),
        idea="Spatial Group Enhance plus ECA channel selection for very light pre-head gating.",
        expected="Improve spatial/channel selectivity with minimal params and latency.",
        risk="Group size must divide channels safely; effect may be close to ECA-only.",
        metrics="mask mAP50, mask mAP50-95, healthy FP rate, speed, params, FLOPs",
        notebook_name="sge_eca_head_gate.ipynb",
    ),
    AttentionExperiment(
        folder="ces_lite",
        class_name="CESLite",
        display_name="CES-Lite",
        priority=2,
        placement="P3/P4 before Segment",
        yaml_args=(32, 3, 1e-4, 0.0),
        segment_from=(23, 24, 22),
        source_layers=(16, 19),
        idea="Coordinate Attention plus ECA and SimAM-style neuron saliency.",
        expected="Improve lesion localization while staying lightweight.",
        risk="Coordinate and neuron gates may overlap spatially; dynamic shape/export must be checked.",
        metrics="mask mAP50, mask mAP50-95, recall, per-class AP BG/WSSV, healthy FP",
        notebook_name="ces_lite.ipynb",
    ),
    AttentionExperiment(
        folder="low_fp_residual_spatial_gate",
        class_name="LowFPResidualSpatialGate",
        display_name="Low-FP Residual Spatial Gate",
        priority=3,
        placement="P3/P4 before Segment",
        yaml_args=(3, 0.0),
        segment_from=(23, 24, 22),
        source_layers=(16, 19),
        idea="Centered residual spatial gate from mean/max descriptors for healthy FP reduction.",
        expected="Reduce healthy false positives with very low overhead.",
        risk="A strong spatial gate can suppress small lesions and lower recall.",
        metrics="healthy images with FP, FP masks/image, precision, recall, mask mAP50",
        notebook_name="low_fp_residual_spatial_gate.ipynb",
    ),
    AttentionExperiment(
        folder="p3p4_semantic_attention_gate",
        class_name="P3P4SemanticAttentionGate",
        display_name="P3P4 Semantic Attention Gate",
        priority=4,
        placement="P3/P4 before Segment, single-input safe variant",
        yaml_args=(16, 3, 0.0),
        segment_from=(23, 24, 22),
        source_layers=(16, 19),
        idea="Self-context semantic-style gate as a safe first pass toward guided neck attention.",
        expected="Suppress noisy lateral features and improve mask/detail.",
        risk="True semantic-guided multi-input gate is not used yet; self-context may be weaker.",
        metrics="mask mAP50-95, healthy FP, BG/WSSV per-class AP, inference speed",
        notebook_name="p3p4_semantic_attention_gate.ipynb",
    ),
    AttentionExperiment(
        folder="boundary_aware_lite_attention",
        class_name="BoundaryAwareLiteAttention",
        display_name="Boundary-Aware Lite Attention",
        priority=5,
        placement="P3/P4 before Segment",
        yaml_args=(3, 8, 0.0),
        segment_from=(23, 24, 22),
        source_layers=(16, 19),
        idea="Depthwise local contrast gate for boundary/detail cues.",
        expected="Improve mask boundary and mask mAP50-95 on irregular lesions.",
        risk="May amplify healthy texture/noise and increase false positives.",
        metrics="mask mAP50-95, qualitative boundary, healthy FP, precision",
        notebook_name="boundary_aware_lite_attention.ipynb",
    ),
    AttentionExperiment(
        folder="context_suppression_gate_lite",
        class_name="ContextSuppressionGateLite",
        display_name="Context-Suppression Gate Lite",
        priority=6,
        placement="P3/P4 before Segment, single-input context variant",
        yaml_args=(16, 0.0),
        segment_from=(23, 24, 22),
        source_layers=(16, 19),
        idea="Global pooled context gate as a safe single-input approximation of P5-guided suppression.",
        expected="Improve precision and reduce false positives with low compute.",
        risk="Context can suppress true disease if disease/healthy images are visually similar.",
        metrics="healthy FP, recall, mask mAP50, latency",
        notebook_name="context_suppression_gate_lite.ipynb",
    ),
    AttentionExperiment(
        folder="ca_spatial_low_fp_gate",
        class_name="CASpatialLowFPGate",
        display_name="CA-Spatial Low-FP Gate",
        priority=7,
        placement="P3/P4 before Segment",
        yaml_args=(32, 3, 0.0),
        segment_from=(23, 24, 22),
        source_layers=(16, 19),
        idea="Coordinate Attention plus centered spatial gate for localization and FP suppression.",
        expected="Balance position-aware localization with background suppression.",
        risk="Two spatial/position gates can be redundant and reduce small-lesion recall.",
        metrics="recall, precision, healthy FP, mask AP50-95, latency",
        notebook_name="ca_spatial_low_fp_gate.ipynb",
    ),
    AttentionExperiment(
        folder="prototype_aware_mask_gate_lite",
        class_name="PrototypeAwareMaskGateLite",
        display_name="Prototype-Aware Mask Gate Lite",
        priority=8,
        placement="P3 only before Segment",
        yaml_args=(3, 3, 0.0),
        segment_from=(23, 19, 22),
        source_layers=(16,),
        idea="P3-oriented mask hint gate as a safe pre-Segment alternative to editing the prototype branch.",
        expected="Improve high-resolution mask detail with limited graph changes.",
        risk="P3 texture is noisy; false positives may increase.",
        metrics="mask mAP50-95, boundary quality, healthy FP, speed",
        notebook_name="prototype_aware_mask_gate_lite.ipynb",
    ),
)


EXPERIMENT_BY_FOLDER = {exp.folder: exp for exp in EXPERIMENTS}
PRIORITY_FOLDERS = tuple(exp.folder for exp in sorted(EXPERIMENTS, key=lambda item: item.priority))


def build_model_dict(exp: AttentionExperiment) -> dict[str, Any]:
    """Return a YOLO11n-seg model dictionary with attention layers inserted."""

    model = {
        "nc": BASE_MODEL["nc"],
        "depth_multiple": BASE_MODEL["depth_multiple"],
        "width_multiple": BASE_MODEL["width_multiple"],
        "backbone": [list(layer) for layer in BASE_MODEL["backbone"]],
        "head": [list(layer) for layer in BASE_MODEL["head"]],
    }
    for source_layer in exp.source_layers:
        model["head"].append([source_layer, 1, exp.class_name, list(exp.yaml_args)])
    model["head"].append([list(exp.segment_from), 1, "Segment", ["nc", 32, 64]])
    return model


def write_model_yaml(exp: AttentionExperiment, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(build_model_dict(exp), f, sort_keys=False)


def all_experiments() -> tuple[AttentionExperiment, ...]:
    return EXPERIMENTS


__all__ = [
    "AttentionExperiment",
    "BASE_MODEL",
    "EXPERIMENTS",
    "EXPERIMENT_BY_FOLDER",
    "PRIORITY_FOLDERS",
    "all_experiments",
    "build_model_dict",
    "write_model_yaml",
]
