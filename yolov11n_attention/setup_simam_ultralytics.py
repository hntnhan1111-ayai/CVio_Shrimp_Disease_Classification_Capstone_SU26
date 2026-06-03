"""Clone Ultralytics if needed and apply the YOLO11n-seg + SimAM source patch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ULTRALYTICS_DIR = ROOT / "ultralytics"

MODEL_YAML_TEXT = """# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO11-seg instance segmentation model with SimAM attention before the Segment head.
# Model docs: https://docs.ultralytics.com/models/yolo11
# Task docs: https://docs.ultralytics.com/tasks/segment

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n-seg.yaml' will call yolo11-seg.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 203 layers, 2876848 parameters, 2876832 gradients, 10.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 203 layers, 10113248 parameters, 10113232 gradients, 35.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 253 layers, 22420896 parameters, 22420880 gradients, 123.9 GFLOPs
  l: [1.00, 1.00, 512] # summary: 379 layers, 27678368 parameters, 27678352 gradients, 143.0 GFLOPs
  x: [1.00, 1.50, 512] # summary: 379 layers, 62142656 parameters, 62142640 gradients, 320.2 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  # SimAM attention inserted before Segment head for P3, P4, P5 features
  - [16, 1, SimAM, []] # 23-P3/8 SimAM
  - [19, 1, SimAM, []] # 24-P4/16 SimAM
  - [22, 1, SimAM, []] # 25-P5/32 SimAM
  - [[23, 24, 25], 1, Segment, [nc, 32, 256]] # Segment(P3, P4, P5)
"""

SIMAM_CLASS_TEXT = """


class SimAM(nn.Module):
    \"""
    SimAM: Simple, Parameter-Free Attention Module.

    This implementation is designed to be compatible with Ultralytics YAML parsing.
    It accepts optional channel input but does not use it, because SimAM is parameter-free.

    Args:
        c1: optional input channels, ignored.
        e_lambda: small constant for numerical stability.
    \"""

    def __init__(self, c1=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1

        if n <= 0:
            return x

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        return x * self.activation(y)
"""


def replace_once(text: str, old: str, new: str, file_path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {file_path}: {old!r}")
    return text.replace(old, new, 1)


def clone_ultralytics() -> None:
    if (ULTRALYTICS_DIR / "ultralytics").exists():
        print("Ultralytics source found:", ULTRALYTICS_DIR)
        return
    print("Cloning Ultralytics source into:", ULTRALYTICS_DIR)
    subprocess.run(
        ["git", "clone", "https://github.com/ultralytics/ultralytics.git", str(ULTRALYTICS_DIR)],
        check=True,
    )


def patch_conv() -> None:
    path = ULTRALYTICS_DIR / "ultralytics" / "nn" / "modules" / "conv.py"
    text = path.read_text(encoding="utf-8")
    if '"SimAM",' not in text:
        text = replace_once(
            text,
            '    "RepConv",\n    "SpatialAttention",',
            '    "RepConv",\n    "SimAM",\n    "SpatialAttention",',
            path,
        )
    if "class SimAM(nn.Module):" not in text:
        text = text.rstrip() + SIMAM_CLASS_TEXT + "\n"
    path.write_text(text, encoding="utf-8")
    print("Patched:", path)


def patch_modules_init() -> None:
    path = ULTRALYTICS_DIR / "ultralytics" / "nn" / "modules" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    conv_import_block = text.split("from .conv import (", 1)[1].split(")\n", 1)[0]
    if "\n    SimAM," not in conv_import_block:
        text = replace_once(
            text,
            "    RepConv,\n    SpatialAttention,",
            "    RepConv,\n    SimAM,\n    SpatialAttention,",
            path,
        )
    all_block = text.split("__all__ = (", 1)[1]
    if '\n    "SimAM",' not in all_block:
        text = replace_once(
            text,
            '    "SemanticSegment",\n    "SpatialAttention",',
            '    "SemanticSegment",\n    "SimAM",\n    "SpatialAttention",',
            path,
        )
    path.write_text(text, encoding="utf-8")
    print("Patched:", path)


def patch_tasks() -> None:
    path = ULTRALYTICS_DIR / "ultralytics" / "nn" / "tasks.py"
    text = path.read_text(encoding="utf-8")
    modules_import_block = text.split("from ultralytics.nn.modules import (", 1)[1].split(")\n", 1)[0]
    if "\n    SimAM," not in modules_import_block:
        text = replace_once(
            text,
            "    Segment,\n    Segment26,\n    SemanticSegment,",
            "    Segment,\n    Segment26,\n    SemanticSegment,\n    SimAM,",
            path,
        )
    if "elif m is SimAM:" not in text:
        text = replace_once(
            text,
            "        elif m is Concat:\n            c2 = sum(ch[x] for x in f)\n",
            "        elif m is Concat:\n            c2 = sum(ch[x] for x in f)\n"
            "        elif m is SimAM:\n"
            "            c2 = ch[f]\n"
            "            args = [c2, *args]\n",
            path,
        )
    path.write_text(text, encoding="utf-8")
    print("Patched:", path)


def write_model_yaml() -> None:
    path = ULTRALYTICS_DIR / "ultralytics" / "cfg" / "models" / "11" / "yolo11n-seg-simam-head.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MODEL_YAML_TEXT, encoding="utf-8")
    print("Wrote:", path)


def main() -> None:
    clone_ultralytics()
    patch_conv()
    patch_modules_init()
    patch_tasks()
    write_model_yaml()
    print("SimAM Ultralytics setup complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        raise
