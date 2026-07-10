"""Sanity checks for YOLO11n-seg custom attention YAMLs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_attention._shared.attention_modules import CUSTOM_ATTENTION_MODULES  # noqa: E402
from custom_attention._shared.register_attention import register_custom_attention  # noqa: E402


def _flatten_modules(module: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in module.modules()]


def _custom_modules_in_model(model: torch.nn.Module) -> list[str]:
    custom_names = {cls.__name__ for cls in CUSTOM_ATTENTION_MODULES}
    found = []
    for module in _flatten_modules(model):
        name = module.__class__.__name__
        if name in custom_names and name not in found:
            found.append(name)
    return found


def sanity_check_yolo_seg_model(model_yaml: str, imgsz: int = 640, device: str | None = None) -> dict[str, Any]:
    """Build a YOLO segmentation model and run one dummy forward pass.

    Args:
        model_yaml: Path to a YOLO model YAML.
        imgsz: Square dummy image size.
        device: Optional torch device string. Defaults to CUDA if available, else CPU.

    Returns:
        A small dictionary with status metadata.
    """

    from ultralytics import YOLO

    register_custom_attention()

    model_yaml_path = Path(model_yaml)
    if not model_yaml_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_yaml_path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building YOLO model: {model_yaml_path}")
    yolo = YOLO(str(model_yaml_path))
    yolo.model.to(device)
    yolo.model.eval()

    dummy = torch.zeros(1, 3, int(imgsz), int(imgsz), device=device)
    with torch.no_grad():
        _ = yolo.model(dummy)

    found = _custom_modules_in_model(yolo.model)
    params = sum(param.numel() for param in yolo.model.parameters())
    print(f"Custom attention modules found: {', '.join(found) if found else 'none'}")
    print(f"Total parameters: {params:,}")
    print(f"PASS: {model_yaml_path}")
    return {
        "model_yaml": str(model_yaml_path),
        "status": "PASS",
        "custom_modules": found,
        "params": params,
        "device": device,
        "imgsz": int(imgsz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check a custom YOLO11n-seg attention YAML.")
    parser.add_argument("--model", required=True, help="Path to model.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Dummy square image size")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda")
    args = parser.parse_args()
    sanity_check_yolo_seg_model(args.model, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
