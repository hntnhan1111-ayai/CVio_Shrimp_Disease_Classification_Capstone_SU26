"""Sanity checks for YOLO11n-seg research attention YAMLs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterator

import torch


THIS_FILE = Path(__file__).resolve()
ATTENTION_ROOT = THIS_FILE.parents[2]
if str(ATTENTION_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTENTION_ROOT))

from custom_attention_research_modules._shared.attention_modules import (  # noqa: E402
    RESEARCH_ATTENTION_MODULES,
)
from custom_attention_research_modules._shared.register_attention import (  # noqa: E402
    ensure_project_import_paths,
    register_research_attention,
)


def _flatten_tensors(obj: Any) -> Iterator[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _flatten_tensors(item)


def _custom_modules_in_model(model: torch.nn.Module) -> list[str]:
    custom_names = {cls.__name__ for cls in RESEARCH_ATTENTION_MODULES}
    found: list[str] = []
    for module in model.modules():
        name = module.__class__.__name__
        if name in custom_names and name not in found:
            found.append(name)
    return found


def _resolve_yaml_path(model_yaml: str | Path) -> Path:
    path = Path(model_yaml)
    if path.exists():
        return path
    candidate = ATTENTION_ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Model YAML not found: {model_yaml}")


def sanity_check_yolo_seg_model(model_yaml: str, imgsz: int = 640, device: str | None = None) -> dict[str, Any]:
    """Build a YOLO segmentation model and run one dummy forward pass."""

    ensure_project_import_paths()
    register_research_attention()

    from ultralytics import YOLO

    model_yaml_path = _resolve_yaml_path(model_yaml)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building YOLO model: {model_yaml_path}")
    yolo = YOLO(str(model_yaml_path))
    yolo.model.to(device)
    yolo.model.eval()

    print("Model summary:")
    if hasattr(yolo.model, "info"):
        try:
            yolo.model.info(verbose=True)
        except TypeError:
            yolo.model.info()
    else:
        print(yolo.model)

    dummy = torch.zeros(1, 3, int(imgsz), int(imgsz), device=device)
    with torch.no_grad():
        output = yolo.model(dummy)

    tensors = list(_flatten_tensors(output))
    if not tensors:
        print("Warning: no tensor output found to check for NaN.")
    for idx, tensor in enumerate(tensors):
        if torch.isnan(tensor.detach()).any():
            raise RuntimeError(f"NaN detected in output tensor #{idx} with shape {tuple(tensor.shape)}")

    found = _custom_modules_in_model(yolo.model)
    if not found:
        raise RuntimeError("No research attention module was found inside the built YOLO model.")

    params = sum(param.numel() for param in yolo.model.parameters())
    print(f"Custom attention modules found: {', '.join(found)}")
    print(f"Checked output tensors: {len(tensors)}")
    print(f"Total parameters: {params:,}")
    print(f"PASS: {model_yaml_path}")
    return {
        "model_yaml": str(model_yaml_path),
        "status": "PASS",
        "custom_modules": found,
        "params": params,
        "device": device,
        "imgsz": int(imgsz),
        "output_tensors_checked": len(tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check a YOLO11n-seg research attention YAML.")
    parser.add_argument("--model", required=True, help="Path to model YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Dummy square image size")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda")
    args = parser.parse_args()
    sanity_check_yolo_seg_model(args.model, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
