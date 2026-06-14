"""Generate one Grad-CAM example per class for a YOLO classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from cvio_asl_ldam.attention.patch_yolo import register_checkpoint_safe_globals
from cvio_asl_ldam.data.audit_dataset import resolve_dataset_root
from cvio_asl_ldam.utils.io import load_yaml
from cvio_asl_ldam.utils.paths import ProjectPaths, resolve_device


def _examples(path: Path) -> list[dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test" and row["class_name"] not in selected:
                selected[row["class_name"]] = row
    return [selected[name] for name in ("Healthy", "BG", "WSSV", "WSSV_BG")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", default="artifacts/manifests/split_manifest_seed42.csv")
    parser.add_argument("--dataset-config", default="configs/dataset/shrimpdiseasebd_seed42.yaml")
    parser.add_argument("--xai-config", default="configs/xai/gradcam.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from torchvision import transforms
    from ultralytics import YOLO

    paths = ProjectPaths.from_environment()
    dataset_config = load_yaml(args.dataset_config)
    xai_config = load_yaml(args.xai_config)
    dataset_root, _ = resolve_dataset_root(args.data_dir or paths.data_dir, dataset_config)
    output_dir = Path(args.output_dir or xai_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = resolve_device(args.device).split(",")[0]
    device = torch.device(f"cuda:{device_name}" if device_name.isdigit() else "cpu")

    register_checkpoint_safe_globals()
    yolo = YOLO(args.weights)
    network = yolo.model.to(device).eval()

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tensor):
            output = self.model(tensor)
            if torch.is_tensor(output):
                return output
            if isinstance(output, (list, tuple)):
                for item in output:
                    if torch.is_tensor(item) and item.ndim == 2:
                        return item
            raise TypeError("Could not extract classifier logits for Grad-CAM")

    target_layers = [module for module in network.modules() if isinstance(module, nn.Conv2d)]
    if not target_layers:
        raise RuntimeError("No convolutional layer found for Grad-CAM")
    transform = transforms.Compose(
        [
            transforms.Resize((int(xai_config["imgsz"]), int(xai_config["imgsz"]))),
            transforms.ToTensor(),
        ]
    )
    cam = GradCAM(model=Wrapper(network), target_layers=[target_layers[-1]])
    metadata = []
    for row in _examples(Path(args.manifest)):
        image = Image.open(dataset_root / row["rel_path"]).convert("RGB")
        resized = image.resize((int(xai_config["imgsz"]), int(xai_config["imgsz"])))
        rgb = np.asarray(resized).astype(np.float32) / 255.0
        tensor = transform(image).unsqueeze(0).to(device)
        grayscale = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(int(row["label"]))],
        )[0]
        visualization = show_cam_on_image(rgb, grayscale, use_rgb=True)
        output = output_dir / f"{row['class_name']}__gradcam.png"
        Image.fromarray(visualization).save(output)
        metadata.append(
            {
                "class_name": row["class_name"],
                "rel_path": row["rel_path"],
                "output": output.as_posix(),
            }
        )
    with (output_dir / "xai_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Wrote XAI outputs: {output_dir}")


if __name__ == "__main__":
    main()
