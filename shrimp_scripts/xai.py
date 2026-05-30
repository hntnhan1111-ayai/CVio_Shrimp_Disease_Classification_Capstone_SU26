"""Selected Grad-CAM style XAI generation with fail-soft behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from . import config
from .models_torch import create_torch_model, eval_transform
from .utils import ensure_dir, read_json, save_csv


def find_last_conv_module(model):
    import torch.nn as nn

    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    return last_name, last_module


def generate_torch_gradcam(run_dir: Path, split_manifest: pd.DataFrame, output_dir: Path, max_per_class: int = 1) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    rows: list[dict[str, Any]] = []
    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        return [{"run_id": run_dir.name, "backend": "torch", "status": "failed", "reason": "checkpoint_best.pt missing"}]
    state = torch.load(checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model_key = state.get("model_key")
    model, _model_cfg = create_torch_model(model_key)
    model.load_state_dict(state["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    target_name, target_layer = find_last_conv_module(model)
    if target_layer is None:
        return [{"run_id": run_dir.name, "backend": "torch", "status": "failed", "reason": "no Conv2d target layer found"}]
    samples = split_manifest[split_manifest["split"] == "test"].groupby("class_name", group_keys=False).head(max_per_class).reset_index(drop=True)
    xai_dir = ensure_dir(output_dir / "figures" / "xai" / run_dir.name)
    transform = eval_transform()
    for sample in samples.itertuples(index=False):
        try:
            pil = Image.open(sample.processed_path).convert("RGB")
            tensor = transform(pil).unsqueeze(0).to(device)
            rgb = np.asarray(pil.resize((config.IMG_SIZE, config.IMG_SIZE))).astype(np.float32) / 255.0
            with GradCAM(model=model, target_layers=[target_layer]) as cam:
                grayscale = cam(input_tensor=tensor, targets=None)[0]
            overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
            out_path = xai_dir / f"gradcam__{sample.class_name}__{Path(sample.processed_path).stem}.png"
            Image.fromarray(overlay).save(out_path)
            rows.append({"run_id": run_dir.name, "backend": "torch", "method": "GradCAM", "class_name": sample.class_name, "rel_path": sample.rel_path, "target_layer": target_name, "output_path": str(out_path), "status": "success", "reason": ""})
        except Exception as exc:
            rows.append({"run_id": run_dir.name, "backend": "torch", "method": "GradCAM", "class_name": sample.class_name, "rel_path": sample.rel_path, "target_layer": target_name, "output_path": "", "status": "failed", "reason": repr(exc)})
    return rows


def generate_yolo_classification_gradcam(run_dir: Path, yolo_manifest: pd.DataFrame, output_dir: Path, max_per_class: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except Exception as exc:
        return [{"run_id": run_dir.name, "backend": "ultralytics", "status": "failed", "reason": "missing XAI dependency: " + repr(exc)}]
    audit = read_json(run_dir / "run_audit.json", default={})
    checkpoint = Path(str(audit.get("checkpoint_path", "")))
    if not checkpoint.exists():
        return [{"run_id": run_dir.name, "backend": "ultralytics", "status": "failed", "reason": "YOLO checkpoint missing"}]
    try:
        yolo = YOLO(str(checkpoint))
        model = yolo.model
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        target_name, target_layer = find_last_conv_module(model)
        if target_layer is None:
            return [{"run_id": run_dir.name, "backend": "ultralytics", "status": "failed", "reason": "no Conv2d target layer found"}]
        samples = yolo_manifest[yolo_manifest["split"] == "test"].groupby("class_name", group_keys=False).head(max_per_class).reset_index(drop=True)
        xai_dir = ensure_dir(output_dir / "figures" / "xai" / run_dir.name)
        for sample in samples.itertuples(index=False):
            try:
                image = Image.open(sample.yolo_path).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
                rgb = np.asarray(image).astype(np.float32) / 255.0
                bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                tensor = torch.from_numpy(bgr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                with GradCAM(model=model, target_layers=[target_layer]) as cam:
                    grayscale = cam(input_tensor=tensor, targets=None)[0]
                overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
                out_path = xai_dir / f"gradcam__{sample.class_name}__{Path(sample.yolo_path).stem}.png"
                Image.fromarray(overlay).save(out_path)
                rows.append({"run_id": run_dir.name, "backend": "ultralytics", "method": "GradCAM", "class_name": sample.class_name, "rel_path": sample.rel_path, "target_layer": target_name, "output_path": str(out_path), "status": "success", "reason": ""})
            except Exception as exc:
                rows.append({"run_id": run_dir.name, "backend": "ultralytics", "method": "GradCAM", "class_name": sample.class_name, "rel_path": sample.rel_path, "target_layer": target_name, "output_path": "", "status": "failed", "reason": repr(exc)})
    except Exception as exc:
        rows.append({"run_id": run_dir.name, "backend": "ultralytics", "status": "failed", "reason": repr(exc)})
    return rows


def generate_selected_xai(output_dir: str | Path, metrics_frame: pd.DataFrame, split_manifest: pd.DataFrame | None = None, yolo_manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    if metrics_frame.empty:
        frame = pd.DataFrame([{"status": "skipped", "reason": "no completed metrics"}])
        save_csv(frame, output_dir / "xai_status.csv")
        return frame
    selected = metrics_frame.sort_values(["test_macro_f1", "cohen_kappa"], ascending=False).head(4)
    for record in selected.to_dict(orient="records"):
        run_dir = output_dir / "runs" / str(record["run_id"])
        backend = str(record.get("backend", ""))
        try:
            if backend in {"torchvision", "timm"}:
                if split_manifest is None:
                    rows.append({"run_id": run_dir.name, "backend": backend, "status": "failed", "reason": "split_manifest missing"})
                else:
                    rows.extend(generate_torch_gradcam(run_dir, split_manifest, output_dir))
            elif backend == "ultralytics":
                if yolo_manifest is None:
                    rows.append({"run_id": run_dir.name, "backend": backend, "status": "failed", "reason": "yolo_manifest missing"})
                else:
                    rows.extend(generate_yolo_classification_gradcam(run_dir, yolo_manifest, output_dir))
        except Exception as exc:
            rows.append({"run_id": run_dir.name, "backend": backend, "status": "failed", "reason": repr(exc)})
    frame = pd.DataFrame(rows)
    save_csv(frame, output_dir / "xai_status.csv")
    return frame

