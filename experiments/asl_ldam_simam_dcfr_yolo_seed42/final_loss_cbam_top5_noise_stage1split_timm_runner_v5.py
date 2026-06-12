#!/usr/bin/env python3
"""
TIMM/PyTorch companion runner for final_loss_cbam_top5_noise_v5_stage1split.

This runner is intentionally separate from the native Ultralytics YOLO runner:
- YOLO models must use native Ultralytics training.
- TIMM/TorchVision models use a native PyTorch/TIMM training loop.

It reproduces the Stage-1/ShrimpXNet-style image-level split:
- sorted class folders
- stratified 70/15/15 split
- seed-controlled train_test_split
- expected 804/172/173 images

It evaluates the same candidate method families used for the YOLO experiment:
- baseline_ce
- asl_ldam_margin
- asl_ldam_margin_cbam
- ce_effective_num_weighted
- ce_effective_num_weighted_cbam
- asl_ldam_simam_dcfr_top1

Noise robustness uses the fixed top-5 corruptions selected from CE-baseline weakness:
- impulse_noise, gaussian_noise, contrast_reduction, defocus_blur, low_light
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    import timm
except Exception as exc:
    raise RuntimeError("Missing timm. Install only missing package: /opt/miniconda3/bin/python -m pip install timm") from exc

try:
    import torchvision
except Exception:
    torchvision = None

CLASS_FOLDERS = ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]
CLASS_NAMES = ["Healthy", "BG", "WSSV", "WSSV_BG"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_FOLDERS)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TRAIN_COUNTS = torch.tensor([282.0, 139.0, 229.0, 154.0], dtype=torch.float32)

TIMM_MODEL_ALIASES = {
    "mobilenetv3_large_100": ["mobilenetv3_large_100", "mobilenetv3_large_100.ra_in1k", "tf_mobilenetv3_large_100"],
    "efficientnet_b0": ["efficientnet_b0", "tf_efficientnet_b0", "efficientnet_b0.ra_in1k"],
    "tf_efficientnetv2_s": ["tf_efficientnetv2_s", "tf_efficientnetv2_s.in21k", "efficientnetv2_rw_s"],
    "shufflenet_v2_x1_0": ["shufflenet_v2_x1_0"],
    "ghostnetv2_100": ["ghostnetv2_100", "ghostnet_100"],
    "mobilevit_s": ["mobilevit_s", "mobilevit_s.cvnets_in1k"],
    "efficientvit_m1": ["efficientvit_m1.r224_in1k", "efficientvit_m1"],
    "fastvit_t8": ["fastvit_t8.apple_in1k", "fastvit_t8"],
    "convnext_tiny": ["convnext_tiny.fb_in22k", "convnext_tiny.in12k_ft_in1k", "convnext_tiny"],
}

METHODS = [
    {"variant_key": "baseline_ce", "loss_key": "ce", "attention_key": "none_baseline", "family": "baseline_ce", "description": "Native PyTorch CE baseline"},
    {"variant_key": "asl_ldam_margin", "loss_key": "asl_ldam_margin", "attention_key": "none_baseline", "family": "loss_only", "description": "ASL-LDAM margin loss"},
    {"variant_key": "asl_ldam_margin_cbam", "loss_key": "asl_ldam_margin", "attention_key": "cbam", "family": "loss_plus_attention", "description": "ASL-LDAM + CBAM"},
    {"variant_key": "ce_effective_num_weighted", "loss_key": "ce_effective_num_weighted", "attention_key": "none_baseline", "family": "ce_weighted", "description": "CE effective-number weighted"},
    {"variant_key": "ce_effective_num_weighted_cbam", "loss_key": "ce_effective_num_weighted", "attention_key": "cbam", "family": "ce_weighted_plus_attention", "description": "CE effective-number weighted + CBAM"},
    {"variant_key": "asl_ldam_simam_dcfr_top1", "loss_key": "asl_ldam_margin", "attention_key": "simam_gated_residual__dcfr_texture", "family": "loss_plus_attention", "description": "ASL-LDAM + SimAM gated residual + DCFR texture"},
]

TOP5_CORRUPTIONS = ["impulse_noise", "gaussian_noise", "contrast_reduction", "defocus_blur", "low_light"]


def now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sanitize(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_").replace(".", "_").replace("-", "-")


def shell(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception as exc:
        return repr(exc)


def validate_dataset(dataset_root: Path, output_dir: Path) -> pd.DataFrame:
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    rows, unreadable = [], []
    for label, folder in enumerate(CLASS_FOLDERS):
        d = dataset_root / folder
        if not d.exists():
            raise FileNotFoundError(f"Missing class folder: {d}")
        files = sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
        print(f"{folder}: {len(files)} images", flush=True)
        for p in files:
            try:
                with Image.open(p) as im:
                    im.verify()
                readable = True
                err = ""
            except Exception as exc:
                readable = False
                err = repr(exc)
                unreadable.append({"path": str(p), "error": err})
            rows.append({
                "source_path": str(p),
                "rel_path": str(p.relative_to(dataset_root)),
                "class_folder": folder,
                "class_name": CLASS_NAMES[label],
                "label": label,
                "readable": readable,
                "read_error": err,
                "file_size": p.stat().st_size,
                "md5": md5_file(p) if readable else "",
            })
    df = pd.DataFrame(rows)
    if len(df) != 1149:
        raise RuntimeError(f"Expected 1149 images, got {len(df)} at {dataset_root}")
    if unreadable:
        pd.DataFrame(unreadable).to_csv(output_dir / "S_split_manifests" / "unreadable_images.csv", index=False)
        raise RuntimeError(f"Found unreadable images: {len(unreadable)}")
    audit = {
        "dataset_root": str(dataset_root),
        "total_images": int(len(df)),
        "class_counts": df["class_folder"].value_counts().reindex(CLASS_FOLDERS).fillna(0).astype(int).to_dict(),
        "unreadable_count": len(unreadable),
        "class_folders": CLASS_FOLDERS,
        "class_names": CLASS_NAMES,
    }
    save_json(output_dir / "S_split_manifests" / "dataset_audit.json", audit)
    df.to_csv(output_dir / "S_split_manifests" / "source_manifest.csv", index=False)
    return df


def create_stage1_split(df: pd.DataFrame, seed: int, output_dir: Path, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_csv = output_dir / "S_split_manifests" / "split_manifest.csv"
    if split_csv.exists() and not force:
        split_df = pd.read_csv(split_csv)
    else:
        train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed, shuffle=True)
        val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["label"], random_state=seed, shuffle=True)
        parts = []
        for split, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            z = part.copy()
            z["split"] = split
            parts.append(z)
        split_df = pd.concat(parts, ignore_index=True)
        split_csv.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(split_csv, index=False)
    counts = split_df["split"].value_counts().to_dict()
    expected = {"train": 804, "val": 172, "test": 173}
    if counts != expected:
        raise RuntimeError(f"Stage1 split mismatch. Expected {expected}, got {counts}")
    # duplicate audits
    for col in ["source_path", "md5"]:
        sets = {s: set(split_df.loc[split_df["split"] == s, col]) for s in ["train", "val", "test"]}
        assert sets["train"].isdisjoint(sets["val"])
        assert sets["train"].isdisjoint(sets["test"])
        assert sets["val"].isdisjoint(sets["test"])
    dist = pd.crosstab(split_df["split"], split_df["class_folder"]).reindex(index=["train", "val", "test"], columns=CLASS_FOLDERS).fillna(0).astype(int)
    dist.to_csv(output_dir / "S_split_manifests" / "class_distribution.csv")
    save_json(output_dir / "S_split_manifests" / "split_protocol.json", {
        "split_type": "fixed random stratified image-level 70/15/15 split",
        "seed": seed,
        "train_val_test_counts": counts,
        "stratified_by": "class label",
        "grouped_by_shrimp_or_image_id": False,
        "note": "Matches Stage-1/ShrimpXNet-style split protocol; not group-safe.",
    })
    return (
        split_df,
        split_df[split_df["split"] == "train"].copy(),
        split_df[split_df["split"] == "val"].copy(),
        split_df[split_df["split"] == "test"].copy(),
    )


def build_transforms(imgsz: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomResizedCrop(imgsz, scale=(0.82, 1.0), ratio=(0.90, 1.10), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


class ShrimpFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None, corruption: Optional[str] = None, severity: int = 0, sample_save_dir: Optional[Path] = None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.corruption = corruption
        self.severity = int(severity or 0)
        self.sample_save_dir = sample_save_dir
        self.saved_sample = False
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img = Image.open(row["source_path"]).convert("RGB")
        if self.corruption and self.corruption != "clean" and self.severity > 0:
            img = apply_corruption(img, self.corruption, self.severity, seed=idx)
            if self.sample_save_dir is not None and not self.saved_sample:
                self.sample_save_dir.mkdir(parents=True, exist_ok=True)
                try:
                    img.save(self.sample_save_dir / f"sample_{self.corruption}_s{self.severity}.png")
                    self.saved_sample = True
                except Exception:
                    pass
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"]), str(row["source_path"])


def apply_corruption(img: Image.Image, corruption: str, severity: int, seed: int = 0) -> Image.Image:
    severity = int(severity)
    rng = np.random.default_rng(seed + severity * 1000)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if corruption == "gaussian_noise":
        sigma = {1: 0.04, 2: 0.08, 3: 0.14}[severity]
        arr = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 1)
    elif corruption == "impulse_noise":
        amount = {1: 0.015, 2: 0.035, 3: 0.070}[severity]
        mask = rng.random(arr.shape[:2]) < amount
        salt = rng.random(arr.shape[:2]) < 0.5
        arr[mask & salt] = 1.0
        arr[mask & ~salt] = 0.0
    elif corruption == "contrast_reduction":
        factor = {1: 0.75, 2: 0.55, 3: 0.35}[severity]
        return ImageEnhance.Contrast(img).enhance(factor)
    elif corruption == "defocus_blur":
        radius = {1: 1.0, 2: 2.0, 3: 3.0}[severity]
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    elif corruption == "low_light":
        factor = {1: 0.70, 2: 0.45, 3: 0.25}[severity]
        return ImageEnhance.Brightness(img).enhance(factor)
    else:
        return img
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


class LocalCBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        scale = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * scale
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True).values], dim=1)
        return x * torch.sigmoid(self.spatial(pooled))


class LocalSimAMDCFR(nn.Module):
    def __init__(self, channels: int, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.texture = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=max(1, channels), bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, h, w = x.shape
        n = max(1, h * w - 1)
        xm = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        sim = x * torch.sigmoid(xm / (4 * (xm.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5)
        tex = x * self.texture(x)
        return x + self.gate(x) * (sim + tex) * 0.5


def make_attention(attention_key: str, channels: int) -> nn.Module:
    key = str(attention_key or "none_baseline")
    if key in {"none", "none_baseline", "baseline", ""}:
        return nn.Identity()
    # Prefer user's patched attention implementation when available.
    try:
        from shrimp_scripts import attention as att
        if hasattr(att, "make_attention"):
            return att.make_attention(key, channels)
    except Exception as exc:
        print(f"WARNING: shrimp_scripts.attention.make_attention failed for {key}: {exc!r}; using local fallback if possible", flush=True)
    if key == "cbam":
        return LocalCBAM(channels)
    if key == "simam_gated_residual__dcfr_texture":
        return LocalSimAMDCFR(channels)
    raise KeyError(f"Unsupported attention_key for TIMM runner: {key}")


class TimmFeatureClassifier(nn.Module):
    def __init__(self, model_key: str, num_classes: int, attention_key: str, imgsz: int, pretrained: bool = True):
        super().__init__()
        self.model_key = model_key
        self.attention_key = attention_key
        self.timm_name, self.backend_source, self.base = self._create_base(model_key, pretrained=pretrained)
        self.channels, self.feature_layout = self._infer_channels(imgsz)
        self.attention = make_attention(attention_key, self.channels)
        self.classifier = nn.Linear(self.channels, num_classes)
    def _create_base(self, model_key: str, pretrained: bool):
        errors = []
        for name in TIMM_MODEL_ALIASES.get(model_key, [model_key]):
            try:
                base = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
                return name, "timm", base
            except Exception as exc:
                errors.append(f"{name}:{type(exc).__name__}:{exc}")
        # torchvision fallback for ShuffleNet only if timm does not have it.
        if model_key == "shufflenet_v2_x1_0" and torchvision is not None:
            try:
                from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
                weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
                tv = shufflenet_v2_x1_0(weights=weights)
                modules = list(tv.children())[:-1]
                base = nn.Sequential(*modules)
                return model_key, "torchvision", base
            except Exception as exc:
                errors.append(f"torchvision:{type(exc).__name__}:{exc}")
        raise RuntimeError(f"Could not create model {model_key}. Tried: {errors}")
    def _raw_features(self, x):
        if self.backend_source == "timm":
            if hasattr(self.base, "forward_features"):
                return self.base.forward_features(x)
            return self.base(x)
        return self.base(x)
    @staticmethod
    def _features_to_nchw(feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 4:
            # Convert NHWC to NCHW when spatial dimensions appear before channel dimension.
            if feat.shape[1] <= 32 and feat.shape[-1] >= 32:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            return feat
        if feat.ndim == 3:
            # NLC -> NC1L. If token dim is first after batch, keep C as last.
            if feat.shape[-1] >= feat.shape[1]:
                feat = feat.transpose(1, 2).unsqueeze(-1).contiguous()
            else:
                feat = feat.unsqueeze(-1).contiguous()
            return feat
        if feat.ndim == 2:
            return feat.unsqueeze(-1).unsqueeze(-1)
        raise RuntimeError(f"Unsupported feature tensor shape: {tuple(feat.shape)}")
    def _infer_channels(self, imgsz: int) -> Tuple[int, str]:
        was = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, imgsz, imgsz)
            feat = self._features_to_nchw(self._raw_features(x))
        if was:
            self.train()
        return int(feat.shape[1]), str(tuple(feat.shape))
    def forward(self, x):
        feat = self._features_to_nchw(self._raw_features(x))
        feat = self.attention(feat)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.classifier(pooled)


def freeze_backbone(model: TimmFeatureClassifier):
    for p in model.base.parameters():
        p.requires_grad = False
    for p in model.attention.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_backbone(model: TimmFeatureClassifier):
    for p in model.parameters():
        p.requires_grad = True


class ASLLDAMFallback(nn.Module):
    def __init__(self, train_counts: torch.Tensor = DEFAULT_TRAIN_COUNTS, max_m: float = 0.5, scale: float = 30.0, gamma_neg: float = 2.0):
        super().__init__()
        m = 1.0 / torch.sqrt(torch.sqrt(train_counts.float()))
        m = m * (max_m / m.max().clamp_min(1e-12))
        self.register_buffer("margins", m)
        self.scale = float(scale)
        self.gamma_neg = float(gamma_neg)
    def forward(self, logits, targets):
        y = targets.long().view(-1).to(logits.device)
        margins = self.margins.to(device=logits.device, dtype=logits.dtype)
        one_hot = F.one_hot(y, num_classes=logits.shape[1]).to(device=logits.device, dtype=logits.dtype)
        logits_m = self.scale * (logits - one_hot * margins.view(1, -1))
        ce = F.cross_entropy(logits_m, y, reduction="none")
        pt = torch.exp(-ce).clamp(1e-8, 1.0)
        return (((1 - pt) ** self.gamma_neg) * ce).mean()


class CEEffectiveNumWeighted(nn.Module):
    def __init__(self, train_counts: torch.Tensor = DEFAULT_TRAIN_COUNTS, beta: float = 0.9999):
        super().__init__()
        counts = train_counts.float()
        beta_t = torch.tensor(beta, dtype=torch.float32)
        weights = (1.0 - beta_t) / (1.0 - torch.pow(beta_t, counts)).clamp_min(1e-12)
        weights = weights / weights.mean().clamp_min(1e-12)
        self.register_buffer("weights", weights)
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets.long().view(-1).to(logits.device), weight=self.weights.to(logits.device, logits.dtype))


def make_loss(loss_key: str) -> nn.Module:
    key = str(loss_key)
    if key in {"ce", "baseline_ce", "native_ce"}:
        return nn.CrossEntropyLoss()
    # Prefer user's exact patched loss implementation when available.
    try:
        from shrimp_scripts import losses as shrimp_losses
        if hasattr(shrimp_losses, "make_loss"):
            return shrimp_losses.make_loss(key)
    except Exception as exc:
        print(f"WARNING: shrimp_scripts.losses.make_loss failed for {key}: {exc!r}; using local fallback if possible", flush=True)
    if key == "asl_ldam_margin":
        return ASLLDAMFallback()
    if key == "ce_effective_num_weighted":
        return CEEffectiveNumWeighted()
    raise KeyError(f"Unsupported loss_key: {key}")


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def model_size_mb(path: Optional[Path]) -> float:
    if path and path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def predict_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion: Optional[nn.Module] = None, timed: bool = False) -> Dict[str, Any]:
    model.eval()
    y_true, y_pred, paths, confs = [], [], [], []
    total_loss, n = 0.0, 0
    if timed and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time() if timed else None
    with torch.no_grad():
        for ims, labels, batch_paths in loader:
            ims = ims.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(ims)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += float(loss.detach().cpu()) * ims.shape[0]
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            confs.extend(probs.max(dim=1).values.detach().cpu().numpy().tolist())
            paths.extend(list(batch_paths))
            n += ims.shape[0]
    if timed and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.time() - t0) if timed else None
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    return {
        "accuracy": float(acc), "macro_precision": float(precision), "macro_recall": float(recall),
        "macro_f1": float(macro_f1), "cohen_kappa": float(kappa), "loss": float(total_loss / max(n, 1)) if criterion else None,
        "elapsed_s": float(elapsed) if elapsed is not None else None,
        "fps": float(n / elapsed) if elapsed and elapsed > 0 else None,
        "latency_ms": float(elapsed / n * 1000) if elapsed and n > 0 else None,
        "n": int(n), "y_true": y_true, "y_pred": y_pred, "paths": paths, "confidence": confs,
    }


def save_eval_artifacts(prefix: str, metrics: Dict[str, Any], out_dir: Path):
    pred_rows = []
    for p, yt, yp, cf in zip(metrics["paths"], metrics["y_true"], metrics["y_pred"], metrics["confidence"]):
        pred_rows.append({"path": p, "true_label": yt, "true_class": CLASS_NAMES[int(yt)], "pred_label": yp, "pred_class": CLASS_NAMES[int(yp)], "confidence": cf})
    pd.DataFrame(pred_rows).to_csv(out_dir / "A_clean_metrics" / f"{prefix}_predictions.csv", index=False)
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(4)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "D_confusion_reports" / f"{prefix}_confusion_matrix.csv")
    report = classification_report(metrics["y_true"], metrics["y_pred"], labels=list(range(4)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / "D_confusion_reports" / f"{prefix}_classification_report.csv")


def train_one_method(args, method: Dict[str, str], train_df, val_df, test_df, train_tf, eval_tf, device: torch.device) -> Dict[str, Any]:
    run_key = f"{sanitize(args.model)}__{method['variant_key']}__seed{args.seed}"
    run_dir = args.output_dir / "runs" / run_key
    status_path = run_dir / "status.json"
    best_ckpt = run_dir / "best.pth"
    final_metrics_path = run_dir / "final_metrics.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(status_path, {"status": "started", "run_key": run_key, "time": now(), "method": method})

    if final_metrics_path.exists() and not args.force:
        print(f"SKIP completed TIMM run: {run_key}", flush=True)
        return json.loads(final_metrics_path.read_text(encoding="utf-8"))

    train_loader = DataLoader(ShrimpFrameDataset(train_df, train_tf), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(ShrimpFrameDataset(val_df, eval_tf), batch_size=args.eval_batch, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(ShrimpFrameDataset(test_df, eval_tf), batch_size=args.eval_batch, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())

    print("\n" + "="*110, flush=True)
    print(f"TRAIN TIMM METHOD {run_key}", flush=True)
    print(json.dumps(method, indent=2), flush=True)
    print("="*110, flush=True)

    model = TimmFeatureClassifier(args.model, num_classes=4, attention_key=method["attention_key"], imgsz=args.imgsz, pretrained=True).to(device)
    criterion = make_loss(method["loss_key"]).to(device)
    freeze_backbone(model)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    scaler = GradScaler(enabled=args.amp and torch.cuda.is_available())

    best_val_f1, best_epoch, bad_epochs = -1.0, -1, 0
    history = []
    train_start = time.time()

    for epoch in range(args.epochs):
        if epoch == min(args.warmup_epochs, args.epochs):
            print("Switching to fine-tune phase: unfreezing TIMM backbone", flush=True)
            unfreeze_backbone(model)
            optimizer = optim.AdamW([
                {"params": model.base.parameters(), "lr": args.lr_backbone},
                {"params": model.attention.parameters(), "lr": args.lr_head},
                {"params": model.classifier.parameters(), "lr": args.lr_head},
            ], weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        tr_loss, tr_n, tr_correct = 0.0, 0, 0
        for step, (ims, labels, _) in enumerate(train_loader):
            ims = ims.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=args.amp and torch.cuda.is_available()):
                logits = model(ims)
                loss = criterion(logits, labels)
                loss_scaled = loss / max(1, args.accumulation_steps)
            scaler.scale(loss_scaled).backward()
            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            tr_loss += float(loss.detach().cpu()) * ims.shape[0]
            tr_correct += int((logits.argmax(1) == labels).sum().detach().cpu())
            tr_n += int(labels.numel())
        scheduler.step()
        val_metrics = predict_model(model, val_loader, device, criterion=criterion, timed=False)
        row = {"epoch": epoch + 1, "train_loss": tr_loss / max(1, tr_n), "train_accuracy": tr_correct / max(1, tr_n), "val_macro_f1": val_metrics["macro_f1"], "val_accuracy": val_metrics["accuracy"], "val_loss": val_metrics["loss"], "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f} val_acc={row['val_accuracy']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_key": args.model,
                "timm_name": model.timm_name,
                "backend_source": model.backend_source,
                "attention_key": method["attention_key"],
                "loss_key": method["loss_key"],
                "variant_key": method["variant_key"],
                "seed": args.seed,
                "imgsz": args.imgsz,
                "class_names": CLASS_NAMES,
                "channels": model.channels,
                "feature_layout": model.feature_layout,
            }, best_ckpt)
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch+1}; best_epoch={best_epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(args.output_dir / "L_training_curves" / f"{run_key}_history.csv", index=False)

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    clean = predict_model(model, test_loader, device, criterion=criterion, timed=True)
    save_eval_artifacts(f"{run_key}_clean", clean, args.output_dir)

    clean_public = {k: clean[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "cohen_kappa", "loss", "elapsed_s", "fps", "latency_ms", "n"]}
    result = {
        "backend": "timm_pytorch",
        "model": args.model,
        "resolved_model_name": model.timm_name,
        "backend_source": model.backend_source,
        "variant_key": method["variant_key"],
        "loss_key": method["loss_key"],
        "attention_key": method["attention_key"],
        "method_family": method["family"],
        "seed": args.seed,
        "best_epoch": best_epoch,
        "train_time_s": time.time() - train_start,
        "params_m": count_params(model),
        "checkpoint_path": str(best_ckpt),
        "model_size_mb": model_size_mb(best_ckpt),
        **{f"clean_{k}": v for k, v in clean_public.items()},
        "status": "evaluated_clean",
    }

    # noise evaluation
    noise_rows = []
    if not args.skip_noise:
        for corr in args.corruptions:
            for sev in args.severities:
                noise_loader = DataLoader(
                    ShrimpFrameDataset(test_df, eval_tf, corruption=corr, severity=sev, sample_save_dir=args.output_dir / "Q_noise_samples"),
                    batch_size=args.eval_batch, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available()
                )
                nm = predict_model(model, noise_loader, device, criterion=criterion, timed=True)
                prefix = f"{run_key}_{corr}_s{sev}"
                # For space, store predictions for smoke/full both; user asked complete review artifacts.
                save_eval_artifacts(prefix, nm, args.output_dir)
                row = {
                    "backend": "timm_pytorch", "model": args.model, "variant_key": method["variant_key"], "loss_key": method["loss_key"], "attention_key": method["attention_key"],
                    "seed": args.seed, "split_name": f"{corr}_s{sev}", "corruption": corr, "severity": sev,
                    "test_macro_f1": nm["macro_f1"], "test_accuracy": nm["accuracy"], "cohen_kappa": nm["cohen_kappa"], "test_macro_precision": nm["macro_precision"], "test_macro_recall": nm["macro_recall"],
                    "clean_macro_f1": clean["macro_f1"], "drop_macro_f1_vs_own_clean": clean["macro_f1"] - nm["macro_f1"],
                    "fps": nm["fps"], "latency_ms": nm["latency_ms"], "status": "evaluated",
                }
                noise_rows.append(row)
        pd.DataFrame(noise_rows).to_csv(args.output_dir / "E_noise_metrics" / f"{run_key}_noise_metrics.csv", index=False)

    result["status"] = "completed"
    save_json(final_metrics_path, result)
    save_json(status_path, {"status": "completed", "run_key": run_key, "time": now()})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def make_dirs(output_dir: Path):
    for name in [
        "A_clean_metrics", "B_method_deltas", "C_classwise_metrics", "D_confusion_reports", "E_noise_metrics", "F_robustness_drops", "G_efficiency", "H_experiment_configs", "I_environment", "J_hyperparameters", "K_status_failures", "L_training_curves", "M_confusion_figures", "N_ranking_figures", "O_robustness_figures", "P_sample_images", "Q_noise_samples", "R_xai_heatmaps", "S_split_manifests", "T_per_run_configs", "U_readme_artifact_index", "runs", "command_logs"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def collect_current_output(output_dir: Path, zip_path: Optional[Path] = None):
    metric_files = list((output_dir / "runs").glob("*/final_metrics.json"))
    clean_rows = []
    for p in metric_files:
        try:
            clean_rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    clean_df = pd.DataFrame(clean_rows)
    if not clean_df.empty:
        clean_df.to_csv(output_dir / "A_clean_metrics" / "timm_clean_results_all_methods.csv", index=False)
        rank = clean_df.sort_values(["clean_macro_f1", "clean_cohen_kappa", "clean_accuracy", "clean_latency_ms", "model_size_mb"], ascending=[False, False, False, True, True])
        rank.to_csv(output_dir / "A_clean_metrics" / "timm_clean_ranking.csv", index=False)
        eff_cols = [c for c in ["model", "variant_key", "seed", "params_m", "model_size_mb", "clean_latency_ms", "clean_fps"] if c in clean_df.columns]
        clean_df[eff_cols].to_csv(output_dir / "G_efficiency" / "timm_efficiency.csv", index=False)
    noise_files = list((output_dir / "E_noise_metrics").glob("*_noise_metrics.csv"))
    noise_rows = []
    for p in noise_files:
        try:
            noise_rows.append(pd.read_csv(p))
        except Exception:
            pass
    if noise_rows:
        noise = pd.concat(noise_rows, ignore_index=True)
        noise.to_csv(output_dir / "E_noise_metrics" / "timm_noise_metrics_all.csv", index=False)
        robust = noise.groupby(["model", "variant_key", "loss_key", "attention_key", "seed"], as_index=False).agg(
            mean_corrupted_macro_f1=("test_macro_f1", "mean"),
            worst_corrupted_macro_f1=("test_macro_f1", "min"),
            mean_drop_macro_f1=("drop_macro_f1_vs_own_clean", "mean"),
            mean_accuracy=("test_accuracy", "mean"),
            mean_kappa=("cohen_kappa", "mean"),
        ).sort_values(["mean_corrupted_macro_f1", "worst_corrupted_macro_f1", "mean_kappa"], ascending=[False, False, False])
        robust.to_csv(output_dir / "F_robustness_drops" / "timm_robustness_ranking.csv", index=False)
    index = {"generated_at": now(), "categories": {}}
    for d in sorted([x for x in output_dir.iterdir() if x.is_dir()]):
        files = [str(p.relative_to(output_dir)) for p in sorted(d.rglob("*")) if p.is_file() and p.suffix not in {".pth", ".pt"}]
        index["categories"][d.name] = files[:200]
    save_json(output_dir / "U_readme_artifact_index" / "artifact_index.json", index)
    (output_dir / "U_readme_artifact_index" / "README_TIMM_RESULTS.md").write_text(
        "# TIMM V5 Results\n\nThis folder contains TIMM/PyTorch results for the Stage-1/ShrimpXNet-style split. YOLO models are trained separately with native Ultralytics YOLO training. TIMM cannot use native YOLO training.\n",
        encoding="utf-8",
    )
    if zip_path:
        if zip_path.exists():
            zip_path.unlink()
        # Exclude raw images and checkpoints from review ZIP.
        shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", output_dir)
        # Remove heavy checkpoint files from zip by rebuilding through shell zip if available.
        tmp = zip_path.with_suffix(".tmp.zip")
        if tmp.exists():
            tmp.unlink()
        cmd = ["bash", "-lc", f"cd '{output_dir}' && zip -qr '{tmp}' . -x '*.pth' '*.pt' '*.cache' 'runs/*/best.pth'"]
        subprocess.run(cmd, check=False)
        if tmp.exists():
            if zip_path.exists(): zip_path.unlink()
            tmp.rename(zip_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset_root", default="/home/drnguyenvinh/notebooks/processed-images/processed_images")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--paper_batch", type=int, default=128)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--step_size", type=int, default=3)
    ap.add_argument("--step_gamma", type=float, default=0.9)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--corruptions", default=",".join(TOP5_CORRUPTIONS))
    ap.add_argument("--severities", default="1,2,3")
    ap.add_argument("--skip_noise", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force_split", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--zip_path", default=None)
    args = ap.parse_args()
    args.dataset_root = Path(args.dataset_root)
    args.output_dir = Path(args.output_dir)
    args.zip_path = Path(args.zip_path) if args.zip_path else None
    args.accumulation_steps = max(1, int(round(args.paper_batch / max(1, args.batch))))
    args.corruptions = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    args.severities = [int(x.strip()) for x in args.severities.split(",") if x.strip()]

    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    set_seed(args.seed)
    make_dirs(args.output_dir)
    save_json(args.output_dir / "I_environment" / "environment.json", {
        "python": sys.version,
        "platform": sys.platform,
        "torch": torch.__version__,
        "timm": getattr(timm, "__version__", None),
        "torchvision": getattr(torchvision, "__version__", None) if torchvision is not None else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": shell(["nvidia-smi"]),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "zip_path"},
        "protocol_note": "TIMM/PyTorch companion runner; YOLO is handled by native Ultralytics runner only.",
    })

    df = validate_dataset(args.dataset_root, args.output_dir)
    split_df, train_df, val_df, test_df = create_stage1_split(df, args.seed, args.output_dir, force=args.force_split)
    train_tf, eval_tf = build_transforms(args.imgsz)

    if args.methods == "all":
        methods = METHODS
    else:
        wanted = {x.strip() for x in args.methods.split(",") if x.strip()}
        methods = [m for m in METHODS if m["variant_key"] in wanted]
        missing = wanted - {m["variant_key"] for m in methods}
        if missing:
            raise ValueError(f"Unknown methods: {sorted(missing)}")

    all_results = []
    for method in methods:
        try:
            result = train_one_method(args, method, train_df, val_df, test_df, train_tf, eval_tf, device)
            all_results.append(result)
        except Exception as exc:
            err = traceback.format_exc()
            fail = {"backend": "timm_pytorch", "model": args.model, "variant_key": method["variant_key"], "loss_key": method["loss_key"], "attention_key": method["attention_key"], "seed": args.seed, "status": "failed", "error": repr(exc), "traceback": err}
            save_json(args.output_dir / "K_status_failures" / f"{sanitize(args.model)}__{method['variant_key']}__seed{args.seed}_failed.json", fail)
            print("FAILED TIMM METHOD", json.dumps(fail, indent=2)[:2000], flush=True)
            all_results.append(fail)
    pd.DataFrame(all_results).to_csv(args.output_dir / "A_clean_metrics" / f"{sanitize(args.model)}_seed{args.seed}_timm_method_results.csv", index=False)
    collect_current_output(args.output_dir, args.zip_path)
    print("DONE TIMM RUN", args.output_dir, flush=True)
    if args.zip_path:
        print("ZIP", args.zip_path, "exists", args.zip_path.exists(), flush=True)


if __name__ == "__main__":
    main()
