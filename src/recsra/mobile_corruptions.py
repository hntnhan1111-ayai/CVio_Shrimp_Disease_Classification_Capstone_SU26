#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

PROTOCOL = json.loads((Path(__file__).resolve().parent / "noise_protocol.json").read_text())
NOISES = {item["id"]: item for item in PROTOCOL["noises"]}


def stable_seed(relative_name: str, noise_id: str, severity: int, global_seed: int = 42) -> int:
    token = f"{global_seed}|{relative_name}|{noise_id}|{severity}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _disc_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = radius * 2 + 1
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    k = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    k /= max(float(k.sum()), 1.0)
    return k


def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    size = max(3, int(size) | 1)
    k = np.zeros((size, size), np.float32)
    c = size // 2
    length = c
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad) * length, np.sin(rad) * length
    p1 = (int(round(c - dx)), int(round(c - dy)))
    p2 = (int(round(c + dx)), int(round(c + dy)))
    cv2.line(k, p1, p2, 1.0, 1, cv2.LINE_AA)
    k /= max(float(k.sum()), 1e-6)
    return k


def apply_corruption(image_bgr: np.ndarray, noise_id: str, severity: int, seed: int) -> np.ndarray:
    if noise_id not in NOISES:
        raise KeyError(noise_id)
    if severity not in range(1, 6):
        raise ValueError("severity must be 1..5")
    rng = np.random.default_rng(seed)
    idx = severity - 1
    cfg = NOISES[noise_id]["severity"]
    x = image_bgr.astype(np.float32) / 255.0
    h, w = x.shape[:2]

    if noise_id == "N01":
        exposure = cfg["exposure_factor"][idx]
        peak = float(cfg["photon_peak"][idx])
        read_sigma = float(cfg["read_sigma_255"][idx]) / 255.0
        dark = np.clip(x * exposure, 0, 1)
        shot = rng.poisson(dark * peak).astype(np.float32) / peak
        read = rng.normal(0.0, read_sigma, dark.shape).astype(np.float32)
        y = shot + read

    elif noise_id == "N02":
        gain = float(cfg["gain"][idx])
        y = np.clip(x * gain, 0, 1)

    elif noise_id == "N03":
        alpha = float(cfg["alpha"][idx])
        spots = int(cfg["spots"][idx])
        mask = np.zeros((h, w), np.float32)
        for _ in range(spots):
            cx = int(rng.uniform(0.20, 0.80) * w)
            cy = int(rng.uniform(0.20, 0.80) * h)
            ax = max(8, int(rng.uniform(0.05, 0.14) * w))
            ay = max(8, int(rng.uniform(0.03, 0.10) * h))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 1.0, -1, cv2.LINE_AA)
        sigma = max(3.0, min(h, w) * (0.012 + 0.006 * idx))
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        mask = mask / max(float(mask.max()), 1e-6)
        glare = np.ones_like(x)
        y = x * (1 - alpha * mask[..., None]) + glare * alpha * mask[..., None]

    elif noise_id == "N04":
        size = int(cfg["kernel"][idx])
        angle = float(rng.uniform(0, 180))
        y = cv2.filter2D(x, -1, _motion_kernel(size, angle), borderType=cv2.BORDER_REFLECT101)

    elif noise_id == "N05":
        radius = int(cfg["radius"][idx])
        y = cv2.filter2D(x, -1, _disc_kernel(radius), borderType=cv2.BORDER_REFLECT101)
        y = cv2.GaussianBlur(y, (0, 0), 0.15 + 0.10 * radius)

    elif noise_id == "N06":
        delta = float(cfg["delta"][idx])
        warm = bool(rng.integers(0, 2))
        # OpenCV BGR order.
        gains = np.array([1 - delta, 1.0, 1 + delta] if warm else [1 + delta, 1.0, 1 - delta], np.float32)
        y = x * gains.reshape(1, 1, 3)
        y /= max(float(y.mean()), 1e-6) / max(float(x.mean()), 1e-6)

    elif noise_id == "N07":
        quality = int(cfg["quality"][idx])
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG decoding failed")
        return decoded

    elif noise_id == "N08":
        scale = float(cfg["scale"][idx])
        small = cv2.resize(x, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        y = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    elif noise_id == "N09":
        darkness = float(cfg["darkness"][idx])
        mask_small = np.zeros((max(8, h // 16), max(8, w // 16)), np.float32)
        pts = np.stack([
            rng.integers(0, mask_small.shape[1], size=6),
            rng.integers(0, mask_small.shape[0], size=6),
        ], axis=1).astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask_small, hull, 1.0)
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = cv2.GaussianBlur(mask, (0, 0), max(h, w) * 0.035)
        mask = np.clip(mask, 0, 1)
        y = x * (1.0 - darkness * mask[..., None])

    elif noise_id == "N10":
        coverage = float(cfg["coverage"][idx])
        sigma = float(cfg["sigma"][idx])
        mask = np.zeros((h, w), np.float32)
        area_target = coverage * h * w
        current = 0.0
        while current < area_target:
            ax = max(10, int(rng.uniform(0.05, 0.16) * w))
            ay = max(10, int(rng.uniform(0.04, 0.14) * h))
            cx = int(rng.uniform(0, w))
            cy = int(rng.uniform(0, h))
            cv2.ellipse(mask, (cx, cy), (ax, ay), float(rng.uniform(0, 180)), 0, 360, 1.0, -1, cv2.LINE_AA)
            current += np.pi * ax * ay
        mask = cv2.GaussianBlur(mask, (0, 0), max(3.0, sigma * 0.8))
        mask = np.clip(mask, 0, 1)
        blurred = cv2.GaussianBlur(x, (0, 0), sigma)
        veil = np.full_like(x, float(rng.uniform(0.55, 0.75)))
        local = 0.72 * blurred + 0.28 * veil
        y = x * (1 - mask[..., None]) + local * mask[..., None]

    else:
        raise AssertionError(noise_id)

    return _u8(y * 255.0)
