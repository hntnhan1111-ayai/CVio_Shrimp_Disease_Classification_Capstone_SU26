"""The official top-5 image corruptions used by the paper."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


TOP5_CORRUPTIONS = (
    "impulse_noise",
    "gaussian_noise",
    "contrast_reduction",
    "defocus_blur",
    "low_light",
)


def _as_pil(image: Any) -> tuple[Image.Image, bool]:
    if isinstance(image, Image.Image):
        return image.convert("RGB"), True
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = np.clip(array * 255.0, 0, 255)
        array = array.astype(np.uint8)
    return Image.fromarray(array).convert("RGB"), False


def apply_corruption(
    image: Any,
    corruption: str,
    severity: int,
    seed: int = 0,
) -> Any:
    if corruption not in TOP5_CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    pil_image, return_pil = _as_pil(image)
    rng = np.random.default_rng(int(seed) + int(severity) * 1000)
    array = np.asarray(pil_image).astype(np.float32) / 255.0

    if corruption == "gaussian_noise":
        sigma = {1: 0.04, 2: 0.08, 3: 0.14}[severity]
        array = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 1.0)
        output = Image.fromarray((array * 255.0).round().astype(np.uint8))
    elif corruption == "impulse_noise":
        amount = {1: 0.015, 2: 0.035, 3: 0.070}[severity]
        mask = rng.random(array.shape[:2]) < amount
        salt = rng.random(array.shape[:2]) < 0.5
        array[mask & salt] = 1.0
        array[mask & ~salt] = 0.0
        output = Image.fromarray((array * 255.0).round().astype(np.uint8))
    elif corruption == "contrast_reduction":
        output = ImageEnhance.Contrast(pil_image).enhance(
            {1: 0.75, 2: 0.55, 3: 0.35}[severity]
        )
    elif corruption == "defocus_blur":
        output = pil_image.filter(
            ImageFilter.GaussianBlur(radius={1: 1.0, 2: 2.0, 3: 3.0}[severity])
        )
    else:
        output = ImageEnhance.Brightness(pil_image).enhance(
            {1: 0.70, 2: 0.45, 3: 0.25}[severity]
        )
    return output if return_pil else np.asarray(output)
