#!/usr/bin/env python3
"""Materialize representative mobile-test images with the paper's top-5 noises.

The paper evaluates these corruptions in memory only.  This utility creates a
small, reproducible set of files for manual mobile-app testing without
modifying the clean test split.  It uses the same five corruption functions,
severity values, and seed convention as the paper branch:

    paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp

Default output: 12 source images (three deterministic examples per class) x
five corruptions at severity 2 = 60 PNG files, plus a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
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

PAPER_PARAMETERS: dict[str, dict[int, float]] = {
    "impulse_noise": {1: 0.015, 2: 0.035, 3: 0.070},
    "gaussian_noise": {1: 0.04, 2: 0.08, 3: 0.14},
    "contrast_reduction": {1: 0.75, 2: 0.55, 3: 0.35},
    "defocus_blur": {1: 1.0, 2: 2.0, 3: 3.0},
    "low_light": {1: 0.70, 2: 0.45, 3: 0.25},
}

PARAMETER_NAMES = {
    "impulse_noise": "amount",
    "gaussian_noise": "sigma",
    "contrast_reduction": "factor",
    "defocus_blur": "radius",
    "low_light": "factor",
}


def apply_paper_corruption(
    image: Image.Image,
    corruption: str,
    severity: int,
    seed: int,
) -> Image.Image:
    """Apply the corresponding corruption implementation from the paper."""
    if corruption not in TOP5_CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")

    source = image.convert("RGB")
    rng = np.random.default_rng(int(seed) + int(severity) * 1000)
    array = np.asarray(source).astype(np.float32) / 255.0

    if corruption == "gaussian_noise":
        sigma = PAPER_PARAMETERS[corruption][severity]
        array = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 1.0)
        return Image.fromarray((array * 255.0).round().astype(np.uint8))

    if corruption == "impulse_noise":
        amount = PAPER_PARAMETERS[corruption][severity]
        mask = rng.random(array.shape[:2]) < amount
        salt = rng.random(array.shape[:2]) < 0.5
        array[mask & salt] = 1.0
        array[mask & ~salt] = 0.0
        return Image.fromarray((array * 255.0).round().astype(np.uint8))

    if corruption == "contrast_reduction":
        return ImageEnhance.Contrast(source).enhance(PAPER_PARAMETERS[corruption][severity])

    if corruption == "defocus_blur":
        return source.filter(
            ImageFilter.GaussianBlur(radius=PAPER_PARAMETERS[corruption][severity])
        )

    return ImageEnhance.Brightness(source).enhance(PAPER_PARAMETERS[corruption][severity])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_representative_sources(
    test_root: Path,
    samples_per_class: int,
) -> list[tuple[int, Path, Path]]:
    """Pick first, middle, and last filenames of every sorted class split.

    The returned index is the stable index of the source in the complete test
    set.  Adding it to seed 42 follows the paper evaluation's seed convention.
    """
    class_dirs = sorted(path for path in test_root.iterdir() if path.is_dir())
    all_sources = [
        image
        for class_dir in class_dirs
        for image in sorted(path for path in class_dir.iterdir() if path.is_file())
    ]
    source_indices = {path: index for index, path in enumerate(all_sources)}
    selected: list[tuple[int, Path, Path]] = []

    for class_dir in class_dirs:
        images = sorted(path for path in class_dir.iterdir() if path.is_file())
        if len(images) < samples_per_class:
            raise ValueError(
                f"{class_dir.name} has only {len(images)} image(s), but "
                f"{samples_per_class} are required."
            )
        positions = [
            round(index * (len(images) - 1) / (samples_per_class - 1))
            for index in range(samples_per_class)
        ]
        for position in positions:
            image = images[position]
            selected.append((source_indices[image], class_dir, image))
    return selected


def write_readme(output_dir: Path, severity: int, samples_per_class: int) -> None:
    total_sources = len(TOP5_CORRUPTIONS) * samples_per_class * 4
    parameter_lines = "\n".join(
        f"- `{name}`: `{PARAMETER_NAMES[name]}` = `{values[severity]}`"
        for name, values in PAPER_PARAMETERS.items()
    )
    (output_dir / "README.md").write_text(
        "# Noisy mobile test samples\n\n"
        f"This folder contains {total_sources} lossless PNG variants from 12 clean test images "
        f"(three per class). Each source has every paper top-5 corruption at severity {severity}.\n\n"
        "The clean source images remain in `../yolo_dataset/test/` and are never modified. "
        "See `manifest.csv` for source paths, seeds, parameter values, and file hashes.\n\n"
        "## Paper-compatible settings\n\n"
        f"{parameter_lines}\n\n"
        "The noise functions and settings match the `paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp` "
        "branch. PNG is used to avoid adding JPEG-compression artifacts after corruption.\n",
        encoding="utf-8",
    )


def generate(args: argparse.Namespace) -> int:
    test_root = args.test_root.resolve()
    output_dir = args.output_dir.resolve()
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test folder not found: {test_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output folder is not empty: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    selected = choose_representative_sources(test_root, args.samples_per_class)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    for source_index, class_dir, source_path in selected:
        with Image.open(source_path) as opened:
            source_image = opened.convert("RGB")
        relative_source = source_path.relative_to(test_root)
        for corruption in TOP5_CORRUPTIONS:
            output_class_dir = output_dir / f"{corruption}_s{args.severity}" / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"{source_path.stem}__{corruption}_s{args.severity}.png"
            output_path = output_class_dir / output_name
            effective_seed = args.seed + source_index
            noisy_image = apply_paper_corruption(
                source_image,
                corruption,
                args.severity,
                effective_seed,
            )
            # Compression changes file size only, not pixels. Level 1 keeps the
            # 60-image mobile demo fast to generate while preserving the exact
            # corruption output.
            noisy_image.save(output_path, format="PNG", compress_level=1)
            manifest_rows.append(
                {
                    "source_rel_path": relative_source.as_posix(),
                    "source_index": source_index,
                    "class_dir": class_dir.name,
                    "corruption": corruption,
                    "severity": args.severity,
                    "seed": effective_seed,
                    "parameter_name": PARAMETER_NAMES[corruption],
                    "parameter_value": PAPER_PARAMETERS[corruption][args.severity],
                    "output_rel_path": output_path.relative_to(output_dir).as_posix(),
                    "output_sha256": sha256(output_path),
                }
            )

    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_readme(output_dir, args.severity, args.samples_per_class)
    print(f"Created {len(manifest_rows)} noisy PNG files in {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-root",
        type=Path,
        default=root / "yolo_dataset" / "test",
        help="Clean mobile-test images. They are read only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "noisy_test_samples",
        help="Folder for generated samples and their manifest.",
    )
    parser.add_argument("--severity", choices=(1, 2, 3), type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a previous generated output folder; clean test images are never changed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(generate(parse_args()))
