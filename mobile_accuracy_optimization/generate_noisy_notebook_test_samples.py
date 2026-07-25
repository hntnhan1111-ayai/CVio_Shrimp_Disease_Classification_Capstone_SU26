#!/usr/bin/env python3
"""Create mobile-test samples using the top five SimAM+CA strong noise conditions.

The source is the ``lets-try-this/shrimpdishandsegv2`` Roboflow dataset,
version 1, downloaded as ``yolo26`` in ``cote-gate-strong-augmentation.ipynb``.
The notebook first creates a shrimp-grouped, disease-stratified split with
seed 42. This script recreates that split without moving source images and
materializes a small subset for mobile testing:

    12 representative test images (3 per disease) x 5 noise conditions
    = 60 lossless PNG files.

The selected conditions, their transforms, and their ranking are copied from
the SimAM+CA + strong-augmentation noise ablation. They are not the five
paper-corruption presets used by ``generate_noisy_test_samples.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SEED = 42
PARAMETER_SOURCE_NOTEBOOK = "yolov11n-simam-noise-ablation-v3 (1).ipynb"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_TEST_COUNTS = {"BG": 24, "Healthy": 41, "WSSV": 38, "WSSV_BG": 26}
SHRIMP_NAME_PATTERN = re.compile(
    r"^(?P<disease>Healthy|BG|WSSV_BG|WSSV)-(?P<shrimp_id>.+)-img-(?P<img_num>\d+)$",
    re.IGNORECASE,
)

# Ranked by labeled_test_mask_map50 from the YOLO11n-seg + SimAM + CA run
# trained with the strong augmentation configuration.
SIMAM_CA_STRONG_TOP5 = (
    {
        "noise_id": "N06_gauss_blur",
        "rank": 1,
        "labeled_test_mask_map50": 0.576253,
        "description": "Gaussian blur (out-of-focus movement)",
        "parameters": {"sigma": 2.5, "kernel_size": 17},
    },
    {
        "noise_id": "N03_jpeg_artifact",
        "rank": 2,
        "labeled_test_mask_map50": 0.563891,
        "description": "JPEG compression artifact",
        "parameters": {"jpeg_quality": 15},
    },
    {
        "noise_id": "N04_color_cast",
        "rank": 3,
        "labeled_test_mask_map50": 0.546478,
        "description": "Blue color cast",
        "parameters": {"channel": "blue", "bgr_channel_index": 0, "shift": 40},
    },
    {
        "noise_id": "N09_overexposure",
        "rank": 4,
        "labeled_test_mask_map50": 0.524561,
        "description": "Overexposure in HSV value channel",
        "parameters": {"hsv_v_shift": 0.4, "brightness_multiplier": 1.4},
    },
    {
        "noise_id": "N08_motion_blur",
        "rank": 5,
        "labeled_test_mask_map50": 0.512273,
        "description": "Motion blur with a random direction",
        "parameters": {"kernel_size": 11, "angle_range_degrees": [0, 180]},
    },
)
NOISE_BY_ID = {entry["noise_id"]: entry for entry in SIMAM_CA_STRONG_TOP5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_roboflow_stem(stem: str) -> str:
    """Recover the original filename stem from a Roboflow YOLO export."""
    stem = re.sub(r"_(jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-f]+$", "", stem, flags=re.I)
    return re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.I)


def parse_source(path: Path) -> tuple[str, str]:
    """Return the original disease and a no-leakage shrimp group key."""
    match = SHRIMP_NAME_PATTERN.match(normalize_roboflow_stem(path.stem))
    if not match:
        raise ValueError(f"Cannot infer disease/shrimp ID from source filename: {path.name}")
    disease = match.group("disease")
    canonical_disease = next(
        name for name in EXPECTED_TEST_COUNTS if name.lower() == disease.lower()
    )
    return canonical_disease, f"{canonical_disease.lower()}::{match.group('shrimp_id')}"


def split_one_stratum(items: list[tuple[str, list[Path]]]) -> tuple[list, list, list]:
    """Use precisely the count logic from the notebook's grouped split cell."""
    count = len(items)
    train_count = int(0.80 * count)
    valid_count = int(0.10 * count)
    test_count = count - train_count - valid_count
    if count >= 3:
        if valid_count == 0:
            valid_count = 1
            train_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1
    if train_count < 1 and count > 0:
        train_count = 1
    while train_count + valid_count + test_count > count:
        train_count -= 1
    test_count = count - train_count - valid_count
    return (
        items[:train_count],
        items[train_count : train_count + valid_count],
        items[train_count + valid_count :],
    )


def recreate_notebook_test_split(dataset_root: Path) -> list[tuple[int, str, str, Path]]:
    """Recreate the seed-42 grouped test split from the immutable train export."""
    source_dir = dataset_root / "train" / "images"
    source_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not source_paths:
        raise FileNotFoundError(f"No source images found in {source_dir}")

    groups: dict[str, list[Path]] = defaultdict(list)
    group_disease: dict[str, str] = {}
    for source_path in source_paths:
        disease, group_key = parse_source(source_path)
        groups[group_key].append(source_path)
        group_disease[group_key] = disease

    strata: dict[str, list[tuple[str, list[Path]]]] = defaultdict(list)
    for group_key, images in groups.items():
        strata[group_disease[group_key]].append((group_key, sorted(images)))

    rng = random.Random(SEED)
    test_groups: list[tuple[str, list[Path]]] = []
    for disease in sorted(strata):
        stratum = sorted(strata[disease], key=lambda item: item[0])
        rng.shuffle(stratum)
        _, _, test_items = split_one_stratum(stratum)
        test_groups.extend(test_items)

    test_records: list[tuple[int, str, str, Path]] = []
    for group_key, images in sorted(test_groups, key=lambda item: item[0]):
        disease = group_disease[group_key]
        for source_path in images:
            test_records.append((0, disease, group_key, source_path))
    test_records.sort(key=lambda record: record[3].name)
    test_records = [
        (index, disease, group_key, source_path)
        for index, (_, disease, group_key, source_path) in enumerate(test_records)
    ]

    actual_counts = dict(sorted(Counter(record[1] for record in test_records).items()))
    if len(test_records) != 129 or actual_counts != EXPECTED_TEST_COUNTS:
        raise RuntimeError(
            "The recreated split does not match the notebook output: "
            f"found {len(test_records)} images with {actual_counts}."
        )
    return test_records


def choose_representative_sources(
    test_records: list[tuple[int, str, str, Path]], samples_per_disease: int
) -> list[tuple[int, str, str, Path]]:
    selected: list[tuple[int, str, str, Path]] = []
    for disease in sorted(EXPECTED_TEST_COUNTS):
        disease_records = sorted(
            (record for record in test_records if record[1] == disease),
            key=lambda record: record[3].name,
        )
        if len(disease_records) < samples_per_disease:
            raise ValueError(f"Not enough {disease} images for {samples_per_disease} samples.")
        positions = [
            round(index * (len(disease_records) - 1) / (samples_per_disease - 1))
            for index in range(samples_per_disease)
        ]
        selected.extend(disease_records[position] for position in positions)
    return selected


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV cannot read source image: {path}")
    return image


def advance_rng_to_motion_blur(
    test_records: list[tuple[int, str, str, Path]], seed: int
) -> dict[Path, float]:
    """Return the same per-image N08 angle sequence used in the ablation notebook.

    The original code shares one RNG across N01, N02, and N07 before it
    creates N08. Replaying those draws preserves its seed-42 motion direction
    for every image in the recreated 129-image test split.
    """
    rng = np.random.default_rng(seed)
    images = [(record[3], read_bgr(record[3])) for record in test_records]

    # N01_gaussian: _rng.normal(0, sigma, img.shape).astype(np.float32)
    for _, image in images:
        rng.normal(0, 20, image.shape).astype(np.float32)

    # N02_salt_pepper: _rng.random(img.shape[:2])
    for _, image in images:
        rng.random(image.shape[:2])

    # N07_heavy_erase: exactly the random calls in the notebook function.
    for _, image in images:
        height, width = image.shape[:2]
        for _ in range(3):
            if rng.random() < 0.5:
                patch_height = int(rng.uniform(0.1, 0.4) * height)
                patch_width = int(rng.uniform(0.1, 0.4) * width)
                rng.uniform(0, height - patch_height)
                rng.uniform(0, width - patch_width)
                rng.integers(100, 150, (patch_height, patch_width, 3), dtype=np.uint8)

    # N08_motion_blur: angle = _rng.uniform(0, 180) for each test image.
    return {path: float(rng.uniform(0, 180)) for path, _ in images}


def apply_noise(image: np.ndarray, noise_id: str, motion_angle: float | None) -> np.ndarray:
    """Apply one selected test-noise condition, matching the ablation code."""
    if noise_id == "N06_gauss_blur":
        return cv2.GaussianBlur(image, (17, 17), 2.5)
    if noise_id == "N03_jpeg_artifact":
        encoded_ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 15]
        )
        if not encoded_ok:
            raise RuntimeError("Could not JPEG-encode image for N03_jpeg_artifact")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if noise_id == "N04_color_cast":
        output = image.astype(np.int32)
        output[:, :, 0] = np.clip(output[:, :, 0] + 40, 0, 255)
        return output.astype(np.uint8)
    if noise_id == "N09_overexposure":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if noise_id == "N08_motion_blur":
        if motion_angle is None:
            raise ValueError("N08_motion_blur needs a precomputed angle")
        kernel_size = 11
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        matrix = cv2.getRotationMatrix2D(
            (kernel_size // 2, kernel_size // 2), motion_angle, 1
        )
        kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
        kernel /= kernel.sum() if kernel.sum() > 0 else 1
        return cv2.filter2D(image, -1, kernel)
    raise ValueError(f"Unsupported noise condition: {noise_id}")


def write_split_manifest(
    output_dir: Path, test_records: list[tuple[int, str, str, Path]], dataset_root: Path
) -> None:
    with (output_dir / "recreated_notebook_test_split.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("test_index", "disease", "shrimp_group", "source_rel_path"),
        )
        writer.writeheader()
        for index, disease, group_key, source_path in test_records:
            writer.writerow(
                {
                    "test_index": index,
                    "disease": disease,
                    "shrimp_group": group_key,
                    "source_rel_path": source_path.relative_to(dataset_root).as_posix(),
                }
            )


def write_readme(output_dir: Path, samples_per_disease: int, seed: int) -> None:
    rows = "\n".join(
        "| {rank} | `{noise_id}` | {description} | {metric:.6f} | `{parameters}` |".format(
            rank=noise["rank"],
            noise_id=noise["noise_id"],
            description=noise["description"],
            metric=noise["labeled_test_mask_map50"],
            parameters=json.dumps(noise["parameters"], ensure_ascii=False),
        )
        for noise in SIMAM_CA_STRONG_TOP5
    )
    total = len(SIMAM_CA_STRONG_TOP5) * len(EXPECTED_TEST_COUNTS) * samples_per_disease
    (output_dir / "README.md").write_text(
        "# Top-5 noisy test samples — SimAM + CA + strong\n\n"
        "The five conditions below are ranked by `labeled_test_mask_map50` from "
        "the **YOLO11n-seg + SimAM + CA** experiment trained with **strong augmentation**. "
        "They replace the previous paper-corruption set in this folder.\n\n"
        f"Parameters were verified directly against `{PARAMETER_SOURCE_NOTEBOOK}`.\n\n"
        "| Rank | Noise ID | Transform | Labeled Test Mask mAP50 | Parameters |\n"
        "| ---: | --- | --- | ---: | --- |\n"
        f"{rows}\n\n"
        "Source: Roboflow `lets-try-this/shrimpdishandsegv2`, version 1, `yolo26` export. "
        "The grouped, disease-stratified seed-42 split was recreated without moving source files: "
        "129 test images (Healthy 41, BG 24, WSSV 38, WSSV_BG 26).\n\n"
        f"This folder contains {total} lossless PNG variants from {samples_per_disease} representative "
        "test images per disease. `recreated_notebook_test_split.csv` records every source in the "
        "recreated test split; `manifest.csv` records each output, ranking metric, transform parameters, "
        "and SHA-256.\n\n"
        f"The N08 motion-blur direction exactly replays the original ablation RNG sequence with seed `{seed}`. "
        "PNG avoids adding another JPEG compression step after a transform. Source images are read-only.\n",
        encoding="utf-8",
    )


def generate(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    staging_dir = output_dir.with_name(f"{output_dir.name}.staging")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output folder is not empty: {output_dir}; pass --overwrite to replace it.")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    test_records = recreate_notebook_test_split(dataset_root)
    selected_records = choose_representative_sources(test_records, args.samples_per_disease)
    motion_angles = advance_rng_to_motion_blur(test_records, args.seed)
    staging_dir.mkdir(parents=True, exist_ok=False)
    write_split_manifest(staging_dir, test_records, dataset_root)

    manifest_rows: list[dict[str, object]] = []
    for test_index, disease, group_key, source_path in selected_records:
        source = read_bgr(source_path)
        for noise in SIMAM_CA_STRONG_TOP5:
            noise_id = noise["noise_id"]
            output_class_dir = staging_dir / noise_id / disease
            output_class_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_class_dir / f"{source_path.stem}__{noise_id}.png"
            noisy = apply_noise(source, noise_id, motion_angles.get(source_path))
            if not cv2.imwrite(str(output_path), noisy, [int(cv2.IMWRITE_PNG_COMPRESSION), 1]):
                raise RuntimeError(f"Could not write {output_path}")
            manifest_rows.append(
                {
                    "source_rel_path": source_path.relative_to(dataset_root).as_posix(),
                    "test_index": test_index,
                    "disease": disease,
                    "shrimp_group": group_key,
                    "noise_id": noise_id,
                    "rank": noise["rank"],
                    "labeled_test_mask_map50": f"{noise['labeled_test_mask_map50']:.6f}",
                    "parameters": json.dumps(noise["parameters"], sort_keys=True),
                    "rng_seed": args.seed,
                    "motion_angle_degrees": (
                        f"{motion_angles[source_path]:.12f}"
                        if noise_id == "N08_motion_blur"
                        else ""
                    ),
                    "output_rel_path": output_path.relative_to(staging_dir).as_posix(),
                    "output_sha256": sha256(output_path),
                }
            )

    with (staging_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_readme(staging_dir, args.samples_per_disease, args.seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging_dir.replace(output_dir)
    print(f"Created {len(manifest_rows)} noisy PNG files in {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root.parent / "tmp" / "notebook_source_data",
        help="Root of the Roboflow yolo26 export used by the notebook.",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "noisy_test_samples")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--samples-per-disease", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(generate(parse_args()))
