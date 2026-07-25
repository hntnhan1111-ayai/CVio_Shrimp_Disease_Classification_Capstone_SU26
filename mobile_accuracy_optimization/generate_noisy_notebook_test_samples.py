#!/usr/bin/env python3
"""Create mobile-test noise samples from the Roboflow dataset used by the notebook.

The source is the ``lets-try-this/shrimpdishandsegv2`` Roboflow dataset,
version 1, downloaded as ``yolo26`` in
``cote-gate-strong-augmentation.ipynb``.  The notebook first creates a
shrimp-grouped, disease-stratified split with seed 42.  This script recreates
that split *without moving any source image* and materializes a small subset
for mobile testing:

    12 representative test images (3 per disease) x 5 top-5 corruptions
    at severity 2 = 60 lossless PNG files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from generate_noisy_test_samples import (
    PAPER_PARAMETERS,
    PARAMETER_NAMES,
    TOP5_CORRUPTIONS,
    apply_paper_corruption,
    sha256,
)


SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_TEST_COUNTS = {"BG": 24, "Healthy": 41, "WSSV": 38, "WSSV_BG": 26}
SHRIMP_NAME_PATTERN = re.compile(
    r"^(?P<disease>Healthy|BG|WSSV_BG|WSSV)-(?P<shrimp_id>.+)-img-(?P<img_num>\d+)$",
    re.IGNORECASE,
)


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
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
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
            (record for record in test_records if record[1] == disease), key=lambda record: record[3].name
        )
        if len(disease_records) < samples_per_disease:
            raise ValueError(f"Not enough {disease} images for {samples_per_disease} samples.")
        positions = [
            round(index * (len(disease_records) - 1) / (samples_per_disease - 1))
            for index in range(samples_per_disease)
        ]
        selected.extend(disease_records[position] for position in positions)
    return selected


def write_split_manifest(output_dir: Path, test_records: list[tuple[int, str, str, Path]], dataset_root: Path) -> None:
    with (output_dir / "recreated_notebook_test_split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("test_index", "disease", "shrimp_group", "source_rel_path"))
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


def write_readme(output_dir: Path, severity: int, samples_per_disease: int) -> None:
    settings = "\n".join(
        f"- `{noise}`: `{PARAMETER_NAMES[noise]}` = `{PAPER_PARAMETERS[noise][severity]}`"
        for noise in TOP5_CORRUPTIONS
    )
    total = len(TOP5_CORRUPTIONS) * len(EXPECTED_TEST_COUNTS) * samples_per_disease
    (output_dir / "README.md").write_text(
        "# Noisy test samples from the CoTE-Gate notebook dataset\n\n"
        "Source: Roboflow `lets-try-this/shrimpdishandsegv2`, version 1, `yolo26` export. "
        "The grouped, disease-stratified seed-42 split was recreated without moving source files: "
        "129 test images (Healthy 41, BG 24, WSSV 38, WSSV_BG 26).\n\n"
        f"This folder contains {total} lossless PNG variants from {samples_per_disease} representative "
        f"test images per disease, using all five paper corruptions at severity {severity}.\n\n"
        "`recreated_notebook_test_split.csv` records every source in the recreated test split; "
        "`manifest.csv` records each generated PNG, its source, split index, seed, parameter, and SHA-256.\n\n"
        "## Paper-compatible settings\n\n"
        f"{settings}\n\n"
        "PNG avoids adding JPEG-compression artifacts after corruption. Source images are read-only.\n",
        encoding="utf-8",
    )


def generate(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output folder is not empty: {output_dir}; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_records = recreate_notebook_test_split(dataset_root)
    selected_records = choose_representative_sources(test_records, args.samples_per_disease)
    write_split_manifest(output_dir, test_records, dataset_root)

    manifest_rows: list[dict[str, object]] = []
    for test_index, disease, group_key, source_path in selected_records:
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        for corruption in TOP5_CORRUPTIONS:
            output_class_dir = output_dir / f"{corruption}_s{args.severity}" / disease
            output_class_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_class_dir / f"{source_path.stem}__{corruption}_s{args.severity}.png"
            effective_seed = args.seed + test_index
            noisy = apply_paper_corruption(source, corruption, args.severity, effective_seed)
            noisy.save(output_path, format="PNG", compress_level=1)
            manifest_rows.append(
                {
                    "source_rel_path": source_path.relative_to(dataset_root).as_posix(),
                    "test_index": test_index,
                    "disease": disease,
                    "shrimp_group": group_key,
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
    write_readme(output_dir, args.severity, args.samples_per_disease)
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
    parser.add_argument("--severity", choices=(1, 2, 3), type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--samples-per-disease", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(generate(parse_args()))
