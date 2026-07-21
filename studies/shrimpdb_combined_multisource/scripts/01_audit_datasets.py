#!/usr/bin/env python3
"""Audit datasets: counts, corrupt images, SHA-256, duplicates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from cvio_asl_ldam.utils.io import load_yaml, write_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")


def _load_study() -> dict[str, Any]:
    return load_yaml(STUDY_CONFIG)


def _load_dataset_overrides(study: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    datasets_cfg = study.get("datasets", {})
    for key in ("shrimpdb", "shrimpdiseasedb"):
        if key in datasets_cfg:
            overrides[key] = datasets_cfg[key]
    return overrides


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> str:
    try:
        with Image.open(path) as image:
            image.verify()
        return ""
    except Exception as exc:
        return repr(exc)


def audit_dataset(
    root: Path,
    class_dirs: dict[str, list[str]],
    expected_counts: dict[str, int],
    extensions: set[str],
    verify: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    for class_name, aliases in class_dirs.items():
        folder = None
        for alias in aliases:
            candidate = root / alias
            if candidate.is_dir():
                folder = candidate
                break
        if folder is None:
            raise FileNotFoundError(f"Missing class folder for {class_name} in {root}")
        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
        counts[class_name] = len(files)
        for path in files:
            error = verify_image(path) if verify else ""
            if error:
                unreadable.append({"rel_path": str(path.relative_to(root)), "error": error})
            rows.append(
                {
                    "rel_path": str(path.relative_to(root)).replace("\\", "/"),
                    "filename": path.name,
                    "class_dir": folder.name,
                    "class_name": class_name,
                    "sha256": sha256_file(path) if not error else "",
                    "error": error,
                }
            )

    audit = {
        "dataset_root": str(root),
        "class_counts": counts,
        "expected_class_counts": expected_counts,
        "total_images": len(rows),
        "unreadable_count": len(unreadable),
        "passed": counts == expected_counts and not unreadable,
    }
    if unreadable:
        audit["unreadable"] = unreadable
    if counts != expected_counts:
        raise AssertionError(f"Counts mismatch: expected={expected_counts} actual={counts}")
    if unreadable:
        raise AssertionError(f"Found {len(unreadable)} unreadable images")
    return rows, audit


def detect_duplicates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_label_removal: list[dict[str, Any]] = []
    cross_label_quarantine: list[dict[str, Any]] = []
    seen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seen[row["sha256"]].append(row)

    for sha, group in seen.items():
        if len(group) <= 1:
            continue
        classes = {item["class_name"] for item in group}
        if len(classes) == 1:
            for item in group[1:]:
                same_label_removal.append(
                    {"sha256": sha, "keep": group[0]["rel_path"], "remove": item["rel_path"], "class_name": item["class_name"]}
                )
        else:
            for item in group:
                cross_label_quarantine.append(
                    {"sha256": sha, "rel_path": item["rel_path"], "class_name": item["class_name"]}
                )
    return same_label_removal, cross_label_quarantine


def run(config_path: str | Path = STUDY_CONFIG) -> int:
    study = _load_study()
    overrides = _load_dataset_overrides(study)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    manifest_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"datasets": {}}

    for key, ds_cfg in overrides.items():
        root = Path(ds_cfg.get("root", f"datasets/{key}"))
        expected_counts = {str(k): int(v) for k, v in ds_cfg.get("expected_counts", {}).items()}
        class_dirs: dict[str, list[str]] = {}
        for cls_name in expected_counts:
            class_dirs[cls_name] = [cls_name]
        if key == "shrimpdb":
            mapping = study.get("experiments", {}).get("shrimpdb3", {}).get("source_mapping", {})
            class_dirs = {mapping.get(k, k): [k] for k in expected_counts}

        LOGGER.info("Auditing %s at %s", key, root)
        rows, audit = audit_dataset(root, class_dirs, expected_counts, extensions)
        same_label, cross_label = detect_duplicates(rows)
        audit["same_label_duplicates"] = same_label
        audit["cross_label_duplicates"] = cross_label
        metadata["datasets"][key] = audit
        for row in rows:
            row["dataset"] = key
            manifest_rows.append(row)

    meta_path = Path("artifacts/metadata/dataset_audit.json")
    manifest_path = Path("artifacts/manifests/dataset_manifest.csv")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(meta_path, metadata)
    if manifest_rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

    LOGGER.info("Wrote audit to %s and manifest to %s", meta_path, manifest_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit study datasets")
    parser.add_argument("--config", default=str(STUDY_CONFIG), help="Path to study.yaml")
    args = parser.parse_args()
    try:
        return run(args.config)
    except Exception as exc:
        LOGGER.error("Audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
