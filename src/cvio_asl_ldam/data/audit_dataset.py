"""Audit the downloaded four-class image dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

from cvio_asl_ldam.utils.io import load_yaml, write_json


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_dir(root: Path, aliases: list[str]) -> Path | None:
    for alias in aliases:
        candidate = root / alias
        if candidate.is_dir():
            return candidate
    return None


def looks_like_dataset_root(root: Path, config: dict[str, Any]) -> bool:
    return root.is_dir() and all(
        _class_dir(root, [str(alias) for alias in item["aliases"]]) is not None
        for item in config["classes"]
    )


def candidate_dataset_roots(base: str | Path, max_depth: int = 3) -> list[Path]:
    base_path = Path(base).expanduser().resolve()
    candidates = [base_path]
    if not base_path.exists():
        return candidates
    for path in base_path.rglob("*"):
        if not path.is_dir():
            continue
        try:
            depth = len(path.relative_to(base_path).parts)
        except ValueError:
            continue
        if depth <= max_depth:
            candidates.append(path)
    return candidates


def resolve_dataset_root(base: str | Path, config: dict[str, Any]) -> tuple[Path, list[str]]:
    candidates = candidate_dataset_roots(base)
    for candidate in candidates:
        if looks_like_dataset_root(candidate, config):
            return candidate, [str(path) for path in candidates]
    raise FileNotFoundError(
        "Could not find a folder containing all configured classes. "
        f"Candidates: {[str(path) for path in candidates]}"
    )


def audit_dataset(
    data_dir: str | Path,
    config_path: str | Path,
    verify_images: bool = True,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    config = load_yaml(config_path)
    root, candidates = resolve_dataset_root(data_dir, config)
    extensions = {str(ext).lower() for ext in config["image_extensions"]}
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    unreadable: list[dict[str, str]] = []

    for item in config["classes"]:
        class_name = str(item["name"])
        folder = _class_dir(root, [str(alias) for alias in item["aliases"]])
        if folder is None:
            raise FileNotFoundError(f"Missing class folder for {class_name}")
        files = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        counts[class_name] = len(files)
        for path in files:
            error = ""
            if verify_images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:
                    error = repr(exc)
                    unreadable.append({"rel_path": path.relative_to(root).as_posix(), "error": error})
            rows.append(
                {
                    "rel_path": path.relative_to(root).as_posix(),
                    "filename": path.name,
                    "class_dir": folder.name,
                    "class_name": class_name,
                    "label": int(item["label"]),
                    "md5": md5_file(path) if not error else "",
                    "source_path": str(path.resolve()),
                }
            )

    expected = {str(item["name"]): int(item["total"]) for item in config["classes"]}
    audit = {
        "dataset_id": config["dataset_id"],
        "dataset_root": str(root),
        "candidate_folders": candidates,
        "class_counts": counts,
        "expected_class_counts": expected,
        "total_images": len(rows),
        "unreadable_count": len(unreadable),
        "passed": counts == expected and not unreadable,
    }
    if counts != expected:
        raise AssertionError(f"Dataset class counts differ: expected={expected}, actual={counts}")
    if unreadable:
        audit["unreadable"] = unreadable
        raise AssertionError(f"Found {len(unreadable)} unreadable images")
    return root, rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset/shrimpdiseasebd_seed42.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="artifacts/manifests/dataset_audit.json")
    parser.add_argument("--skip-image-verify", action="store_true")
    args = parser.parse_args()
    _root, _rows, audit = audit_dataset(
        args.data_dir,
        args.config,
        verify_images=not args.skip_image_verify,
    )
    write_json(args.output, audit)
    print(f"Dataset audit passed: {audit['total_images']} images")


if __name__ == "__main__":
    main()
