"""Dataset download, strict audit, split manifest, and YOLO folder prep."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from . import config
from .utils import ensure_dir, md5_file, save_csv, write_json


def image_files(folder: Path) -> list[Path]:
    return sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS], key=lambda p: p.name)


def looks_like_dataset_root(path: str | Path) -> bool:
    root = Path(path)
    return root.is_dir() and all((root / class_dir).is_dir() for class_dir in config.CLASS_DIRS)


def candidate_dataset_roots(base: str | Path) -> list[Path]:
    base = Path(base)
    names = ["processed-images", "processed_images"]
    candidates = [base]
    for name in names:
        candidates.extend([base / name, base / name / name])
    if base.exists():
        for child in base.iterdir():
            if child.is_dir():
                candidates.append(child)
                for name in names:
                    candidates.extend([child / name, child / name / name])
    return candidates


def resolve_dataset_root(downloaded_path: str | Path) -> tuple[Path, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []
    for candidate in candidate_dataset_roots(downloaded_path):
        resolved = candidate.resolve()
        attempts.append({"candidate": str(resolved), "looks_like_dataset_root": str(looks_like_dataset_root(resolved))})
        if looks_like_dataset_root(resolved):
            return resolved, attempts
    downloaded = Path(downloaded_path)
    if downloaded.exists():
        for marker in downloaded.rglob(config.CLASS_DIRS[0]):
            root = marker.parent
            attempts.append({"candidate": str(root.resolve()), "looks_like_dataset_root": str(looks_like_dataset_root(root))})
            if looks_like_dataset_root(root):
                return root.resolve(), attempts
    raise FileNotFoundError("Could not resolve dataset root containing the four expected class folders.")


def count_class_images(root: str | Path) -> dict[str, int]:
    root = Path(root)
    return {class_dir: len(image_files(root / class_dir)) if (root / class_dir).is_dir() else -1 for class_dir in config.CLASS_DIRS}


def audit_class_counts(root: str | Path) -> dict[str, Any]:
    counts = count_class_images(root)
    named_counts = {config.DIR_TO_NAME[class_dir]: counts[class_dir] for class_dir in config.CLASS_DIRS}
    total = sum(counts.values())
    audit = {
        "dataset_id": config.DATASET_ID,
        "dataset_root": str(Path(root).resolve()),
        "expected_class_counts_by_dir": config.EXPECTED_CLASS_COUNTS,
        "actual_class_counts_by_dir": counts,
        "expected_class_counts_by_name": config.EXPECTED_NAME_COUNTS,
        "actual_class_counts_by_name": named_counts,
        "expected_total_images": config.EXPECTED_TOTAL_IMAGES,
        "actual_total_images": total,
    }
    if counts != config.EXPECTED_CLASS_COUNTS or total != config.EXPECTED_TOTAL_IMAGES:
        raise AssertionError("dataset_count_mismatch " + str(audit))
    return audit


def download_dataset(output_dir: str | Path, dry_run: bool = False) -> tuple[Path | None, dict[str, Any]]:
    paths = config.output_paths(output_dir)
    ensure_dir(paths["output"])
    if dry_run:
        audit = {"dry_run": True, "dataset_id": config.DATASET_ID, "download_call": 'kagglehub.dataset_download("uynnhy/processed-images")'}
        write_json(paths["output"] / "dataset_resolution_dry_run.json", audit)
        return None, audit
    import kagglehub

    downloaded = Path(kagglehub.dataset_download("uynnhy/processed-images"))
    print("Path to dataset files:", downloaded)
    root, attempts = resolve_dataset_root(downloaded)
    audit = audit_class_counts(root)
    audit["downloaded_path"] = str(downloaded)
    audit["resolution_attempts"] = attempts
    write_json(paths["output"] / "dataset_resolution.json", audit)
    return root, audit


def image_info(path: str | Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return width, height, mode


def build_manifest(dataset_root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = Path(dataset_root)
    for class_dir in config.CLASS_DIRS:
        label = config.CLASS_TO_LABEL[class_dir]
        class_name = config.CLASS_NAMES[label]
        for path in image_files(root / class_dir):
            width, height, mode = image_info(path)
            checksum = md5_file(path)
            rows.append({
                "rel_path": path.relative_to(root).as_posix(),
                "class_dir": class_dir,
                "class_name": class_name,
                "label": label,
                "source_path": str(path.resolve()),
                "source_md5": checksum,
                "processed_path": str(path.resolve()),
                "processed_md5": checksum,
                "width": width,
                "height": height,
                "image_mode": mode,
                "file_size_bytes": path.stat().st_size,
            })
    frame = pd.DataFrame(rows).sort_values("rel_path").reset_index(drop=True)
    if len(frame) != config.EXPECTED_TOTAL_IMAGES:
        raise AssertionError(f"manifest_total_mismatch expected={config.EXPECTED_TOTAL_IMAGES} actual={len(frame)}")
    counts = frame.groupby("class_dir").size().to_dict()
    if counts != config.EXPECTED_CLASS_COUNTS:
        raise AssertionError("manifest_class_count_mismatch " + str(counts))
    if not (frame["source_path"] == frame["processed_path"]).all():
        raise AssertionError("processed_path_must_equal_source_path")
    if not (frame["source_md5"] == frame["processed_md5"]).all():
        raise AssertionError("processed_md5_must_equal_source_md5")
    paths = config.output_paths(output_dir)
    save_csv(frame, paths["output"] / "source_processed_manifest_with_md5.csv")
    save_csv(
        frame[["class_dir", "class_name", "label"]].groupby(["class_dir", "class_name", "label"]).size().reset_index(name="count"),
        paths["output"] / "class_distribution.csv",
    )
    return frame


def create_split_manifest(manifest: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    frame = manifest.sort_values("rel_path").reset_index(drop=True).copy()
    labels = frame["label"].to_numpy()
    indices = list(range(len(frame)))
    try:
        from sklearn.model_selection import train_test_split

        train_idx, temp_idx, _y_train, y_temp = train_test_split(
            indices,
            labels,
            train_size=config.TRAIN_RATIO,
            random_state=config.SPLIT_SEED,
            stratify=labels,
            shuffle=True,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=config.SPLIT_SEED,
            stratify=y_temp,
            shuffle=True,
        )
    except ModuleNotFoundError:
        train_idx, val_idx, test_idx = fallback_stratified_split(frame)
    split = [""] * len(frame)
    for idx in train_idx:
        split[idx] = "train"
    for idx in val_idx:
        split[idx] = "val"
    for idx in test_idx:
        split[idx] = "test"
    frame["split"] = split
    if frame.groupby("source_md5")["split"].nunique().gt(1).any():
        overlap = frame.groupby("source_md5")["split"].nunique()
        raise AssertionError("split_overlap_by_source_md5 " + str(overlap[overlap > 1].to_dict()))
    columns = ["rel_path", "class_dir", "class_name", "label", "split", "source_path", "source_md5", "processed_path", "processed_md5"]
    split_frame = frame[columns].sort_values(["split", "label", "rel_path"]).reset_index(drop=True)
    paths = config.output_paths(output_dir)
    save_csv(split_frame, paths["output"] / "fixed_split_manifest_seed42_with_md5.csv")
    save_csv(
        split_frame.groupby(["split", "class_name"]).size().unstack(fill_value=0).reset_index(),
        paths["output"] / "split_class_distribution.csv",
    )
    return split_frame


def fallback_stratified_split(frame: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    """Local fallback for import validation when scikit-learn is unavailable."""
    import random

    rng = random.Random(config.SPLIT_SEED)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for _label, class_frame in frame.groupby("label"):
        class_indices = class_frame.index.tolist()
        rng.shuffle(class_indices)
        train_n = round(len(class_indices) * config.TRAIN_RATIO)
        val_n = round(len(class_indices) * config.VAL_RATIO)
        train_idx.extend(class_indices[:train_n])
        val_idx.extend(class_indices[train_n:train_n + val_n])
        test_idx.extend(class_indices[train_n + val_n:])
    return train_idx, val_idx, test_idx


def prepare_yolo_dataset(split_manifest: pd.DataFrame, output_dir: str | Path, force_rebuild: bool = False) -> pd.DataFrame:
    paths = config.output_paths(output_dir)
    yolo_root = paths["yolo_dataset"]
    manifest_path = paths["output"] / "yolo_split_manifest_seed42_with_md5.csv"
    if force_rebuild and yolo_root.exists():
        shutil.rmtree(yolo_root)
    rows: list[dict[str, Any]] = []
    for row in split_manifest.sort_values("rel_path").itertuples(index=False):
        source = Path(row.source_path)
        safe_name = row.rel_path.replace("/", "__").replace("\\", "__").replace(" ", "_")
        dest = yolo_root / row.split / row.class_name / f"{row.source_md5[:10]}__{safe_name}"
        if not dest.exists():
            ensure_dir(dest.parent)
            shutil.copy2(source, dest)
        yolo_md5 = md5_file(dest)
        if yolo_md5 != row.source_md5:
            raise AssertionError(f"yolo_copy_md5_mismatch {source} -> {dest}")
        item = row._asdict()
        item.update({"yolo_path": str(dest.resolve()), "yolo_md5": yolo_md5})
        rows.append(item)
    yolo_manifest = pd.DataFrame(rows).sort_values(["split", "label", "rel_path"]).reset_index(drop=True)
    if yolo_manifest["yolo_path"].duplicated().any():
        raise AssertionError("duplicated_yolo_paths")
    if not (yolo_manifest["source_md5"] == yolo_manifest["yolo_md5"]).all():
        raise AssertionError("yolo_md5_mismatch")
    save_csv(yolo_manifest, manifest_path)
    return yolo_manifest


def prepare_all(output_dir: str | Path, dry_run: bool = False, force_rebuild_yolo: bool = False) -> dict[str, Any]:
    paths = config.output_paths(output_dir)
    for path in paths.values():
        ensure_dir(path)
    root, audit = download_dataset(output_dir, dry_run=dry_run)
    if dry_run:
        return {"dry_run": True, "output_dir": str(paths["output"]), "dataset_audit": audit}
    assert root is not None
    manifest = build_manifest(root, output_dir)
    split_manifest = create_split_manifest(manifest, output_dir)
    yolo_manifest = prepare_yolo_dataset(split_manifest, output_dir, force_rebuild=force_rebuild_yolo)
    return {
        "dataset_root": str(root),
        "manifest_rows": len(manifest),
        "split_rows": len(split_manifest),
        "yolo_rows": len(yolo_manifest),
        "output_dir": str(paths["output"]),
    }


def load_split_manifest(output_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(config.output_paths(output_dir)["output"] / "fixed_split_manifest_seed42_with_md5.csv")


def load_yolo_manifest(output_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(config.output_paths(output_dir)["output"] / "yolo_split_manifest_seed42_with_md5.csv")
