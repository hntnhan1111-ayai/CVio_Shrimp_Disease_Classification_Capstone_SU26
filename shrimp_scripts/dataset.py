"""Dataset download, strict audit, split manifest, and YOLO folder prep."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from . import config
from .progress import log_event, progress_iter
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


def audit_class_counts(root: str | Path, output_dir: str | Path | None = None, progress_enabled: bool = True) -> dict[str, Any]:
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
        if progress_enabled:
            log_event("Dataset class-count audit failed before fail-fast exit.", level="ERROR", output_dir=output_dir, extra=audit)
        raise AssertionError("dataset_count_mismatch " + str(audit))
    return audit


def download_dataset(output_dir: str | Path, dry_run: bool = False, progress_enabled: bool = True) -> tuple[Path | None, dict[str, Any]]:
    paths = config.output_paths(output_dir)
    ensure_dir(paths["output"])
    if dry_run:
        audit = {"dry_run": True, "dataset_id": config.DATASET_ID, "download_call": 'kagglehub.dataset_download("uynnhy/processed-images")'}
        write_json(paths["output"] / "dataset_resolution_dry_run.json", audit)
        return None, audit
    import kagglehub

    if progress_enabled:
        log_event("Downloading processed dataset with kagglehub.", output_dir=output_dir, extra={"dataset_id": config.DATASET_ID})
    downloaded = Path(kagglehub.dataset_download("uynnhy/processed-images"))
    print("Path to dataset files:", downloaded)
    if progress_enabled:
        log_event("Dataset download completed.", output_dir=output_dir, extra={"downloaded_path": str(downloaded)})
    root, attempts = resolve_dataset_root(downloaded)
    if progress_enabled:
        log_event("Dataset root resolved.", output_dir=output_dir, extra={"dataset_root": str(root)})
    audit = audit_class_counts(root, output_dir=output_dir, progress_enabled=progress_enabled)
    if progress_enabled:
        log_event("Dataset class-count audit passed.", output_dir=output_dir, extra={"class_counts": audit["actual_class_counts_by_name"], "total": audit["actual_total_images"]})
    audit["downloaded_path"] = str(downloaded)
    audit["resolution_attempts"] = attempts
    write_json(paths["output"] / "dataset_resolution.json", audit)
    return root, audit


def image_info(path: str | Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return width, height, mode


def build_manifest(dataset_root: str | Path, output_dir: str | Path, progress_enabled: bool = True) -> pd.DataFrame:
    if progress_enabled:
        log_event("Building source/processed manifest with MD5 hashes.", output_dir=output_dir, extra={"dataset_root": str(dataset_root)})
    rows: list[dict[str, Any]] = []
    root = Path(dataset_root)
    for class_dir in config.CLASS_DIRS:
        label = config.CLASS_TO_LABEL[class_dir]
        class_name = config.CLASS_NAMES[label]
        files = image_files(root / class_dir)
        for path in progress_iter(files, desc=f"manifest {class_name}", total=len(files), enabled=progress_enabled, leave=False):
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
    if progress_enabled:
        log_event("Manifest creation completed.", output_dir=output_dir, extra={"rows": len(frame), "path": str(paths["output"] / "source_processed_manifest_with_md5.csv")})
    return frame


def create_split_manifest(manifest: pd.DataFrame, output_dir: str | Path, progress_enabled: bool = True) -> pd.DataFrame:
    if progress_enabled:
        log_event("Creating fixed stratified train/val/test split.", output_dir=output_dir, extra={"rows": len(manifest), "split_seed": config.SPLIT_SEED})
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
    if progress_enabled:
        log_event("Split manifest completed.", output_dir=output_dir, extra={"rows": len(split_frame), "path": str(paths["output"] / "fixed_split_manifest_seed42_with_md5.csv")})
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


def yolo_class_dir_name(label: int) -> str:
    label = int(label)
    return f"{label:02d}_{config.CLASS_NAMES[label]}"


def expected_yolo_class_dirs() -> list[str]:
    return [yolo_class_dir_name(index) for index in range(config.NUM_CLASSES)]


def discover_yolo_class_dirs(yolo_root: str | Path) -> dict[str, list[str]]:
    root = Path(yolo_root)
    discovered: dict[str, list[str]] = {}
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if split_dir.is_dir():
            discovered[split] = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        else:
            discovered[split] = []
    return discovered


def yolo_tree_has_unexpected_class_dirs(yolo_root: str | Path) -> bool:
    expected = set(expected_yolo_class_dirs())
    discovered = discover_yolo_class_dirs(yolo_root)
    return any(set(class_dirs) - expected for class_dirs in discovered.values())


def prepare_yolo_dataset(split_manifest: pd.DataFrame, output_dir: str | Path, force_rebuild: bool = False, progress_enabled: bool = True) -> pd.DataFrame:
    paths = config.output_paths(output_dir)
    yolo_root = paths["yolo_dataset"]
    manifest_path = paths["output"] / "yolo_split_manifest_seed42_with_md5.csv"
    if progress_enabled:
        log_event("Preparing YOLO classification folder tree.", output_dir=output_dir, extra={"rows": len(split_manifest), "yolo_root": str(yolo_root)})
    if yolo_root.exists() and (force_rebuild or yolo_tree_has_unexpected_class_dirs(yolo_root)):
        shutil.rmtree(yolo_root)
    rows: list[dict[str, Any]] = []
    ordered_rows = list(split_manifest.sort_values("rel_path").itertuples(index=False))
    for row in progress_iter(ordered_rows, desc="copy yolo dataset", total=len(ordered_rows), enabled=progress_enabled, leave=False):
        source = Path(row.source_path)
        safe_name = row.rel_path.replace("/", "__").replace("\\", "__").replace(" ", "_")
        yolo_class_dir = yolo_class_dir_name(row.label)
        dest = yolo_root / row.split / yolo_class_dir / f"{row.source_md5[:10]}__{safe_name}"
        if not dest.exists():
            ensure_dir(dest.parent)
            shutil.copy2(source, dest)
        yolo_md5 = md5_file(dest)
        if yolo_md5 != row.source_md5:
            raise AssertionError(f"yolo_copy_md5_mismatch {source} -> {dest}")
        item = row._asdict()
        item.update({
            "yolo_path": str(dest.resolve()),
            "yolo_md5": yolo_md5,
            "yolo_class_dir": yolo_class_dir,
            "yolo_project_idx": int(row.label),
        })
        rows.append(item)
    yolo_manifest = pd.DataFrame(rows).sort_values(["split", "label", "rel_path"]).reset_index(drop=True)
    if yolo_manifest["yolo_path"].duplicated().any():
        raise AssertionError("duplicated_yolo_paths")
    if not (yolo_manifest["source_md5"] == yolo_manifest["yolo_md5"]).all():
        raise AssertionError("yolo_md5_mismatch")
    discovered = discover_yolo_class_dirs(yolo_root)
    expected_dirs = expected_yolo_class_dirs()
    for split, class_dirs in discovered.items():
        if class_dirs != expected_dirs:
            raise AssertionError(f"yolo_class_dir_mismatch split={split} expected={expected_dirs} actual={class_dirs}")
    save_csv(yolo_manifest, manifest_path)
    if progress_enabled:
        log_event("YOLO dataset preparation completed.", output_dir=output_dir, extra={"rows": len(yolo_manifest), "manifest": str(manifest_path), "class_dirs": discovered})
    return yolo_manifest


def prepare_all(output_dir: str | Path, dry_run: bool = False, force_rebuild_yolo: bool = False, progress_enabled: bool = True) -> dict[str, Any]:
    paths = config.output_paths(output_dir)
    for path in paths.values():
        ensure_dir(path)
    if progress_enabled:
        log_event("Starting dataset preparation.", output_dir=output_dir, extra={"output_dir": str(paths["output"])})
    root, audit = download_dataset(output_dir, dry_run=dry_run, progress_enabled=progress_enabled)
    if dry_run:
        return {"dry_run": True, "output_dir": str(paths["output"]), "dataset_audit": audit}
    assert root is not None
    manifest = build_manifest(root, output_dir, progress_enabled=progress_enabled)
    split_manifest = create_split_manifest(manifest, output_dir, progress_enabled=progress_enabled)
    yolo_manifest = prepare_yolo_dataset(split_manifest, output_dir, force_rebuild=force_rebuild_yolo, progress_enabled=progress_enabled)
    if progress_enabled:
        log_event("Dataset preparation completed.", output_dir=output_dir, extra={
            "manifest": str(paths["output"] / "source_processed_manifest_with_md5.csv"),
            "split_manifest": str(paths["output"] / "fixed_split_manifest_seed42_with_md5.csv"),
            "yolo_manifest": str(paths["output"] / "yolo_split_manifest_seed42_with_md5.csv"),
        })
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
