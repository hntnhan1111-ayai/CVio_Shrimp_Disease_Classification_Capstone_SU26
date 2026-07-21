"""Create or materialize the fixed seed-42 Stage-1 split."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from cvio_asl_ldam.data.audit_dataset import audit_dataset
from cvio_asl_ldam.utils.io import load_yaml, write_json


SPLITS = ("train", "val", "test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty split manifest")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reference_mapping(path: Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    for row in _read_csv(path):
        mapping[(row["class_name"], row["filename"])] = row["split"]
    return mapping


def _stratified_split(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("scikit-learn is required when the reference manifest is unavailable") from exc

    indices = list(range(len(rows)))
    labels = [int(row["label"]) for row in rows]
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    temp_labels = [labels[index] for index in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=seed,
        shuffle=True,
    )
    assignments = {
        **{index: "train" for index in train_idx},
        **{index: "val" for index in val_idx},
        **{index: "test" for index in test_idx},
    }
    return [{**row, "split": assignments[index]} for index, row in enumerate(rows)]


def _validate_split(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    expected: dict[str, dict[str, int]] = {
        split: {str(item["name"]): int(item[split]) for item in config["classes"]}
        for split in SPLITS
    }
    actual = {
        split: {
            str(item["name"]): sum(
                row["split"] == split and row["class_name"] == item["name"] for row in rows
            )
            for item in config["classes"]
        }
        for split in SPLITS
    }
    paths = {split: {row["rel_path"] for row in rows if row["split"] == split} for split in SPLITS}
    hashes = {split: {row["md5"] for row in rows if row["split"] == split} for split in SPLITS}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if not paths[left].isdisjoint(paths[right]):
            raise AssertionError(f"Path leakage between {left} and {right}")
        if "" not in hashes[left] and "" not in hashes[right] and not hashes[left].isdisjoint(hashes[right]):
            raise AssertionError(f"Content-hash leakage between {left} and {right}")
    if actual != expected:
        raise AssertionError(f"Split distribution mismatch: expected={expected}, actual={actual}")
    return {
        "seed": int(config["seed"]),
        "split_protocol": config["split_protocol"],
        "distribution": actual,
        "total": len(rows),
        "no_path_leakage": True,
        "no_md5_leakage": True,
    }


def create_split(
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    _root, source_rows, dataset_audit = audit_dataset(data_dir, config_path)
    reference_path = Path(config["reference_manifest"])
    if not reference_path.is_absolute():
        reference_path = (Path.cwd() / reference_path).resolve()
    reference = _reference_mapping(reference_path)

    if reference:
        missing: list[tuple[str, str]] = []
        split_rows = []
        for row in source_rows:
            key = (str(row["class_name"]), str(row["filename"]))
            split = reference.get(key)
            if split is None:
                missing.append(key)
            else:
                split_rows.append({**row, "split": split})
        if missing:
            raise AssertionError(f"Reference manifest did not match {len(missing)} images")
        source = "curated_reference_manifest"
    else:
        split_rows = _stratified_split(source_rows, seed)
        source = "regenerated_stratified_split"

    split_rows = sorted(split_rows, key=lambda row: (SPLITS.index(row["split"]), row["label"], row["rel_path"]))
    portable_rows = [
        {
            "rel_path": row["rel_path"],
            "filename": row["filename"],
            "class_dir": row["class_dir"],
            "class_name": row["class_name"],
            "label": row["label"],
            "md5": row["md5"],
            "split": row["split"],
        }
        for row in split_rows
    ]
    audit = _validate_split(portable_rows, config)
    audit.update({"manifest_source": source, "dataset_audit": dataset_audit})
    output = Path(output_dir)
    manifest_path = output / "split_manifest_seed42_generated.csv"
    _write_csv(manifest_path, portable_rows)
    write_json(output / "split_audit_seed42.json", audit)
    return manifest_path


def materialize_split_tree(
    dataset_root: str | Path,
    manifest_path: str | Path,
    output_root: str | Path,
) -> Path:
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    rows = _read_csv(Path(manifest_path))
    for row in rows:
        source = dataset_root / row["rel_path"]
        class_dir = f"{int(row['label']):02d}_{row['class_name']}"
        destination = output_root / row["split"] / class_dir / row["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = create_split(args.config, args.data_dir, args.output_dir, args.seed)
    print(f"Wrote fixed split manifest: {path}")


if __name__ == "__main__":
    main()
