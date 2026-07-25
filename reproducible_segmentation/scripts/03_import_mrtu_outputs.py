"""Import compact, reviewable expanded-dataset results from output ZIPs.

The script keeps the original ZIPs in outputs/ untouched. It extracts CSV
learning histories and run arguments for every run, while retaining figures only
for the selected strong runs. Checkpoints and notebooks are not copied into the
reproducibility layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SOURCE_DIR = REPO_ROOT / "outputs"
ARTIFACT_ROOT = ROOT / "artifacts"
DATASET_ID = "expanded_dataset_mrtu_v1"
DATASET_LABEL = "Expanded dataset (ShrimpDisBD-TigerShrimp_MrTuDat v1)"

SOURCES = (
    {
        "zip_name": "yolo11n_simam_ca_new_data_leakage.zip",
        "candidate_id": "simam_ca_strong",
        "selected_runs": {"simam_ca_mixed_split_run_simam_ca_strong"},
    },
    {
        "zip_name": "yolo11n_seg_dpca_p4_new_data_leakage.zip",
        "candidate_id": "dpca_p4_strong",
        "selected_runs": {
            "yolo11n_seg_dpca_p4_new_data_leakage/yolo11n_seg_dpca_p4_strong_200e"
        },
    },
    {
        "zip_name": "yolo11n_seg_dpca_p3p4p5_new_data_leakage.zip",
        "candidate_id": "dpca_p3p4p5_strong",
        "selected_runs": {
            "yolo11n_seg_dpca_p3p4p5_new_data_leakage/yolo11n_seg_dpca_p3p4p5_strong_200e"
        },
    },
    {
        "zip_name": "yolo11n_seg_cote_boundarylite_p4_strong_new_data_leakage.zip",
        "candidate_id": "cote_boundarylite_p4_strong",
        "selected_runs": {
            "yolo11n_seg_cote_boundarylite_p4_strong_new_data_leakage/yolo11n_seg_cote_boundary_p4_strong_200e"
        },
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_entry(archive: zipfile.ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8-sig")


def write_entry(archive: zipfile.ZipFile, source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(source) as input_stream, destination.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)


def matching_arg_entry(names: set[str], run_prefix: str) -> str | None:
    for filename in ("args.yaml", "args.txt"):
        candidate = f"{run_prefix}/{filename}"
        if candidate in names:
            return candidate
    return None


def import_archive(source: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    zip_path = SOURCE_DIR / str(source["zip_name"])
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    selected_runs = set(source["selected_runs"])
    candidate_id = str(source["candidate_id"])
    zip_stem = zip_path.stem
    raw_root = ARTIFACT_ROOT / "raw_runs" / zip_stem
    figure_root = ARTIFACT_ROOT / "figures" / "mrtu_selected"
    rows: list[dict[str, object]] = []

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        result_entries = sorted(name for name in names if name.endswith("/results.csv"))
        for result_entry in result_entries:
            run_prefix = result_entry.removesuffix("/results.csv")
            run_name = run_prefix.rsplit("/", 1)[-1]
            selected = run_prefix in selected_runs
            raw_destination = raw_root / run_name
            write_entry(archive, result_entry, raw_destination / "results.csv")

            args_entry = matching_arg_entry(names, run_prefix)
            if args_entry:
                write_entry(archive, args_entry, raw_destination / Path(args_entry).name)

            history = list(csv.DictReader(read_entry(archive, result_entry).splitlines()))
            if not history:
                raise ValueError(f"No metric rows in {zip_path.name}:{result_entry}")
            final = history[-1]
            row: dict[str, object] = {
                "dataset_id": DATASET_ID,
                "dataset_label": DATASET_LABEL,
                "candidate_id": candidate_id,
                "run_name": run_name,
                "augmentation": "strong" if "strong" in run_name else "light",
                "selected_for_review": str(selected).lower(),
                "completed_epoch": final["epoch"],
                "elapsed_seconds": final["time"],
                "box_map50": final["metrics/mAP50(B)"],
                "box_map50_95": final["metrics/mAP50-95(B)"],
                "mask_precision": final["metrics/precision(M)"],
                "mask_recall": final["metrics/recall(M)"],
                "mask_map50": final["metrics/mAP50(M)"],
                "mask_map50_95": final["metrics/mAP50-95(M)"],
                "source_zip": zip_path.name,
                "source_run": run_prefix,
                "raw_results_csv": (raw_destination / "results.csv").relative_to(ROOT).as_posix(),
            }
            rows.append(row)

            if selected:
                review_destination = figure_root / candidate_id / run_name
                for figure in ("results.png", "confusion_matrix_normalized.png"):
                    figure_entry = f"{run_prefix}/{figure}"
                    if figure_entry in names:
                        write_entry(archive, figure_entry, review_destination / figure)

                extra_prefix = f"{run_prefix}/extra_test_metrics/"
                for extra_entry in sorted(
                    name
                    for name in names
                    if name.startswith(extra_prefix) and name.endswith(".csv")
                ):
                    write_entry(
                        archive,
                        extra_entry,
                        review_destination / "extra_test_metrics" / Path(extra_entry).name,
                    )

    manifest = {
        "zip": zip_path.name,
        "sha256": sha256(zip_path),
        "candidate_id": candidate_id,
        "runs_imported": len(rows),
        "selected_runs": sorted(selected_runs),
    }
    return rows, manifest


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for source in SOURCES:
        imported_rows, manifest = import_archive(source)
        rows.extend(imported_rows)
        manifests.append(manifest)

    rows.sort(key=lambda item: (str(item["candidate_id"]), str(item["run_name"])))
    table_path = ARTIFACT_ROOT / "tables" / "mrtu_training_final_metrics.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected_path = ARTIFACT_ROOT / "tables" / "mrtu_selected_strong_runs.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(row for row in rows if row["selected_for_review"] == "true")

    manifest_path = ARTIFACT_ROOT / "metadata" / "mrtu_output_import.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source_directory": "outputs/",
                "dataset_id": DATASET_ID,
                "dataset_label": DATASET_LABEL,
                "selection_rule": "Strong run with the highest completed epoch in each imported package.",
                "source_packages": manifests,
                "notes": [
                    "Metrics are final Ultralytics validation metrics from results.csv.",
                    "They are not a replacement for a separately documented held-out test evaluation.",
                    "The imported runs are results on the expanded dataset.",
                    "Weights and notebooks remain only in the original ZIP packages.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Imported {len(rows)} runs.")
    print(table_path)
    print(selected_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
