"""Dataset/protocol catalog without committing raw images or labels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProtocol:
    identifier: str
    display_name: str
    status: str
    class_names: tuple[str, ...]
    image_count: int | None
    required_input: str
    notes: str


DATASETS: tuple[DatasetProtocol, ...] = (
    DatasetProtocol(
        identifier="legacy_129_test",
        display_name="Legacy report protocol (129-image test)",
        status="historical_only",
        class_names=("BG", "WSSV"),
        image_count=None,
        required_input="Original data.yaml and immutable split manifest are not present in this workspace.",
        notes="The report records 129 test images: 41 healthy negatives and 88 disease/labeled images.",
    ),
    DatasetProtocol(
        identifier="mrtu_v1",
        display_name="Expanded dataset - ShrimpDisBD-TigerShrimp_MrTuDat v1",
        status="audit_available",
        class_names=("BG", "WSSV"),
        image_count=1452,
        required_input="External YOLO data.yaml pointing to the preserved train/valid/test export.",
        notes="Expanded/combined dataset audited locally for labels, filename groups and near duplicates; use a leakage-safe grouped split.",
    ),
)


def dataset_by_id(identifier: str) -> DatasetProtocol:
    for dataset in DATASETS:
        if dataset.identifier == identifier:
            return dataset
    available = ", ".join(item.identifier for item in DATASETS)
    raise KeyError(f"Unknown dataset '{identifier}'. Available: {available}")
