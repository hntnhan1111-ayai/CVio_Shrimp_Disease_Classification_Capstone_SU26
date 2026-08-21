"""Validate source references in the selected-candidate catalog."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from cvio_shrimp_seg.candidates import BASELINE, TOP_FIVE  # noqa: E402
from cvio_shrimp_seg.datasets import DATASETS  # noqa: E402


def main() -> None:
    failed: list[str] = []
    for candidate in (BASELINE, *TOP_FIVE):
        for source in (*candidate.legacy_sources, *candidate.new_dataset_notebooks):
            if not (REPO_ROOT / source).is_file():
                failed.append(source)
        print(f"{candidate.identifier}: {candidate.port_status}")

    for dataset in DATASETS:
        print(f"dataset {dataset.identifier}: {dataset.status}")

    if failed:
        raise SystemExit("Missing legacy sources:\n- " + "\n- ".join(failed))
    print("Catalog validation passed.")


if __name__ == "__main__":
    main()
