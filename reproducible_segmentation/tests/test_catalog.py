from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvio_shrimp_seg.candidates import BASELINE, TOP_FIVE, candidate_by_id
from cvio_shrimp_seg.datasets import DATASETS, dataset_by_id


class CatalogTests(unittest.TestCase):
    def test_exactly_five_selected_candidates(self) -> None:
        self.assertEqual(len(TOP_FIVE), 5)

    def test_candidate_ids_are_unique(self) -> None:
        identifiers = [candidate.identifier for candidate in TOP_FIVE]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_catalog_lookups(self) -> None:
        self.assertEqual(candidate_by_id("dpca_strong").display_name, "DPCA strong")
        self.assertEqual(dataset_by_id("mrtu_v1").image_count, 1452)
        self.assertEqual(len(DATASETS), 2)

    def test_new_dataset_notebook_directories_use_current_names(self) -> None:
        notebooks = [path for candidate in (BASELINE, *TOP_FIVE) for path in candidate.new_dataset_notebooks]
        self.assertTrue(notebooks)
        allowed_roots = {
            "yolov11n_attention/expanded_grouped_data",
            "yolov11n_attention/mrtu_grouped_data",
        }
        self.assertTrue(all("/".join(path.split("/")[:2]) in allowed_roots for path in notebooks))


if __name__ == "__main__":
    unittest.main()
