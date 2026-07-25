from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


class ConfigInventoryTests(unittest.TestCase):
    def test_expected_dataset_and_candidate_configs_exist(self) -> None:
        self.assertEqual(
            {path.stem for path in (CONFIG_ROOT / "datasets").glob("*.yaml")},
            {"legacy_129_test", "mrtu_v1"},
        )
        self.assertEqual(
            {path.stem for path in (CONFIG_ROOT / "candidates").glob("*.yaml")},
            {
                "baseline_strong",
                "simam_ca_strong",
                "lka_simam_head",
                "dpca_strong",
                "cote_boundarylite_p4_strong",
                "simam_ca_wiou_v3",
            },
        )


if __name__ == "__main__":
    unittest.main()
