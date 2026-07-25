from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvio_shrimp_seg.evaluation import healthy_aware_score


class HealthyAwareTests(unittest.TestCase):
    def test_report_formula_uses_fractional_rates(self) -> None:
        score = healthy_aware_score(0.55, 0.1, 0.2, 0.3)
        self.assertAlmostEqual(score, 0.485)

    def test_higher_healthy_fp_lowers_score(self) -> None:
        lower_fp = healthy_aware_score(0.5, 0.0, 0.0, 0.1)
        higher_fp = healthy_aware_score(0.5, 0.0, 0.0, 0.2)
        self.assertGreater(lower_fp, higher_fp)


if __name__ == "__main__":
    unittest.main()
