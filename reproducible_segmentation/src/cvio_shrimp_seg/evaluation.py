"""Metrics explicitly defined in the project report."""

from __future__ import annotations


def healthy_aware_score(
    disease_mask_map50: float,
    mask_count_mae: float,
    disease_miss_rate: float,
    healthy_fp_rate: float,
    *,
    count_penalty: float = 0.05,
    miss_penalty: float = 0.15,
    healthy_fp_penalty: float = 0.10,
) -> float:
    """Return the report's healthy-aware (HA) score.

    All rates are fractions, not percentages. For example, 34.1% is 0.341.
    """

    return (
        disease_mask_map50
        - count_penalty * mask_count_mae
        - miss_penalty * disease_miss_rate
        - healthy_fp_penalty * healthy_fp_rate
    )
