"""Academic naming layer for the frozen checkpoint-compatible implementation.

RECSRA expands to Robust Energy-Guided Circular–Spatial Residual Attention.
The serialized winning checkpoint was created with the historical class names
``SLDRA`` and ``C3k2SLDRA``. They are preserved as compatibility aliases.
"""

from .ultralytics_compat.sldra_block import C3k2SLDRA, SLDRA

RECSRA = SLDRA
C3k2RECSRA = C3k2SLDRA

__all__ = ["RECSRA", "C3k2RECSRA", "SLDRA", "C3k2SLDRA"]
