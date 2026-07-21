#!/usr/bin/env python3
"""Assert cvio_asl_ldam modules are importable."""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def run() -> int:
    import cvio_asl_ldam.attention
    import cvio_asl_ldam.losses
    import cvio_asl_ldam.data
    import cvio_asl_ldam.utils

    assert hasattr(cvio_asl_ldam.attention, "SimAMDCFR")
    assert hasattr(cvio_asl_ldam.losses, "ASLLDAMLoss")
    assert hasattr(cvio_asl_ldam.data, "audit_dataset") or hasattr(cvio_asl_ldam.data, "make_split")
    assert hasattr(cvio_asl_ldam.utils, "io")
    LOGGER.info("All cvio_asl_ldam modules importable")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate imports")
    args = parser.parse_args()
    try:
        return run()
    except Exception as exc:
        LOGGER.error("Import test failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

def test_main() -> None:
    assert run() == 0

