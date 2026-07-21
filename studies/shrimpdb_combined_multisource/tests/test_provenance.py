#!/usr/bin/env python3
"""Verify the source snapshot provenance artifact and final checkpoint registry."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

EXPECTED_FINAL_SHA = "9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606"


def run() -> int:
    sc_path = Path("artifacts/metadata/source_comparison.json")
    assert sc_path.is_file(), f"Missing {sc_path}"
    sc = json.loads(sc_path.read_text(encoding="utf-8"))
    assert sc.get("reported_status") in (None,), "status field present"  # legacy checkout
    assert sc.get("status") in ("match", "match_with_explicit_diff")
    # Note: documented content_mismatches must only contain allowlisted keys.
    for m in sc.get("content_mismatches", []):
        assert m["path"] in sc.get("allowlisted_diffs", {}), f"undocumented mismatch: {m['path']}"

    # Final checkpoint registry
    reg_path = Path("model_registry/checkpoints.json")
    assert reg_path.is_file(), f"Missing {reg_path}"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert reg.get("final_application_checkpoint_sha256") == EXPECTED_FINAL_SHA
    for exp_name, info in reg.get("experiments", {}).items():
        assert info.get("best_checkpoint_sha256"), f"{exp_name}: missing best checkpoint sha"

    cm = json.loads(Path("artifacts/final_application_model/class_mapping.json").read_text(encoding="utf-8"))
    assert cm.get("checkpoint_sha256") == EXPECTED_FINAL_SHA

    LOGGER.info("Provenance assertions passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Provenance validation failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Provenance validation error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def test_main() -> None:
    assert run() == 0
