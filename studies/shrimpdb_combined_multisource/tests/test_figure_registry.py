#!/usr/bin/env python3
"""Verify figure_registry.json against on-disk figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def run() -> int:
    reg_path = Path("artifacts/metadata/figure_registry.json")
    assert reg_path.is_file(), f"Missing {reg_path}"
    entries = json.loads(reg_path.read_text(encoding="utf-8"))
    assert entries, "empty figure registry"
    seen_filenames = set()
    for entry in entries:
        path = Path(entry["filename"])
        assert path.is_file(), f"figure missing: {path}"
        actual = _sha256(path)
        assert actual == entry["sha256"], f"figure hash mismatch: {path}"
        assert path.name not in seen_filenames, f"duplicate figure: {path.name}"
        seen_filenames.add(path.name)
        assert entry["status"] == "verified"
        assert entry["caption"], "empty caption"
        assert entry["experiment"], "missing experiment"
        assert entry["checkpoint"], "missing checkpoint"
        assert entry["split"], "missing split"
        assert entry["verified_method"], "missing verified_method"
        assert entry["identity_audit_reference"], "missing identity_audit_reference"

    LOGGER.info("Figure registry validated (entries=%d)", len(entries))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Figure registry failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Figure registry error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def test_main() -> None:
    assert run() == 0
