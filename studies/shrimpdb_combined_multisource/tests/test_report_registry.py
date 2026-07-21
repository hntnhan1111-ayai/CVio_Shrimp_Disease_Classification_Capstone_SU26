#!/usr/bin/env python3
"""Verify report_registry.json against the HTML report."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def run() -> int:
    reg_path = Path("artifacts/metadata/report_registry.json")
    assert reg_path.is_file(), f"Missing {reg_path}"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    report_path = Path(reg["filename"])
    assert report_path.is_file(), f"report missing: {report_path}"

    actual = _sha256(report_path)
    assert actual == reg["sha256"], "report sha mismatch"
    assert report_path.stat().st_size == reg["size_bytes"], "report size mismatch"

    text = report_path.read_text(encoding="utf-8")
    title_m = re.search(r"<title>(.*?)</title>", text, re.S|re.I)
    assert title_m, "no <title> tag"
    assert reg.get("title"), "title missing from registry"
    assert reg.get("embedded_image_count", 0) > 0 or reg.get("img_tag_count", 0) > 0, "no embedded images"
    for label, ok in reg.get("required_section_check", {}).items():
        assert ok, f"required section not found: {label}"

    LOGGER.info("Report registry validated (sections=%d, images=%d)", len(reg["sections"]), reg["embedded_image_count"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Report registry failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Report registry error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def test_main() -> None:
    assert run() == 0
