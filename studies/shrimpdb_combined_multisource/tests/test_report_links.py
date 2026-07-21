#!/usr/bin/env python3
"""Verify links in README.md and report integrity."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

README = Path("README.md")
REPORT = Path("artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html")


def run() -> int:
    assert README.is_file(), f"Missing {README}"
    content = README.read_text(encoding="utf-8")

    image_links = re.findall(r"!\[.*?\]\((.*?)\)", content)
    for link in image_links:
        if link.startswith(("http://", "https://")):
            continue
        p = Path(link)
        if not p.is_absolute():
            p = README.parent / p
        assert p.is_file(), f"Missing image: {p}"

    table_links = re.findall(r"\[.*?\]\((artifacts/tables/[^)]+)\)", content)
    for link in table_links:
        if link.startswith(("http://", "https://")):
            continue
        p = Path(link)
        if not p.is_absolute():
            p = README.parent / p
        assert p.is_file(), f"Missing table: {p}"

    assert REPORT.is_file(), f"Missing report: {REPORT}"
    assert REPORT.stat().st_size > 0, f"Report is empty: {REPORT}"

    LOGGER.info("All README links and report integrity passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate README and report links")
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Link validation failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Link validation error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

def test_main() -> None:
    assert run() == 0

