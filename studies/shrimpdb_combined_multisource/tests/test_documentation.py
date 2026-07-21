#!/usr/bin/env python3
"""Verify README.md and docs reference real files and use correct epoch wording."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


BAD_PHRASES = [
    r"<= ?15 ?\(early stop\)",
    r"early-?stopped at epoch ?<= ?15",
]


def run() -> int:
    readme = Path("README.md").read_text(encoding="utf-8")
    for bad in BAD_PHRASES:
        assert not re.search(bad, readme, re.I), f"README still contains bad phrasing: {bad}"

    # Image links resolve
    img_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    for link in img_links:
        if link.startswith(("http://", "https://", "data:")):
            continue
        p = (Path(".") / link).resolve()
        assert p.is_file(), f"Broken image in README: {link}"

    # Table links should exist
    for link in re.findall(r"\((?:[^()]*)?(artifacts/[^()]+)\)", readme):
        if link.startswith(("http://", "https://")):
            continue
        p = (Path(".") / link).resolve()
        assert p.is_file(), f"Broken table link in README: {link}"

    # Docs cross-link report
    results = Path("docs/RESULTS.md").read_text(encoding="utf-8")
    assert "CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html" in results, "RESULTS.md must mention HTML report"
    assert "artifacts/final_application_model/effective_epochs.json" in results, "RESULTS.md must reference effective_epochs.json"

    # README mentions the HTML report
    assert "CVio_Final_Academic_Report" in readme, "README must reference the HTML report"

    # README badges: ensure they look syntactically valid
    for badge in re.findall(r"\!\[[^\]]*\]\((https?://img\.shields\.io/[^\)]+)\)", readme):
        assert "%20" in badge or "-" in badge, f"Badge URL syntax suspect: {badge}"

    LOGGER.info("Documentation links and wording validated")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Documentation check failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Documentation check error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def test_main() -> None:
    assert run() == 0
