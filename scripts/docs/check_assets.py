#!/usr/bin/env python3
"""Validate documentation assets for safety, naming, inventory, and size."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
ALLOWED = {".svg", ".gif", ".png", ".md"}
SVG_FORBIDDEN = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"<script\b",
        r"\bon\w+\s*=",
        r"(?:href|src)\s*=\s*[\"']https?://",
        r"<foreignObject\b",
    )
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-gif-mb", type=float, default=2.0)
    args = parser.parse_args()
    failures: list[str] = []
    if not (ASSETS / "README.md").exists():
        failures.append("docs/assets/README.md: asset license inventory is missing")
    for path in sorted(ASSETS.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if " " in path.name:
            failures.append(f"{relative}: filename contains spaces")
        if path.suffix.lower() not in ALLOWED:
            failures.append(f"{relative}: unsupported extension")
        if (
            path.suffix.lower() == ".gif"
            and path.stat().st_size > args.max_gif_mb * 1024 * 1024
        ):
            failures.append(f"{relative}: GIF exceeds {args.max_gif_mb:g} MB")
        if path.suffix.lower() == ".gif" and path.read_bytes()[:6] not in {
            b"GIF87a",
            b"GIF89a",
        }:
            failures.append(f"{relative}: invalid GIF signature")
        if (
            path.suffix.lower() == ".png"
            and path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
        ):
            failures.append(f"{relative}: invalid PNG signature")
        if path.suffix.lower() == ".svg":
            content = path.read_text(encoding="utf-8")
            try:
                ET.fromstring(content)
            except ET.ParseError as error:
                failures.append(f"{relative}: malformed SVG: {error}")
            for pattern in SVG_FORBIDDEN:
                if pattern.search(content):
                    failures.append(
                        f"{relative}: unsafe SVG content matched {pattern.pattern}"
                    )
            if "<title" not in content or "<desc" not in content:
                failures.append(f"{relative}: SVG requires title and description")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Documentation assets passed safety, inventory, naming, and size checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
