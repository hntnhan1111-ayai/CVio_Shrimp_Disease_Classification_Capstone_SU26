#!/usr/bin/env python3
"""Perform lightweight offline Markdown structure checks."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def check(path: Path) -> list[str]:
    failures: list[str] = []
    headings: set[str] = set()
    last_level = 0
    fences = 0
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.rstrip() != line:
            failures.append(f"{path.relative_to(ROOT)}:{number}: trailing whitespace")
        if line.startswith("```"):
            fences += 1
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match:
            level, text = len(match.group(1)), match.group(2).lower()
            if text in headings:
                failures.append(
                    f"{path.relative_to(ROOT)}:{number}: duplicate heading {text!r}"
                )
            headings.add(text)
            if last_level and level > last_level + 1:
                failures.append(
                    f"{path.relative_to(ROOT)}:{number}: heading level jumps"
                )
            last_level = level
    if fences % 2:
        failures.append(f"{path.relative_to(ROOT)}: unbalanced fenced code block")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = (
        [ROOT / path for path in args.paths]
        if args.paths
        else sorted(ROOT.rglob("*.md"))
    )
    failures = [item for path in paths for item in check(path)]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Validated Markdown structure in {len(paths)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
