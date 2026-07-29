#!/usr/bin/env python3
"""Validate local Markdown/HTML links and anchors in repository documentation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(
    r"!?\[[^\]]*]\(([^)]+)\)|(?:src|href)=[\"']([^\"']+)[\"']", re.IGNORECASE
)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def slug(text: str) -> str:
    value = re.sub(r"<[^>]+>", "", text).strip().lower()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    return re.sub(r"[\s\-]+", "-", value).strip("-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    values: set[str] = set()
    for _, heading in HEADING.findall(path.read_text(encoding="utf-8")):
        base = slug(heading)
        count = seen.get(base, 0)
        values.add(base if count == 0 else f"{base}-{count}")
        seen[base] = count + 1
    return values


def check(path: Path) -> list[str]:
    failures: list[str] = []
    text = path.read_text(encoding="utf-8")
    for match in LINK.finditer(text):
        target = next(group for group in match.groups() if group)
        target = target.strip().split(maxsplit=1)[0].strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        file_part, _, fragment = target.partition("#")
        destination = (
            path if not file_part else (path.parent / unquote(file_part)).resolve()
        )
        try:
            destination.relative_to(ROOT)
        except ValueError:
            failures.append(
                f"{path.relative_to(ROOT)}: link escapes repository: {target}"
            )
            continue
        if not destination.exists():
            failures.append(f"{path.relative_to(ROOT)}: missing target: {target}")
        elif (
            fragment
            and destination.suffix.lower() == ".md"
            and fragment not in anchors(destination)
        ):
            failures.append(f"{path.relative_to(ROOT)}: missing anchor: {target}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path, help="Markdown paths relative to the repository."
    )
    args = parser.parse_args()
    paths = (
        [ROOT / path for path in args.paths]
        if args.paths
        else sorted(ROOT.rglob("*.md"))
    )
    failures = [failure for path in paths for failure in check(path)]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Validated local links in {len(paths)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
