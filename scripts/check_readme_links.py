"""Check local Markdown links and image targets in the release documentation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "export" / "README.md"]
    files.extend((ROOT / "docs").rglob("*.md"))
    files.extend((ROOT / "weights").rglob("*.md"))
    return sorted(set(files))


def local_target(raw: str) -> str | None:
    target = raw.strip().strip("<>").split(maxsplit=1)[0]
    if target.startswith(SKIP_PREFIXES):
        return None
    return unquote(target.split("#", 1)[0])


def main() -> int:
    checked = 0
    failures: list[str] = []
    for document in markdown_files():
        text = document.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = local_target(match.group(1))
            if not target:
                continue
            checked += 1
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{document.relative_to(ROOT)}:{line}: missing {target}"
                )
    print(f"markdown_files={len(markdown_files())}")
    print(f"local_links_checked={checked}")
    if failures:
        print("\n".join(failures))
        return 1
    print("status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
