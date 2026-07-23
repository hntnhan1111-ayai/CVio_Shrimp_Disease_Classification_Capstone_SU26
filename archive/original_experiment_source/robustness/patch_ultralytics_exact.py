#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.find_spec("ultralytics")
if spec is None or spec.submodule_search_locations is None:
    raise RuntimeError("Ultralytics is not installed")

U = Path(next(iter(spec.submodule_search_locations))).resolve()
M = U / "nn" / "modules"
T = U / "nn" / "tasks.py"
I = M / "__init__.py"

for path in (M, T, I, HERE / "sldra_block.py"):
    if not path.exists():
        raise FileNotFoundError(path)

shutil.copy2(HERE / "sldra_block.py", M / "sldra_block.py")

s = I.read_text(encoding="utf-8")
marker = "# BEGIN CVIO RECSRA COMPAT"
if marker not in s:
    s = s.rstrip() + (
        "\n# BEGIN CVIO RECSRA COMPAT\n"
        "from .sldra_block import C3k2SLDRA, SLDRA\n"
        "C3k2RECSRA = C3k2SLDRA\n"
        "RECSRA = SLDRA\n"
        "# END CVIO RECSRA COMPAT\n"
    )
    I.write_text(s, encoding="utf-8")

s = T.read_text(encoding="utf-8")
if "# CVIO_RECSRA_IMPORT" not in s:
    a = s.find("from ultralytics.nn.modules import (")
    if a < 0:
        raise RuntimeError("Cannot find ultralytics.nn.modules import block")
    b = s.find(")", a)
    if b < 0:
        raise RuntimeError("Cannot close import block")
    s = s[:b] + "\n    # CVIO_RECSRA_IMPORT\n    C3k2SLDRA,\n" + s[b:]

def add_to_frozenset(text: str, name: str) -> str:
    a = text.find(f"{name} = frozenset(")
    if a < 0:
        raise RuntimeError(f"Cannot find {name}")
    b = text.find("{", a)
    depth = 0
    for idx in range(b, len(text)):
        depth += (text[idx] == "{") - (text[idx] == "}")
        if depth == 0:
            if "C3k2SLDRA" not in text[b:idx]:
                text = text[:idx] + "\n            C3k2SLDRA,\n" + text[idx:]
            return text
    raise RuntimeError(f"Cannot parse {name}")

s = add_to_frozenset(s, "base_modules")
s = add_to_frozenset(s, "repeat_modules")
T.write_text(s, encoding="utf-8")

print("[PATCHED]", U)
print("[MODULE] C3k2SLDRA checkpoint compatibility")
print("[ACADEMIC NAME] RECSRA")
