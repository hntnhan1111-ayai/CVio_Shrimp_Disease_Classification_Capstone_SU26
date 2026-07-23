"""Idempotent Ultralytics compatibility patch for RECSRA checkpoints."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


def patch_ultralytics() -> Path:
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("Ultralytics is not installed")

    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    modules = root / "nn" / "modules"
    tasks = root / "nn" / "tasks.py"
    init_file = modules / "__init__.py"
    source = Path(__file__).resolve().parent / "ultralytics_compat" / "sldra_block.py"

    for path in (modules, tasks, init_file, source):
        if not path.exists():
            raise FileNotFoundError(path)

    shutil.copy2(source, modules / "sldra_block.py")

    text = init_file.read_text(encoding="utf-8")
    marker = "# BEGIN CVIO RECSRA COMPAT"
    if marker not in text:
        text = text.rstrip() + (
            "\n# BEGIN CVIO RECSRA COMPAT\n"
            "from .sldra_block import C3k2SLDRA, SLDRA\n"
            "C3k2RECSRA = C3k2SLDRA\n"
            "RECSRA = SLDRA\n"
            "# END CVIO RECSRA COMPAT\n"
        )
        init_file.write_text(text, encoding="utf-8")

    text = tasks.read_text(encoding="utf-8")
    if "# CVIO_RECSRA_IMPORT" not in text:
        start = text.find("from ultralytics.nn.modules import (")
        end = text.find(")", start)
        if start < 0 or end < 0:
            raise RuntimeError("Cannot locate Ultralytics module import block")
        text = text[:end] + "\n    # CVIO_RECSRA_IMPORT\n    C3k2SLDRA,\n" + text[end:]

    def add_to_frozenset(value: str, name: str) -> str:
        start = value.find(f"{name} = frozenset(")
        if start < 0:
            raise RuntimeError(f"Cannot find {name}")
        brace = value.find("{", start)
        depth = 0
        for index in range(brace, len(value)):
            depth += (value[index] == "{") - (value[index] == "}")
            if depth == 0:
                if "C3k2SLDRA" not in value[brace:index]:
                    value = value[:index] + "\n            C3k2SLDRA,\n" + value[index:]
                return value
        raise RuntimeError(f"Cannot parse {name}")

    text = add_to_frozenset(text, "base_modules")
    text = add_to_frozenset(text, "repeat_modules")
    tasks.write_text(text, encoding="utf-8")
    return root
