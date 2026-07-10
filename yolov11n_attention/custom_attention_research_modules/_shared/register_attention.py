"""Register research attention modules with Ultralytics at runtime."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

from .attention_modules import RESEARCH_ATTENTION_MODULES, module_name_list


def _attention_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_project_import_paths() -> None:
    """Prefer the vendored Ultralytics package when this repo is the cwd."""

    root = _attention_root()
    vendored = root / "ultralytics"
    if (vendored / "ultralytics" / "__init__.py").exists() and str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))
    if str(root) not in sys.path:
        sys.path.insert(1, str(root))

    loaded = sys.modules.get("ultralytics")
    if loaded is not None and not hasattr(loaded, "__version__") and (vendored / "ultralytics").exists():
        for name in list(sys.modules):
            if name == "ultralytics" or name.startswith("ultralytics."):
                del sys.modules[name]


def register_research_attention() -> None:
    """Inject research attention classes into the Ultralytics parser namespace.

    The patch is in-memory only. It does not edit the vendored or installed
    Ultralytics package and is safe to call repeatedly from notebooks/scripts.
    """

    ensure_project_import_paths()

    import ultralytics.nn.tasks as tasks

    for module_cls in RESEARCH_ATTENTION_MODULES:
        setattr(tasks, module_cls.__name__, module_cls)

    tasks.RESEARCH_ATTENTION_MODULES = RESEARCH_ATTENTION_MODULES

    if getattr(tasks, "_shrimp_research_attention_parse_patched", False):
        return

    source = inspect.getsource(tasks.parse_model)
    insert = """        elif m in RESEARCH_ATTENTION_MODULES:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
"""
    if "elif m in RESEARCH_ATTENTION_MODULES:" in source:
        tasks._shrimp_research_attention_parse_patched = True
        return

    markers = (
        "        elif m in frozenset(\n            {\n                Detect,",
        "        elif m is SemanticSegment:",
        "        elif m in frozenset({TorchVision, Index}):",
    )
    for marker in markers:
        if marker in source:
            patched = source.replace(marker, insert + marker, 1)
            exec(compile(patched, "<shrimp_research_attention_parse_model>", "exec"), tasks.__dict__)
            tasks._shrimp_research_attention_parse_patched = True
            return

    raise RuntimeError(
        "Could not patch ultralytics.nn.tasks.parse_model for research attention modules. "
        f"Modules: {', '.join(module_name_list())}"
    )


__all__ = ["ensure_project_import_paths", "register_research_attention"]
