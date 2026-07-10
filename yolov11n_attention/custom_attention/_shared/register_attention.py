"""Register custom attention modules with Ultralytics at runtime."""

from __future__ import annotations

import inspect

from .attention_modules import CUSTOM_ATTENTION_MODULES, module_name_list


def register_custom_attention() -> None:
    """Inject custom attention classes into Ultralytics parse_model namespace.

    This function patches Ultralytics in memory only. It does not edit the
    installed package or any baseline notebook/source file on disk.
    """

    import ultralytics.nn.tasks as tasks

    for module_cls in CUSTOM_ATTENTION_MODULES:
        setattr(tasks, module_cls.__name__, module_cls)

    tasks.CUSTOM_ATTENTION_MODULES = CUSTOM_ATTENTION_MODULES

    if getattr(tasks, "_shrimp_custom_attention_parse_patched", False):
        return

    source = inspect.getsource(tasks.parse_model)
    marker = "        elif m in frozenset(\n            {\n                Detect,"
    insert = """        elif m in CUSTOM_ATTENTION_MODULES:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
"""
    if marker not in source:
        raise RuntimeError(
            "Could not patch ultralytics.nn.tasks.parse_model: Detect branch marker not found. "
            f"Custom modules: {', '.join(module_name_list())}"
        )

    patched = source.replace(marker, insert + marker, 1)
    exec(compile(patched, "<shrimp_custom_attention_parse_model>", "exec"), tasks.__dict__)
    tasks._shrimp_custom_attention_parse_patched = True


__all__ = ["register_custom_attention"]
