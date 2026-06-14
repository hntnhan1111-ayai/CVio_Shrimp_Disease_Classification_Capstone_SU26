"""Portable project paths and runtime device selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    output_dir: Path

    @classmethod
    def from_environment(cls) -> "ProjectPaths":
        root = _env_path("PROJECT_ROOT", Path.cwd())
        return cls(
            project_root=root,
            data_dir=_env_path("DATA_DIR", root / "datasets" / "processed-images"),
            output_dir=_env_path("OUTPUT_DIR", root / "runs"),
        )


def resolve_device(requested: str | int | None = "auto") -> str:
    """Return `0,1`, `0`, or `cpu`, unless the caller supplied a device."""
    value = str(requested if requested is not None else "auto").strip()
    if value and value.lower() != "auto":
        return value
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        return "0,1" if torch.cuda.device_count() >= 2 else "0"
    except Exception:
        return "cpu"
