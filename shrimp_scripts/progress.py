"""Lightweight progress logging for Kaggle and notebook users."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROGRESS_OUTPUT_DIR = Path("/kaggle/working/shrimp_outputs")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def log_event(message: str, *, level: str = "INFO", run_id: str | None = None, output_dir: str | Path | None = None, extra: dict[str, Any] | None = None) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    prefix = f"[{timestamp}] [{level}]"
    if run_id:
        prefix += f" [{run_id}]"
    print(f"{prefix} {message}", flush=True)
    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_PROGRESS_OUTPUT_DIR
    record = {
        "timestamp_utc": timestamp,
        "level": level,
        "message": message,
        "run_id": run_id,
        "extra": extra or {},
    }
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with (target_dir / "progress_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
    except Exception:
        pass


def progress_iter(iterable: Iterable, *, desc: str, total: int | None = None, enabled: bool = True, leave: bool = True):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)
