"""Run-overwrite guards for training scripts.

Prevents silent overwrites of reviewer-facing run directories. A run is
considered `complete` when its `status.json` reports `status == "ok"`
and `best.pt` exists.
"""

from __future__ import annotations

import json
from pathlib import Path


def is_run_complete(run_dir: str | Path) -> bool:
    run_dir = Path(run_dir)
    status_file = run_dir / "status.json"
    if not status_file.is_file():
        return False
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        status.get("status") == "ok"
        and (run_dir / "weights" / "best.pt").is_file()
    )


def decide_action(
    run_dir: str | Path,
    skip_if_complete: bool,
    force: bool,
    resume: bool,
) -> str:
    """Return one of: `skip`, `force`, `resume`, `train`.

    Precedence: `force` > `resume` > `skip_if_complete` > `train`.
    """
    if force:
        return "force"
    if resume:
        return "resume"
    if skip_if_complete and is_run_complete(run_dir):
        return "skip"
    return "train"


def write_status(run_dir: str | Path, status: str, **extra) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status}
    payload.update(extra)
    path = run_dir / "status.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
