#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

LOCKED = {
    "yolo11s_baseline_best.pt": "b9e30aa76f819126c7e6e9f3d3007a74d0839ad1f95941978a6a214484eb219d",
    "yolo11s_recsra_best.pt": "c1652101bb870a0b174b569ac47a70bb8cfafe119577e1ca2fe2de87f57824ff",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    args = ap.parse_args()
    root = Path(args.checkpoint_dir)
    for name, expected in LOCKED.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"SHA mismatch for {path}: {actual}")
        print(f"[OK] {name} {actual}")


if __name__ == "__main__":
    main()
