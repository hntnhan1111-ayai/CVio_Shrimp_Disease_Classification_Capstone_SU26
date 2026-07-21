#!/usr/bin/env python3
"""Verify recorded checksums match on-disk artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

EXPECTED_CHECKSUMS = {
    "yolo26m_asl_ldam_simam_dcfr_combined4_best.pt": "9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606",
}


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(root: Path = Path(".")) -> int:
    failures = []
    for filename, expected in EXPECTED_CHECKSUMS.items():
        path = root / filename
        if not path.is_file():
            failures.append(f"Missing checkpoint: {path}")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            failures.append(f"Checksum mismatch for {filename}: expected={expected} actual={actual}")

    registry_path = root / "model_registry/checkpoints.json"
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for exp, info in registry.get("checkpoints", {}).items():
            ckpt_path = Path(info["path"])
            if ckpt_path.is_file():
                actual = _sha256_file(ckpt_path)
                if actual != info.get("sha256"):
                    failures.append(f"Registry checksum mismatch for {exp}")

    if failures:
        for f in failures:
            LOGGER.error(f)
        return 1
    LOGGER.info("All checksums verified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify checksums")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    try:
        return run(Path(args.root))
    except Exception as exc:
        LOGGER.error("Checksum verification failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
