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
    "artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt": "9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606",
}

REGISTRY_PATHS = [
    "model_registry/checkpoints.json",
    "model_registry/class_mapping.json",
]


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

    for rel in REGISTRY_PATHS:
        registry_path = root / rel
        if not registry_path.is_file():
            failures.append(f"Missing registry: {rel}")
            continue
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"Invalid registry JSON: {rel}: {exc}")
            continue

        experiments = registry.get("experiments") if isinstance(registry, dict) else None
        if experiments is None:
            continue
        for exp_name, info in experiments.items():
            if not isinstance(info, dict):
                continue
            sha = info.get("best_checkpoint_sha256") or info.get("sha256")
            if not sha:
                continue
            ckpt_rel = info.get("best_checkpoint") or info.get("path")
            if not ckpt_rel:
                continue
            ckpt_path = Path(ckpt_rel)
            if not ckpt_path.is_absolute():
                ckpt_path = root / ckpt_path
            if not ckpt_path.is_file():
                # Files referenced for reproducibility but not packaged are
                # documented in model_registry/checkpoints.json with explicit
                # SHA-256; we record the situation rather than fail.
                LOGGER.info("Registry checkpoint not on disk for %s: %s", exp_name, ckpt_rel)
                continue
            actual = _sha256_file(ckpt_path)
            if actual != sha:
                failures.append(
                    f"Registry checksum mismatch for {exp_name}: expected={sha} actual={actual}"
                )

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
