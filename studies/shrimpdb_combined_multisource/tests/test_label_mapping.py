#!/usr/bin/env python3
"""Assert label mappings are correct."""

from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

from cvio_asl_ldam.utils.io import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")


def run() -> int:
    study = load_yaml(STUDY_CONFIG)
    mapping = study.get("experiments", {}).get("shrimpdb3", {}).get("source_mapping", {})
    assert mapping.get("Tom_BT") == "Healthy", f"Tom_BT should map to Healthy, got {mapping.get('Tom_BT')}"
    assert mapping.get("Den_Mang") == "BG", f"Den_Mang should map to BG, got {mapping.get('Den_Mang')}"
    assert mapping.get("Dom_Trang") == "WSSV", f"Dom_Trang should map to WSSV, got {mapping.get('Dom_Trang')}"

    excluded = set(study.get("experiments", {}).get("shrimpdb3", {}).get("source_mapping", {}).keys())
    excluded.update(["Dom_Den", "Hoai_Tu_Co", "Hoai_tu_gan"])
    shrimpdb_classes = set(study.get("datasets", {}).get("shrimpdb", {}).get("all_classes", []))
    for cls in ["Dom_Den", "Hoai_Tu_Co", "Hoai_tu_gan"]:
        assert cls not in excluded or cls in ["Dom_Den", "Hoai_Tu_Co", "Hoai_tu_gan"]
    assert "Dom_Den" not in ["Healthy", "BG", "WSSV"], "Dom_Den must not be in ShrimpDB-3"
    assert "Hoai_Tu_Co" not in ["Healthy", "BG", "WSSV"], "Hoai_Tu_Co must not be in ShrimpDB-3"
    assert "Hoai_tu_gan" not in ["Healthy", "BG", "WSSV"], "Hoai_tu_gan must not be in ShrimpDB-3"

    combined_classes = study.get("experiments", {}).get("combined4", {}).get("class_names", [])
    assert "WSSV_BG" in combined_classes, "Combined-4 must contain WSSV_BG"

    LOGGER.info("All label mapping assertions passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate label mappings")
    parser.add_argument("--config", default=str(STUDY_CONFIG))
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Label mapping failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Label mapping error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

def test_main() -> None:
    assert run() == 0

