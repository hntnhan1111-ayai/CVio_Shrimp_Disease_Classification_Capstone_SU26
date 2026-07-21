#!/usr/bin/env python3
"""Parse all YAML configs and assert integrity."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from cvio_asl_ldam.utils.io import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")
DATASET_CONFIGS = {
    "shrimpdb3": Path("configs/datasets/shrimpdb.yaml"),
    "combined4": Path("configs/datasets/combined4.yaml"),
}


def run() -> int:
    study = load_yaml(STUDY_CONFIG)
    assert study.get("study", {}).get("seed") == 42, "seed must be 42"

    split = study.get("study", {}).get("split", {})
    assert abs(split.get("train", 0) - 0.70) < 1e-9, "train split must be 0.70"
    assert abs(split.get("val", 0) - 0.15) < 1e-9, "val split must be 0.15"
    assert abs(split.get("test", 0) - 0.15) < 1e-9, "test split must be 0.15"

    training = study.get("training", {})
    assert training.get("model") == "yolo26m-cls"
    assert training.get("method") == "asl_ldam_simam_dcfr"
    assert training.get("imgsz") == 224
    assert training.get("epochs") == 30
    assert training.get("batch") == 32
    assert training.get("optimizer") == "AdamW"
    assert training.get("lr0") == 0.00125
    assert training.get("lrf") == 0.01
    assert training.get("cos_lr") is True
    assert training.get("auto_augment") == "randaugment"
    assert training.get("erasing") == 0.4

    loss = study.get("loss", {})
    assert loss.get("gamma_pos") == 0.0
    assert loss.get("gamma_neg") == 4.0
    assert loss.get("label_smoothing") == 0.1
    assert loss.get("ldam_max_m") == 0.5
    assert loss.get("ldam_scale") == 30.0

    attention = study.get("attention", {})
    assert attention.get("name") == "SimAM_DCFR"
    assert attention.get("e_lambda") == 0.0001

    for exp_name, ds_path in DATASET_CONFIGS.items():
        ds = load_yaml(ds_path)
        assert ds.get("experiment") == exp_name
        assert ds.get("dataset") in {"shrimpdb", "combined4"}
        if exp_name == "shrimpdb3":
            assert ds.get("train_class_counts") == [49, 78, 94]
        elif exp_name == "combined4":
            assert ds.get("train_class_counts") == [331, 217, 323, 154]

    LOGGER.info("All config integrity assertions passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate config integrity")
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Config integrity failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Config integrity error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

def test_main() -> None:
    assert run() == 0

