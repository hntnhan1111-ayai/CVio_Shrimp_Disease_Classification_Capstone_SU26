#!/usr/bin/env python3
"""Verify effective_epochs.json matches log evidence."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def run() -> int:
    eff = json.loads(Path("artifacts/final_application_model/effective_epochs.json").read_text(encoding="utf-8"))
    s3 = eff["experiments"]["shrimpdb3"]
    c4 = eff["experiments"]["combined4"]

    assert s3["executed_epoch_count"] == 21
    assert s3["best_validation_checkpoint_epoch"] == 6
    assert s3["early_stopped"] is True

    assert c4["executed_epoch_count"] == 30
    assert c4["early_stopped"] is False
    assert c4["budget_epoch_count"] == 30

    # Verify log evidence
    s3_log = Path("artifacts/logs/shrimpdb3_training.log").read_text(encoding="utf-8", errors="replace")
    assert "EarlyStopping" in s3_log, "shrimpdb3 log missing EarlyStopping"
    assert "Best results observed at epoch 6" in s3_log, "shrimpdb3 log missing epoch 6 anchor"

    c4_log = Path("artifacts/logs/combined4_training.log").read_text(encoding="utf-8", errors="replace")
    assert "EarlyStopping" not in c4_log, "combined4 log should not contain EarlyStopping"

    LOGGER.info("Effective-epoch wording validated (s3={}, c4={})", s3["executed_epoch_count"], c4["executed_epoch_count"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        return run()
    except AssertionError as exc:
        LOGGER.error("Effective-epoch check failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Effective-epoch check error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def test_main() -> None:
    assert run() == 0
