#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install   torch==2.12.1+cu126 torchvision==0.27.1+cu126 torchaudio==2.11.0+cu126   --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r environment/requirements-runtime.txt
python -m pip install -e .
PYTHONPATH=src python scripts/patch_ultralytics.py
PYTHONPATH=src python scripts/verify_repository.py
