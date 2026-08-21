"""Configuration and safety gates for the lightweight segmentation benchmark."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent


class ConfigurationError(ValueError):
    """Raised when a benchmark configuration cannot produce a comparable run."""


@dataclass(frozen=True)
class RunPlan:
    model_id: str
    checkpoint: str
    data_yaml: Path
    output_root: Path
    run_name: str
    expected_params_millions: float
    parameter_limit_millions: float
    train_args: dict[str, Any]


def load_config(path: Path = ROOT / "config.yaml") -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ConfigurationError("config.yaml must be a mapping with schema_version: 1")
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_reference_source(config: dict[str, Any]) -> Path:
    profile = config["runtime_profiles"]["ultralytics_baseline"]
    source = REPO_ROOT / profile["source"]
    if not source.is_file():
        raise ConfigurationError(f"Reference notebook does not exist: {source}")
    actual = sha256(source)
    expected = str(profile["source_sha256"]).upper()
    if actual != expected:
        raise ConfigurationError(
            f"Reference notebook changed (expected {expected}, found {actual}). "
            "Review the protocol before updating the pinned hash."
        )
    return source


def primary_model(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = config["models"]["primary_instance_track"]
    for model in models:
        if model.get("id") == model_id:
            if model.get("framework") != "ultralytics":
                raise ConfigurationError(
                    f"{model_id} uses {model.get('framework')}; use its separate runner."
                )
            if not model.get("runnable") or not model.get("checkpoint"):
                raise ConfigurationError(f"{model_id} is not runnable by the Ultralytics runner")
            return model
    available = ", ".join(model["id"] for model in models if model.get("framework") == "ultralytics")
    raise ConfigurationError(f"Unknown Ultralytics model '{model_id}'. Available: {available}")


def resolve_data_yaml(config: dict[str, Any], override: str | None = None) -> Path:
    value = override or os.environ.get(config["dataset"]["data_yaml_env"], "")
    if not value:
        raise ConfigurationError(
            f"Set {config['dataset']['data_yaml_env']} or pass --data-yaml before running."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Dataset YAML does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not {"train", "val", "names"}.issubset(data):
        raise ConfigurationError("Dataset YAML must define train, val, and names")
    expected_names = list(config["dataset"]["classes"])
    names = data["names"]
    actual_names = list(names.values()) if isinstance(names, dict) else list(names)
    if actual_names != expected_names:
        raise ConfigurationError(f"Dataset classes must be {expected_names}, found {actual_names}")
    return path


def build_run_plan(
    config: dict[str, Any], model_id: str, *, data_yaml: str | None = None, smoke: bool = False
) -> RunPlan:
    model = primary_model(config, model_id)
    dataset_path = resolve_data_yaml(config, data_yaml)
    training = dict(config["training"])
    training.pop("unspecified_args_policy", None)
    train_args = {
        "data": str(dataset_path),
        **training,
        **config["augmentation"],
        "seed": config["reproducibility"]["seed"],
        "deterministic": config["reproducibility"]["deterministic"],
        "project": str((ROOT / config["experiment"]["output_root"]).resolve()),
        "name": f"{model_id}__seed{config['reproducibility']['seed']}",
        "exist_ok": False,
    }
    if smoke:
        train_args.update(epochs=1, batch=1, imgsz=64, workers=0, name=f"{model_id}__smoke")
    return RunPlan(
        model_id=model_id,
        checkpoint=model["checkpoint"],
        data_yaml=dataset_path,
        output_root=Path(train_args["project"]),
        run_name=train_args["name"],
        expected_params_millions=float(model["expected_params_millions"]),
        parameter_limit_millions=float(config["experiment"]["parameter_limit_millions"]),
        train_args=train_args,
    )


def verify_parameter_count(actual: int, plan: RunPlan, *, tolerance: float = 0.20) -> float:
    actual_millions = actual / 1_000_000
    if actual_millions >= plan.parameter_limit_millions:
        raise ConfigurationError(
            f"{plan.model_id} has {actual_millions:.3f}M parameters; limit is strictly below "
            f"{plan.parameter_limit_millions:.3f}M."
        )
    relative_error = abs(actual_millions - plan.expected_params_millions) / plan.expected_params_millions
    if relative_error > tolerance:
        raise ConfigurationError(
            f"{plan.model_id} parameter count {actual_millions:.3f}M differs from the configured "
            f"{plan.expected_params_millions:.3f}M by more than {tolerance:.0%}."
        )
    return actual_millions
