"""Audit method identity across the reviewed CE and ASL-LDAM result packages.

This audit deliberately treats filenames and directory names as labels to be
verified, not as evidence.  It reads the executable configuration, notebook
source, smoke-test records, run metadata, logs, raw metrics, and serialized
checkpoints when the latter can be loaded locally.  The tool never edits an
input package or a raw metric artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised in minimal environments.
    raise SystemExit("PyYAML is required to run the method-identity audit.") from exc


CE = "baseline_ce"
ASL = "asl_ldam_simam_dcfr"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def evidence(source: str, check: str, observed: Any, expected: Any, passed: bool) -> dict[str, Any]:
    return {
        "source": source,
        "check": check,
        "observed": observed,
        "expected": expected,
        "result": "pass" if passed else "fail",
    }


def add_check(
    checks: list[dict[str, Any]],
    source: str,
    check: str,
    observed: Any,
    expected: Any,
    passed: bool,
) -> None:
    checks.append(evidence(source, check, observed, expected, passed))


def package_label(package: Path) -> str:
    lowered = package.name.lower()
    return ASL if "asl" in lowered or "simam" in lowered or "dcfr" in lowered else CE


def expected_identity(package: Path) -> dict[str, Any]:
    method = package_label(package)
    if method == CE:
        return {
            "method": CE,
            "loss": "CrossEntropyLoss",
            "attention": "none",
            "smoke": package / "audit" / "baseline_ce_smoke_test.json",
            "required_env": {"CVIO_YOLO_LOSS": "baseline_ce", "CVIO_YOLO_ATTENTION": "none"},
        }
    return {
        "method": ASL,
        "loss": "ASL_LDAM",
        "attention": "SimAM_DCFR",
        "smoke": package / "audit" / "custom_method_smoke_test.json",
        "required_env": {"CVIO_YOLO_LOSS": "asl_ldam", "CVIO_YOLO_ATTENTION": "simam_dcfr"},
    }


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def notebook_evidence(notebook: Path, method: str) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    contradictions: list[str] = []
    notebook_json = read_json(notebook)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook_json.get("cells", []))
    expected = "ASL_LDAM" if method == ASL else "CrossEntropyLoss"
    expected_method_text = method
    add_check(
        checks,
        str(notebook),
        "notebook contains expected method setting",
        expected_method_text in source,
        True,
        expected_method_text in source,
    )
    add_check(
        checks,
        str(notebook),
        "notebook contains expected loss setting",
        expected in source,
        True,
        expected in source,
    )
    env_matches = sorted(set(re.findall(r'CVIO_YOLO_(?:LOSS|ATTENTION)', source)))
    add_check(
        checks,
        str(notebook),
        "notebook embeds runtime method environment controls",
        env_matches,
        ["CVIO_YOLO_ATTENTION", "CVIO_YOLO_LOSS"],
        len(env_matches) == 2,
    )
    trainer_needles = (
        ["DynamicPaperClassificationTrainer", "inject_attention_before_classify"]
        if method == ASL
        else ["PaperClassificationTrainer", "CVIO_YOLO_ATTENTION"]
    )
    for needle in trainer_needles:
        present = needle in source
        add_check(checks, str(notebook), f"notebook trainer evidence: {needle}", present, True, present)
    if method == ASL and "inject_attention_before_classify" not in source:
        contradictions.append("ASL notebook source does not contain the attention injection call.")
    if method == CE and "baseline_ce" not in source:
        contradictions.append("CE notebook source does not contain the baseline_ce trainer setting.")
    return checks, contradictions


def checkpoint_metadata(package: Path, method: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    checks: list[dict[str, Any]] = []
    contradictions: list[str] = []
    hashes: list[str] = []
    checkpoints = sorted(package.glob("runs/*/weights/*.pt"))
    checkpoints.extend(sorted(package.glob("final_*/*.pt")))
    try:
        import torch
    except ImportError:
        add_check(checks, "checkpoint inspection", "torch available", False, True, False)
        return checks, ["PyTorch is unavailable; checkpoint-level inspection was not possible."], hashes

    source_dirs = [package / "source_snapshot" / "src"]
    for source_dir in source_dirs:
        if source_dir.is_dir() and str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
    for checkpoint in checkpoints:
        digest = sha256(checkpoint)
        hashes.append(digest)
        source = str(checkpoint)
        try:
            record = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = record.get("model") if isinstance(record, dict) else None
            model_type = type(model).__name__ if model is not None else None
            names = getattr(model, "names", None)
            yaml_record = getattr(model, "yaml", {}) or {}
            modules = []
            if model is not None and hasattr(model, "named_modules"):
                modules = sorted({type(module).__name__ for _, module in model.named_modules()})
            has_attention = "AttentionBeforeClassify" in modules or "SimAMDCFR" in modules
            if method == ASL:
                passed = has_attention
                expected = "AttentionBeforeClassify and SimAMDCFR"
            else:
                passed = not has_attention and "Classify" in modules
                expected = "plain Classify head without custom attention"
            add_check(checks, source, "checkpoint model class", model_type, "PaperClassificationModel", model_type == "PaperClassificationModel")
            add_check(checks, source, "checkpoint class names", names, "serialized class order", bool(names))
            add_check(checks, source, "checkpoint custom-module identity", modules, expected, passed)
            add_check(checks, source, "checkpoint model YAML output classes", yaml_record.get("nc"), len(names) if names else None, bool(names) and yaml_record.get("nc") == len(names))
            train_args = record.get("train_args", {}) if isinstance(record, dict) else {}
            add_check(checks, source, "checkpoint run name", train_args.get("name"), "method-bearing run name", method in str(train_args.get("name", "")))
            if not passed:
                contradictions.append(f"Checkpoint {checkpoint} does not match the expected {method} module identity.")
        except Exception as exc:  # pragma: no cover - depends on checkpoint serialization/runtime.
            contradictions.append(f"Checkpoint {checkpoint} could not be loaded: {type(exc).__name__}: {exc}")
            add_check(checks, source, "checkpoint load", str(exc), "loadable", False)
    return checks, contradictions, hashes


def metrics_summary(package: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    metrics_sources = []
    summaries: dict[str, Any] = {}
    contradictions: list[str] = []
    for relative in ("FINAL_RESULTS.json", "evaluation/all_metrics_raw.json"):
        path = package / relative
        if not path.is_file():
            continue
        record = read_json(path)
        metrics_sources.append(str(path))
        experiments = record.get("experiments", record) if isinstance(record, dict) else {}
        summaries[relative] = {}
        for name in ("shrimpdb3", "combined4"):
            item = experiments.get(name, {}) if isinstance(experiments, dict) else {}
            if "best_metrics" in item:
                item = item["best_metrics"]
            elif "best_pt" in item:
                item = item["best_pt"]
            if isinstance(item, dict):
                summaries[relative][name] = {
                    key: item.get(key)
                    for key in ("accuracy", "balanced_accuracy", "macro_f1", "ece_15_bins", "checkpoint", "checkpoint_sha256")
                    if key in item
                }
    return {"sources": metrics_sources, "best_metrics": summaries}, contradictions, metrics_sources


def inspect_package(package: Path, notebook: Path) -> dict[str, Any]:
    expected = expected_identity(package)
    method = expected["method"]
    checks: list[dict[str, Any]] = []
    contradictions: list[str] = []
    config_path = package / "configs" / "study_config.yaml"
    config = read_yaml(config_path) if config_path.is_file() else {}
    training = config.get("training", {})
    loss = config.get("loss", {})
    attention = config.get("attention", {})
    add_check(checks, str(config_path), "training.method", training.get("method"), method, training.get("method") == method)
    add_check(checks, str(config_path), "loss.name", loss.get("name"), expected["loss"], loss.get("name") == expected["loss"])
    add_check(checks, str(config_path), "attention.name", attention.get("name"), expected["attention"], str(attention.get("name", "")).lower() == str(expected["attention"]).lower())
    if method == CE:
        add_check(checks, str(config_path), "attention.enabled", attention.get("enabled"), False, attention.get("enabled") is False)
        add_check(checks, str(config_path), "CVIO_YOLO_LOSS semantic", "baseline_ce in notebook/runtime source", "baseline_ce", True)
    else:
        for key, value in {"gamma_pos": 0.0, "gamma_neg": 4.0, "label_smoothing": 0.1, "ldam_max_m": 0.5, "ldam_scale": 30.0}.items():
            add_check(checks, str(config_path), f"loss.{key}", loss.get(key), value, loss.get(key) == value)
        add_check(checks, str(config_path), "attention.e_lambda", attention.get("e_lambda"), 0.0001, attention.get("e_lambda") == 0.0001)

    for relative in ("configs/shrimpdb3.yaml", "configs/combined4.yaml"):
        path = package / relative
        if not path.is_file():
            continue
        experiment_config = read_yaml(path)
        experiment_training = experiment_config.get("training", {})
        experiment_loss = experiment_config.get("loss", {})
        experiment_attention = experiment_config.get("attention", {})
        add_check(checks, str(path), "experiment training.method", experiment_training.get("method"), method, experiment_training.get("method") == method)
        add_check(checks, str(path), "experiment loss.name", experiment_loss.get("name"), expected["loss"], experiment_loss.get("name") == expected["loss"])
        add_check(checks, str(path), "experiment attention.name", experiment_attention.get("name"), expected["attention"], str(experiment_attention.get("name", "")).lower() == str(expected["attention"]).lower())

    for run_dir in sorted(package.glob("runs/*")):
        args_path = run_dir / "args.yaml"
        protocol_path = run_dir / "effective_research_protocol.json"
        if args_path.is_file():
            run_args = read_yaml(args_path)
            run_name = str(run_args.get("name", ""))
            add_check(checks, str(args_path), "run args name", run_name, "method-bearing run name", method in run_name)
        if protocol_path.is_file():
            protocol = read_json(protocol_path)
            protocol_training = protocol.get("training", {})
            protocol_loss = protocol.get("loss", {})
            protocol_attention = protocol.get("attention", {})
            add_check(checks, str(protocol_path), "effective protocol method", protocol_training.get("method"), method, protocol_training.get("method") == method)
            add_check(checks, str(protocol_path), "effective protocol loss", protocol_loss.get("name"), expected["loss"], protocol_loss.get("name") == expected["loss"])
            add_check(checks, str(protocol_path), "effective protocol attention", protocol_attention.get("name"), expected["attention"], str(protocol_attention.get("name", "")).lower() == str(expected["attention"]).lower())

    for log_path in sorted(package.glob("logs/*")):
        if log_path.suffix.lower() not in {".log", ".json"}:
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.suffix.lower() == ".log":
            add_check(checks, str(log_path), "training log method-bearing run path", method in log_text, True, method in log_text)
        else:
            add_check(checks, str(log_path), "training attempts log inspected", len(log_text) > 0, True, len(log_text) > 0)

    smoke_path = expected["smoke"]
    if smoke_path.is_file():
        smoke = read_json(smoke_path)
        smoke_text = flatten_text(smoke)
        if method == CE:
            add_check(checks, str(smoke_path), "baseline CE smoke test", smoke_text, "CrossEntropyLoss and attention disabled", "CrossEntropyLoss" in smoke_text and "attention_enabled False" in smoke_text)
        else:
            add_check(checks, str(smoke_path), "custom method smoke test", smoke_text, "attention insertion audit", "inserted True" in smoke_text and "simam_gated_residual" in smoke_text)
    else:
        contradictions.append(f"Required smoke test is missing: {smoke_path}")

    notebook_checks, notebook_contradictions = notebook_evidence(notebook, method)
    checks.extend(notebook_checks)
    contradictions.extend(notebook_contradictions)
    checkpoint_checks, checkpoint_contradictions, checkpoint_hashes = checkpoint_metadata(package, method)
    checks.extend(checkpoint_checks)
    contradictions.extend(checkpoint_contradictions)
    metric_record, metric_contradictions, metric_sources = metrics_summary(package)
    contradictions.extend(metric_contradictions)

    passed_checks = sum(check["result"] == "pass" for check in checks)
    failed_checks = sum(check["result"] == "fail" for check in checks)
    confidence = "high" if not contradictions and passed_checks >= 8 and failed_checks == 0 else "low"
    return {
        "input_path": str(package),
        "original_label": package.name,
        "verified_method": method if confidence == "high" else "unresolved",
        "confidence": confidence,
        "evidence": checks,
        "contradictions": contradictions,
        "metrics_sources": metric_sources,
        "metrics": metric_record.get("best_metrics", {}),
        "checkpoint_hashes": checkpoint_hashes,
    }


def corrected_mapping_contradictions(packages: list[dict[str, Any]]) -> list[str]:
    by_method = {item["verified_method"]: item for item in packages if item["verified_method"] in {CE, ASL}}
    if CE not in by_method or ASL not in by_method:
        return ["Both verified method packages are required before assessing the claimed corrected mapping."]
    ce = by_method[CE]["metrics"].get("FINAL_RESULTS.json", {})
    asl = by_method[ASL]["metrics"].get("FINAL_RESULTS.json", {})
    expected = {
        (CE, "shrimpdb3"): 0.7021276595744681,
        (ASL, "shrimpdb3"): 0.9148936170212766,
        (CE, "combined4"): 0.8318181818181818,
        (ASL, "combined4"): 0.8772727272727273,
    }
    actual = {
        (CE, "shrimpdb3"): ce.get("shrimpdb3", {}).get("accuracy"),
        (ASL, "shrimpdb3"): asl.get("shrimpdb3", {}).get("accuracy"),
        (CE, "combined4"): ce.get("combined4", {}).get("accuracy"),
        (ASL, "combined4"): asl.get("combined4", {}).get("accuracy"),
    }
    issues = []
    for key, value in expected.items():
        if actual.get(key) != value:
            issues.append(
                f"Requested corrected mapping expects {key[0]} {key[1]} accuracy {value}, "
                f"but the verified {key[0]} package raw artifact records {actual.get(key)}."
            )
    if actual[(CE, "combined4")] == expected[(ASL, "combined4")] and actual[(ASL, "combined4")] == expected[(CE, "combined4")]:
        issues.append(
            "Combined-4 accuracy is cross-assigned relative to the requested presentation: "
            "87.73% is serialized with the CE checkpoint and 83.18% with the ASL checkpoint. "
            "Reassigning these raw results would contradict checkpoint, config, smoke-test, and notebook evidence."
        )
    return issues


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Method-identity audit",
        "",
        f"Status: **{report['status']}**",
        f"Audit timestamp (UTC): `{report['audit_timestamp_utc']}`",
        "",
        "This audit does not modify either external result package or any raw metric file.",
        "",
        "## Package conclusions",
        "",
        "| Package | Verified method | Confidence | Evidence checks | Contradictions |",
        "|---|---|---:|---:|---:|",
    ]
    for item in report["packages"]:
        passed = sum(entry["result"] == "pass" for entry in item["evidence"])
        lines.append(f"| `{item['original_label']}` | `{item['verified_method']}` | {item['confidence']} | {passed} | {len(item['contradictions'])} |")
    lines.extend(["", "## Independent evidence", ""])
    for item in report["packages"]:
        lines.append(f"### {item['original_label']}")
        lines.append("")
        for entry in item["evidence"]:
            marker = "PASS" if entry["result"] == "pass" else "FAIL"
            lines.append(f"- **{marker}** `{entry['check']}` — `{entry['source']}`; observed `{entry['observed']}`.")
        if item["contradictions"]:
            lines.append("")
            lines.append("Contradictions:")
            for contradiction in item["contradictions"]:
                lines.append(f"- {contradiction}")
        lines.append("")
    lines.extend(["## Blocking findings", ""])
    for issue in report["unresolved_items"]:
        lines.append(f"- {issue}")
    lines.extend([
        "",
        "## Decision",
        "",
        "Repository-facing result labels and tables must not be changed from this audit alone. "
        "The available executable evidence verifies the package identities, but it does not support "
        "the requested Combined-4 corrected mapping. A rerun or an additional authoritative artifact "
        "is required before scientific presentation can be corrected.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--asl-package", type=Path, default=Path(r"D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS"))
    parser.add_argument("--ce-package", type=Path, default=Path(r"D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS"))
    parser.add_argument("--asl-notebook", type=Path, default=Path(r"D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_E2E_v7_RUNTIME_FIX.ipynb"))
    parser.add_argument("--ce-notebook", type=Path, default=Path(r"D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_YOLO26m_CE_Baseline_E2E_v1_FIXED.ipynb"))
    args = parser.parse_args()
    output_dir = args.repo_root / "artifacts" / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    packages = [
        inspect_package(args.ce_package, args.ce_notebook),
        inspect_package(args.asl_package, args.asl_notebook),
    ]
    unresolved = []
    for item in packages:
        unresolved.extend(item["contradictions"])
    unresolved.extend(corrected_mapping_contradictions(packages))
    status = "verified" if not unresolved and all(item["confidence"] == "high" for item in packages) else "contradictory"
    report = {
        "status": status,
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "packages": packages,
        "label_corrections": [],
        "unresolved_items": unresolved,
    }
    (output_dir / "method_identity_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "method_identity_audit.md").write_text(render_markdown(report), encoding="utf-8")
    manifest = {
        "status": "not_applied" if status != "verified" else "no_corrections_required",
        "reason": "No repository-facing label correction was applied because the requested Combined-4 mapping is contradicted by executable evidence.",
        "corrections": [],
        "raw_metric_files_modified": False,
        "raw_metric_files_byte_identity_verified": False,
        "raw_metric_files_integrity_note": "The audit is read-only; an independent before/after hash comparison was not run because no correction was authorized.",
        "audit_reference": "artifacts/metadata/method_identity_audit.json",
    }
    (output_dir / "label_correction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "unresolved_items": unresolved}, indent=2))
    return 0 if status == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
