from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

STAGE1_BASELINE = {
    "variant_group": "fixed_stage1_reference",
    "variant_type": "ce_baseline_reference",
    "model": "yolo26m-cls",
    "loss_key": "baseline_ce",
    "attention_key": "none_baseline",
    "source": "corrected Stage 1 native Ultralytics baseline matched to run-all-models_2 protocol",
    "status": "completed_reference_not_rerun",
    "test_macro_f1": 0.8902,
    "test_accuracy": 0.8902,
    "cohen_kappa": 0.8505,
    "notes": "Fixed CE/no-added-attention baseline. Do not rerun CE in this final improvement workflow.",
}

LOSS_KEYS = [
    "asl_single_label",
    "class_weighted_asl",
    "coinfection_weighted_asl",
    "boundary_weighted_asl",
    "confusion_aware_negative_asl",
    "soft_target_coinfection_asl",
    "attribute_projection_asl",
    "adaptive_gamma_asl",
    "asl_ldam_margin",
    "dangerous_confidence_penalty_asl",
    "coinfection_logit_adjusted_asl",
]

ATTENTION_KEYS = [
    "eca",
    "simam",
    "coordatt",
    "cbam",
    "se",
    "ema",
    "triplet",
    "c2psa_or_psa",
]


def mask_secret(text: str) -> str:
    import re
    text = re.sub(r"ghp_[A-Za-z0-9_]+", "ghp_***", str(text))
    text = re.sub(r"hf_[A-Za-z0-9_]+", "hf_***", text)
    return text


def run_cmd(cmd: list[str], cwd: Path | None, log_path: Path, env: dict[str, str] | None = None, check: bool = True) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    print("\n[RUN]", mask_secret(" ".join(map(str, cmd))), flush=True)
    print("[CWD]", cwd or Path.cwd(), flush=True)
    print("[LOG]", log_path, flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            safe = mask_secret(line)
            print(safe, end="", flush=True)
            f.write(safe)
        rc = p.wait()
        f.write(f"\n[returncode] {rc}\n")
    if check and rc != 0:
        raise RuntimeError(f"Command failed rc={rc}: {' '.join(map(str, cmd))}. See {log_path}")
    return rc


def run_python(args: list[str], project_dir: Path, log_dir: Path, log_name: str, env: dict[str, str], check: bool = True) -> int:
    return run_cmd([sys.executable, *map(str, args)], cwd=project_dir, log_path=log_dir / log_name, env=env, check=check)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_metrics_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "cohen_kappa" not in out and "test_kappa" in out:
        out["cohen_kappa"] = out.get("test_kappa")
    if "test_kappa" not in out and "cohen_kappa" in out:
        out["test_kappa"] = out.get("cohen_kappa")
    return out


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def find_best(frame: pd.DataFrame, metric_col: str = "test_macro_f1") -> dict[str, Any] | None:
    if frame is None or frame.empty or metric_col not in frame.columns:
        return None
    work = frame.copy()
    for col in [metric_col, "cohen_kappa", "test_accuracy", "latency_ms_image", "model_size_mb"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[work[metric_col].notna()].copy()
    if work.empty:
        return None
    sort_cols = [c for c in [metric_col, "cohen_kappa", "test_accuracy", "latency_ms_image", "model_size_mb"] if c in work.columns]
    asc = [False, False, False, True, True][: len(sort_cols)]
    return work.sort_values(sort_cols, ascending=asc, na_position="last").iloc[0].to_dict()


def read_loss_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = output_dir / "asl_custom_screening_summary.csv"
    ranked_path = output_dir / "asl_custom_screening_ranked.csv"
    if ranked_path.exists():
        df = pd.read_csv(ranked_path)
    elif summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        return pd.DataFrame(), pd.DataFrame()
    if "model" in df.columns:
        df = df[df["model"].astype(str).eq("yolo26m-cls")].copy()
    if "screen_backend" in df.columns:
        df = df[df["screen_backend"].astype(str).eq("yolo")].copy()
    if "loss_key" in df.columns:
        df = df[df["loss_key"].astype(str) != "baseline_ce"].copy()
    for c in ["test_macro_f1", "test_accuracy", "cohen_kappa"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if not df.empty and "test_macro_f1" in df.columns:
        df = df.sort_values(["test_macro_f1", "cohen_kappa", "test_accuracy"], ascending=[False, False, False], na_position="last").reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    failures_path = output_dir / "asl_custom_screening_failures.csv"
    failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    return df, failures


def read_attention_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_path = output_dir / "attention_ranking.csv"
    results_path = output_dir / "attention_screening_results.csv"
    if ranking_path.exists():
        df = pd.read_csv(ranking_path)
    elif results_path.exists():
        df = pd.read_csv(results_path)
    else:
        return pd.DataFrame(), pd.DataFrame()
    if "attention_key" in df.columns:
        df = df[df["attention_key"].astype(str) != "none_baseline"].copy()
    for c in ["test_macro_f1", "test_accuracy", "cohen_kappa"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if not df.empty and "test_macro_f1" in df.columns:
        df = df.sort_values(["test_macro_f1", "cohen_kappa", "test_accuracy"], ascending=[False, False, False], na_position="last").reset_index(drop=True)
        if "rank" in df.columns:
            df = df.drop(columns=["rank"])
        df.insert(0, "rank", range(1, len(df) + 1))
    failed_path = output_dir / "attention_failed_or_skipped.csv"
    failures = pd.read_csv(failed_path) if failed_path.exists() else pd.DataFrame()
    return df, failures


def run_combined_variant(project_dir: Path, output_dir: Path, log_dir: Path, env: dict[str, str], best_loss: str, best_attention: str, epochs: int) -> dict[str, Any]:
    script_path = output_dir / "run_best_loss_attention_combined_variant.py"
    script = f'''
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(r"{project_dir}").resolve()
sys.path.insert(0, str(ROOT))

from shrimp_scripts.dataset import load_yolo_manifest
from shrimp_scripts.models_yolo import train_yolo_with_fallback, yolo_run_id
from shrimp_scripts.utils import read_json, write_json

output_dir = Path(r"{output_dir}")
model = "yolo26m-cls"
condition = {{
    "condition_key": "best_loss_{best_loss}_best_attention_{best_attention}_randaugment",
    "loss_key": "{best_loss}",
    "randaugment": True,
    "attention_key": "{best_attention}",
    "epochs": int({epochs}),
    "experiment_key": "final_yolo26m_improvement_against_stage1_baseline",
    "experiment_group": "final_yolo26m_improvement_against_stage1_baseline",
    "screening_note": "Final combined best-loss + best-attention variant; CE baseline is fixed Stage 1 reference and is not rerun.",
}}
yolo_manifest = load_yolo_manifest(output_dir)
result = train_yolo_with_fallback(model, condition, yolo_manifest, output_dir, resume=True, smoke_test=False, progress_enabled=True)
run_id = yolo_run_id(model, condition)
run_dir = output_dir / "runs" / run_id
metrics = read_json(run_dir / "metrics.json", default={{}})
write_json(output_dir / "final_best_loss_attention_combined_result.json", {{"run_id": run_id, "result": result, "metrics": metrics, "run_dir": str(run_dir)}})
print(json.dumps({{"run_id": run_id, "result": result, "metrics": metrics, "run_dir": str(run_dir)}}, indent=2, default=str))
'''
    script_path.write_text(script, encoding="utf-8")
    run_cmd([sys.executable, str(script_path)], cwd=project_dir, log_path=log_dir / "070_combined_best_loss_attention.log", env=env, check=True)
    result = load_json(output_dir / "final_best_loss_attention_combined_result.json", default={})
    metrics = normalize_metrics_row(result.get("metrics", {}))
    return {"run_id": result.get("run_id"), "variant_group": "combined_best_loss_attention", "variant_type": "best_loss_plus_best_attention", "loss_key": best_loss, "attention_key": best_attention, **metrics}


def build_final_workbook(output_dir: Path, baseline: dict[str, Any]) -> Path:
    reports = output_dir / "final_reports"
    reports.mkdir(parents=True, exist_ok=True)
    baseline_df = pd.DataFrame([baseline])
    loss_df, loss_failures = read_loss_results(output_dir)
    att_df, att_failures = read_attention_results(output_dir)
    combined_result = load_json(output_dir / "final_best_loss_attention_combined_result.json", default={})
    combined_metrics = normalize_metrics_row(combined_result.get("metrics", {})) if combined_result else {}
    combined_df = pd.DataFrame([{**combined_metrics, "run_id": combined_result.get("run_id"), "variant_group": "combined_best_loss_attention"}]) if combined_metrics else pd.DataFrame()

    final_rows = []
    final_rows.append({**baseline, "comparison_to_stage1_macro_f1": 0.0, "comparison_to_stage1_accuracy": 0.0, "comparison_to_stage1_kappa": 0.0})
    for _, row in loss_df.iterrows():
        d = row.to_dict()
        d["variant_group"] = "loss_variant"
        d["variant_type"] = "asl_single_label" if d.get("loss_key") == "asl_single_label" else "custom_asl_loss"
        final_rows.append(d)
    for _, row in att_df.iterrows():
        d = row.to_dict()
        d["variant_group"] = "attention_variant"
        d["variant_type"] = "added_attention"
        final_rows.append(d)
    if not combined_df.empty:
        d = combined_df.iloc[0].to_dict()
        d["variant_group"] = "combined_best_loss_attention"
        d["variant_type"] = "best_loss_plus_best_attention"
        final_rows.append(d)
    final = pd.DataFrame(final_rows)

    for metric in ["test_macro_f1", "test_accuracy", "cohen_kappa"]:
        if metric in final.columns:
            final[metric] = pd.to_numeric(final[metric], errors="coerce")
    final["delta_macro_f1_vs_stage1"] = final.get("test_macro_f1", pd.Series(dtype=float)) - float(baseline["test_macro_f1"])
    final["delta_accuracy_vs_stage1"] = final.get("test_accuracy", pd.Series(dtype=float)) - float(baseline["test_accuracy"])
    final["delta_kappa_vs_stage1"] = final.get("cohen_kappa", pd.Series(dtype=float)) - float(baseline["cohen_kappa"])
    if "test_macro_f1" in final.columns:
        final_rank = final.sort_values(["test_macro_f1", "cohen_kappa", "test_accuracy"], ascending=[False, False, False], na_position="last").reset_index(drop=True)
        final_rank.insert(0, "final_rank", range(1, len(final_rank) + 1))
    else:
        final_rank = final

    best_loss = find_best(loss_df)
    best_attention = find_best(att_df)
    dashboard = pd.DataFrame([
        {"item": "fixed_stage1_yolo26m_ce_baseline", "value": "Macro-F1=0.8902, Accuracy=0.8902, Kappa=0.8505"},
        {"item": "ce_baseline_rerun", "value": "skipped by design"},
        {"item": "best_loss_key", "value": best_loss.get("loss_key") if best_loss else "missing"},
        {"item": "best_loss_macro_f1", "value": best_loss.get("test_macro_f1") if best_loss else None},
        {"item": "best_attention_key", "value": best_attention.get("attention_key") if best_attention else "missing"},
        {"item": "best_attention_macro_f1", "value": best_attention.get("test_macro_f1") if best_attention else None},
        {"item": "planned_loss_runs_excluding_ce", "value": len(LOSS_KEYS)},
        {"item": "planned_attention_runs_excluding_none_baseline", "value": len(ATTENTION_KEYS)},
    ])

    xlsx = reports / "final_yolo26m_improvements_against_stage1_baseline.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        dashboard.to_excel(writer, sheet_name="Dashboard", index=False)
        baseline_df.to_excel(writer, sheet_name="Stage1_Baseline", index=False)
        loss_df.to_excel(writer, sheet_name="Loss_Variants", index=False)
        att_df.to_excel(writer, sheet_name="Attention_Variants", index=False)
        combined_df.to_excel(writer, sheet_name="Best_Combined", index=False)
        final_rank.to_excel(writer, sheet_name="Final_Ranking", index=False)
        loss_failures.to_excel(writer, sheet_name="Loss_Failures", index=False)
        att_failures.to_excel(writer, sheet_name="Attention_Failures", index=False)
    baseline_df.to_csv(reports / "stage1_baseline_anchor.csv", index=False)
    loss_df.to_csv(reports / "loss_variants_filtered.csv", index=False)
    att_df.to_csv(reports / "attention_variants_filtered.csv", index=False)
    combined_df.to_csv(reports / "best_combined_variant.csv", index=False)
    final_rank.to_csv(reports / "final_yolo26m_improvements_ranking.csv", index=False)
    write_json(reports / "final_yolo26m_improvements_summary.json", {
        "baseline": baseline,
        "best_loss": best_loss,
        "best_attention": best_attention,
        "xlsx": str(xlsx),
        "loss_rows": len(loss_df),
        "attention_rows": len(att_df),
        "combined_rows": len(combined_df),
    })
    print("Created workbook:", xlsx)
    print("Final ranking preview:")
    keep = [c for c in ["final_rank", "variant_group", "variant_type", "model", "loss_key", "attention_key", "test_macro_f1", "test_accuracy", "cohen_kappa", "delta_macro_f1_vs_stage1"] if c in final_rank.columns]
    print(final_rank[keep].head(30).to_string(index=False))
    return xlsx


def zip_results(output_dir: Path, project_dir: Path) -> Path:
    zip_path = output_dir.parent / "final_yolo26m_improvements_against_stage1_baseline_for_review.zip"
    if zip_path.exists():
        zip_path.unlink()
    wanted_suffix = {".csv", ".json", ".md", ".txt", ".log", ".xlsx", ".py"}
    skip_parts = {"weights", "ultralytics_train", "yolo_dataset", "__pycache__"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base in [output_dir]:
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if set(p.parts) & skip_parts:
                    continue
                if p.suffix.lower() not in wanted_suffix:
                    continue
                if p.stat().st_size > 25_000_000:
                    continue
                zf.write(p, p.relative_to(output_dir.parent))
    print("Created ZIP:", zip_path, "size_MB", round(zip_path.stat().st_size / (1024 * 1024), 3))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv26m improvement variants against fixed Stage 1 CE baseline.")
    parser.add_argument("--project_dir", default="/home/drnguyenvinh/notebooks/Improving Lightweight Shrimp Disease Classification with Co-Infection-Aware Losses and RandAugment")
    parser.add_argument("--output_dir", default="/home/drnguyenvinh/notebooks/final_yolo26m_improvements_stage1_reference_outputs")
    parser.add_argument("--patch_script", default="/home/drnguyenvinh/notebooks/patch_yolo_attention_injection_v2.py")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_attention_patch", action="store_true")
    parser.add_argument("--skip_losses", action="store_true")
    parser.add_argument("--skip_attention", action="store_true")
    parser.add_argument("--skip_combined", action="store_true")
    parser.add_argument("--no_zip", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = output_dir / "notebook_command_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not project_dir.exists():
        raise FileNotFoundError(f"PROJECT_DIR does not exist: {project_dir}")

    env = {
        "PYTHONPATH": str(project_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "SHRIMP_DATASET_ROOT": os.environ.get("SHRIMP_DATASET_ROOT", "/home/drnguyenvinh/notebooks/processed-images/processed_images"),
    }

    write_json(output_dir / "stage1_yolo26m_baseline_anchor.json", STAGE1_BASELINE)
    pd.DataFrame([STAGE1_BASELINE]).to_csv(output_dir / "stage1_yolo26m_baseline_anchor.csv", index=False)
    write_json(output_dir / "run_plan.json", {
        "baseline_reference": STAGE1_BASELINE,
        "loss_keys_to_run_excluding_ce": LOSS_KEYS,
        "attention_keys_to_run_excluding_none_baseline": ATTENTION_KEYS,
        "epochs": int(args.epochs),
        "project_dir": str(project_dir),
        "output_dir": str(output_dir),
    })

    run_python(["-m", "compileall", "shrimp_scripts", "experiments"], project_dir, log_dir, "000_compileall_before.log", env)

    if not args.skip_prepare:
        run_python(["shrimp_scripts/run_01_prepare_dataset.py", "--output_dir", str(output_dir), "--resume", "--progress"], project_dir, log_dir, "010_prepare_dataset.log", env)

    if not args.skip_attention_patch:
        patch_script = Path(args.patch_script).expanduser().resolve()
        if not patch_script.exists():
            raise FileNotFoundError(f"Patch script not found: {patch_script}. Upload patch_yolo_attention_injection_v2.py to /home/drnguyenvinh/notebooks.")
        run_cmd([sys.executable, str(patch_script), "--project_dir", str(project_dir), "--run_compileall"], cwd=Path.cwd(), log_path=log_dir / "020_patch_attention_v2.log", env=env, check=True)

    if not args.skip_losses:
        for loss_key in LOSS_KEYS:
            run_python(["experiments/asl_custom_loss_screening/run_asl_custom_screen.py", "--output_dir", str(output_dir), "--backend", "yolo", "--model", "yolo26m-cls", "--loss", loss_key, "--self_check_yolo_loss"], project_dir, log_dir, f"030_loss_self_check_{loss_key}.log", env)
        for loss_key in LOSS_KEYS:
            run_python(["experiments/asl_custom_loss_screening/run_asl_custom_screen.py", "--output_dir", str(output_dir), "--backend", "yolo", "--model", "yolo26m-cls", "--loss", loss_key, "--validate_resume"], project_dir, log_dir, f"040_loss_validate_{loss_key}.log", env)
            run_python(["experiments/asl_custom_loss_screening/run_asl_custom_screen.py", "--output_dir", str(output_dir), "--backend", "yolo", "--model", "yolo26m-cls", "--loss", loss_key, "--resume", "--progress"], project_dir, log_dir, f"041_loss_train_{loss_key}.log", env)
        run_python(["experiments/asl_custom_loss_screening/collect_asl_custom_results.py", "--output_dir", str(output_dir), "--backend", "yolo", "--model", "yolo26m-cls"], project_dir, log_dir, "049_loss_collect.log", env)

    if not args.skip_attention:
        run_python(["experiments/yolo_attention_screening/run_yolo_attention_screen.py", "--output_dir", str(output_dir), "--epochs", str(args.epochs), "--self_test_attention_modules"], project_dir, log_dir, "050_attention_module_self_test.log", env)
        run_python(["experiments/yolo_attention_screening/run_yolo_attention_screen.py", "--output_dir", str(output_dir), "--epochs", str(args.epochs), "--self_test_model_injection"], project_dir, log_dir, "051_attention_model_injection_self_test.log", env)
        for attention_key in ATTENTION_KEYS:
            run_python(["experiments/yolo_attention_screening/run_yolo_attention_screen.py", "--output_dir", str(output_dir), "--epochs", str(args.epochs), "--only_attention", attention_key, "--validate_resume"], project_dir, log_dir, f"060_attention_validate_{attention_key}.log", env)
            run_python(["experiments/yolo_attention_screening/run_yolo_attention_screen.py", "--output_dir", str(output_dir), "--epochs", str(args.epochs), "--only_attention", attention_key, "--resume", "--progress"], project_dir, log_dir, f"061_attention_train_{attention_key}.log", env)
        run_python(["experiments/yolo_attention_screening/collect_yolo_attention_results.py", "--output_dir", str(output_dir), "--epochs", str(args.epochs)], project_dir, log_dir, "069_attention_collect.log", env)

    # Build preliminary summary before combined so we can identify best loss/attention.
    build_final_workbook(output_dir, STAGE1_BASELINE)
    loss_df, _loss_failures = read_loss_results(output_dir)
    att_df, _att_failures = read_attention_results(output_dir)
    best_loss = find_best(loss_df)
    best_attention = find_best(att_df)

    if not args.skip_combined:
        if not best_loss or not best_loss.get("loss_key"):
            raise RuntimeError("Cannot run combined variant: no valid best loss found.")
        if not best_attention or not best_attention.get("attention_key"):
            raise RuntimeError("Cannot run combined variant: no valid best attention found.")
        run_combined_variant(project_dir, output_dir, log_dir, env, str(best_loss["loss_key"]), str(best_attention["attention_key"]), int(args.epochs))

    build_final_workbook(output_dir, STAGE1_BASELINE)
    if not args.no_zip:
        zip_results(output_dir, project_dir)


if __name__ == "__main__":
    main()
