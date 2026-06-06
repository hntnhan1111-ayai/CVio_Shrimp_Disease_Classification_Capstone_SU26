from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

STAGE1_CE_REF = {
    "variant": "Stage1 CE fixed reference",
    "loss_key": "baseline_ce",
    "attention_key": "none_baseline",
    "is_run": False,
    "status": "completed_reference_not_rerun",
    "test_macro_f1": 0.8902,
    "test_accuracy": 0.8902,
    "cohen_kappa": 0.8505,
}
ASL_LDAM_REF = {
    "variant": "ASL-LDAM fixed reference",
    "loss_key": "asl_ldam_margin",
    "attention_key": "none_baseline",
    "is_run": False,
    "status": "completed_reference_not_rerun",
    "test_macro_f1": 0.8991,
    "test_accuracy": 0.9017,
    "cohen_kappa": 0.8661,
}
PRIOR_BEST_REF = {
    "variant": "ASL-LDAM + sparse_learnable_fusion_cbam fixed reference",
    "loss_key": "asl_ldam_margin",
    "attention_key": "sparse_learnable_fusion_cbam",
    "is_run": False,
    "status": "external_reference_not_rerun",
    "test_macro_f1": 0.9006,
    "test_accuracy": 0.9017,
    "cohen_kappa": 0.8658,
}

VARIANTS = [
    {"variant": "CE + CBAM", "variant_key": "ce_cbam", "loss_key": "baseline_ce", "attention_key": "cbam"},
    {"variant": "ASL + CBAM", "variant_key": "asl_cbam", "loss_key": "asl_single_label", "attention_key": "cbam"},
    {"variant": "ASL-LDAM + CBAM", "variant_key": "asl_ldam_cbam", "loss_key": "asl_ldam_margin", "attention_key": "cbam"},
]


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n[RUN]", " ".join(map(str, cmd)), flush=True)
    print("[CWD]", cwd, flush=True)
    print("[LOG]", log_path, flush=True)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        rc = p.wait()
        f.write(f"\n[returncode] {rc}\n")
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}. See {log_path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def copy_project(src: Path, dst: Path, force: bool) -> Path:
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        return src
    if dst.exists() and force:
        shutil.rmtree(dst)
    if not dst.exists():
        print(f"Copying project to isolated folder:\n  SRC={src}\n  DST={dst}", flush=True)
        ignore = shutil.ignore_patterns(".git", "__pycache__", ".ipynb_checkpoints", "runs", "outputs", "wandb", "*.pt", "*.pth", "*.onnx", "*.engine", "yolo_fixed_dataset", "processed_images")
        shutil.copytree(src, dst, ignore=ignore)
    return dst


def write_kagglehub_override(project_dir: Path, dataset_root: Path | None) -> None:
    if dataset_root is None:
        return
    dataset_root = dataset_root.resolve()
    return_path = dataset_root.parent if dataset_root.name == "processed_images" else dataset_root
    shim = f'''from pathlib import Path

def dataset_download(handle: str, *args, **kwargs) -> str:
    expected = "uynnhy/processed-images"
    if handle != expected:
        raise RuntimeError(f"Local kagglehub override only supports {{expected}}, got {{handle}}")
    path = Path(r"{return_path}")
    print("Using local KaggleHub override for", handle)
    print("Path to dataset files:", path)
    return str(path)
'''
    (project_dir / "kagglehub.py").write_text(shim, encoding="utf-8")


def validate_project(project_dir: Path) -> None:
    required = [
        project_dir / "shrimp_scripts" / "models_yolo.py",
        project_dir / "shrimp_scripts" / "dataset.py",
        project_dir / "shrimp_scripts" / "attention.py",
        project_dir / "experiments" / "asl_custom_loss_screening" / "run_asl_custom_screen.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing full CVio project code. Missing:\n" + "\n".join(missing))


def prepare_dataset(project_dir: Path, output_dir: Path, env: dict[str, str]) -> None:
    run_cmd([sys.executable, "shrimp_scripts/run_01_prepare_dataset.py", "--output_dir", str(output_dir), "--resume", "--progress"], cwd=project_dir, log_path=output_dir / "notebook_command_logs" / "010_prepare_dataset.log", env=env)


def self_check(project_dir: Path, output_dir: Path, env: dict[str, str]) -> None:
    script = output_dir / "self_check_cbam_and_losses.py"
    script.write_text(f'''from __future__ import annotations
import json
from pathlib import Path
from shrimp_scripts.attention import attention_keys, self_test_attention_modules
keys = attention_keys()
print("attention_keys=", keys)
if "cbam" not in keys:
    raise RuntimeError("cbam not registered: " + repr(keys))
res = self_test_attention_modules(attention_key="cbam")
Path(r"{output_dir / 'cbam_attention_self_test.json'}").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
print("cbam_self_test_status=", res.get("status"))
if res.get("status") != "passed":
    raise RuntimeError("CBAM self-test failed")
''', encoding="utf-8")
    run_cmd([sys.executable, str(script)], cwd=project_dir, log_path=output_dir / "notebook_command_logs" / "020_cbam_self_check.log", env=env)
    loss_script = project_dir / "experiments" / "asl_custom_loss_screening" / "run_asl_custom_screen.py"
    for loss_key in ["asl_single_label", "asl_ldam_margin"]:
        run_cmd([sys.executable, str(loss_script), "--output_dir", str(output_dir), "--backend", "yolo", "--model", "yolo26m-cls", "--loss", loss_key, "--self_check_yolo_loss"], cwd=project_dir, log_path=output_dir / "notebook_command_logs" / f"021_loss_self_check_{loss_key}.log", env=env)


def train_variant(project_dir: Path, output_dir: Path, variant: dict[str, str], epochs: int, batch: int, workers: int, imgsz: int, device: str) -> dict[str, Any]:
    sys.path.insert(0, str(project_dir))
    from shrimp_scripts.dataset import load_yolo_manifest
    from shrimp_scripts.models_yolo import train_yolo_with_fallback, yolo_run_id
    from shrimp_scripts.utils import read_json as project_read_json

    condition = {
        "condition_key": f"{variant['loss_key']}_{variant['attention_key']}_randaugment_seed42",
        "loss_key": variant["loss_key"],
        "randaugment": True,
        "attention_key": variant["attention_key"],
        "epochs": int(epochs),
        "batch": int(batch),
        "workers": int(workers),
        "imgsz": int(imgsz),
        "device": str(device),
        "seed": 42,
        "experiment_key": "yolo26m_cbam_loss_trio",
        "experiment_group": "yolo26m_cbam_loss_trio",
        "variant": variant["variant"],
        "variant_key": variant["variant_key"],
    }
    yolo_manifest = load_yolo_manifest(output_dir)
    result = train_yolo_with_fallback("yolo26m-cls", condition, yolo_manifest, output_dir, resume=True, smoke_test=False, progress_enabled=True)
    run_id = yolo_run_id("yolo26m-cls", condition)
    run_dir = output_dir / "runs" / run_id
    metrics = project_read_json(run_dir / "metrics.json", default={})
    record = {"run_id": run_id, "run_dir": str(run_dir), "variant": variant["variant"], "variant_key": variant["variant_key"], "loss_key": variant["loss_key"], "attention_key": variant["attention_key"], "condition": condition, "result": result, "metrics": metrics}
    write_json(output_dir / "variant_results" / f"{variant['variant_key']}.json", record)
    return record


def normalize_row(variant: dict[str, str], record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
    row = {"variant": variant["variant"], "variant_key": variant["variant_key"], "loss_key": variant["loss_key"], "attention_key": variant["attention_key"], "is_run": True, "run_id": record.get("run_id"), "run_dir": record.get("run_dir")}
    if isinstance(metrics, dict):
        row.update(metrics)
    if "cohen_kappa" not in row and "test_kappa" in row:
        row["cohen_kappa"] = row.get("test_kappa")
    return row


def validity(row: dict[str, Any]) -> tuple[str, str]:
    errors = []
    for col in ["test_macro_f1", "test_accuracy", "cohen_kappa"]:
        try:
            v = float(row.get(col))
            if v != v:
                errors.append(f"{col}=nan")
        except Exception:
            errors.append(f"{col}=missing")
    run_dir = Path(str(row.get("run_dir", "")))
    if not run_dir.exists():
        errors.append("run_dir_missing")
    else:
        if not (run_dir / "class_order_audit.json").exists():
            errors.append("class_order_audit_missing")
        if not (run_dir / "attention_module_audit.json").exists():
            errors.append("attention_module_audit_missing")
    return ("valid" if not errors else "invalid", "; ".join(errors))


def reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    report_dir = output_dir / "final_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [dict(STAGE1_CE_REF), dict(ASL_LDAM_REF), dict(PRIOR_BEST_REF), *rows]
    df = pd.DataFrame(all_rows)
    for c in ["test_macro_f1", "test_accuracy", "cohen_kappa"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["delta_f1_vs_stage1_ce"] = df["test_macro_f1"] - float(STAGE1_CE_REF["test_macro_f1"])
    df["delta_f1_vs_asl_ldam_ref"] = df["test_macro_f1"] - float(ASL_LDAM_REF["test_macro_f1"])
    run_df = df[df["is_run"].astype(str).str.lower().eq("true")].copy() if "is_run" in df.columns else pd.DataFrame()
    valid_df = run_df[run_df["validity_status"].astype(str).eq("valid")].copy() if not run_df.empty and "validity_status" in run_df.columns else pd.DataFrame()
    if not valid_df.empty:
        valid_df = valid_df.sort_values(["test_macro_f1", "cohen_kappa", "test_accuracy"], ascending=[False, False, False]).reset_index(drop=True)
        valid_df.insert(0, "rank", range(1, len(valid_df) + 1))
    df.to_csv(report_dir / "all_references_and_cbam_loss_trio.csv", index=False)
    valid_df.to_csv(report_dir / "ranking_cbam_loss_trio.csv", index=False)
    try:
        with pd.ExcelWriter(report_dir / "cbam_loss_trio_summary.xlsx", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="all_rows", index=False)
            valid_df.to_excel(writer, sheet_name="ranking", index=False)
    except Exception as exc:
        print("xlsx_warning=", repr(exc), flush=True)
    summary = {"planned_runs": len(rows), "valid_ranked_runs": int(len(valid_df)), "reports": str(report_dir)}
    write_json(report_dir / "summary_cbam_loss_trio.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def make_review_zip(output_dir: Path) -> Path:
    zip_path = output_dir.parent / "yolo26m_cbam_loss_trio_for_review.zip"
    exclude_suffixes = {".pt", ".pth", ".onnx", ".engine", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in output_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() in exclude_suffixes:
                continue
            rel = p.relative_to(output_dir.parent)
            if "weights" in rel.parts or "yolo_dataset" in rel.parts or "__pycache__" in rel.parts:
                continue
            z.write(p, rel.as_posix())
    print("review_zip=", zip_path, flush=True)
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--work_project_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--local_dataset_root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force_project_copy", action="store_true")
    parser.add_argument("--stop_on_failure", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = copy_project(args.project_dir, args.work_project_dir, args.force_project_copy)
    validate_project(project_dir)
    write_kagglehub_override(project_dir, args.local_dataset_root)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    env["PYTHONPATH"] = str(project_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    run_cmd([sys.executable, "-m", "compileall", "shrimp_scripts", "experiments"], cwd=project_dir, log_path=output_dir / "notebook_command_logs" / "000_compileall.log", env=env)
    prepare_dataset(project_dir, output_dir, env)
    self_check(project_dir, output_dir, env)

    start = max(1, args.start) - 1
    selected = VARIANTS[start:]
    if args.limit > 0:
        selected = selected[: args.limit]
    write_json(output_dir / "run_plan.json", {"variants": selected, "epochs": args.epochs, "batch": args.batch, "workers": args.workers, "imgsz": args.imgsz, "references": [STAGE1_CE_REF, ASL_LDAM_REF, PRIOR_BEST_REF]})

    rows = []
    for v in selected:
        print("\n" + "=" * 100, flush=True)
        print("TRAIN", v, flush=True)
        try:
            rec = train_variant(project_dir, output_dir, v, args.epochs, args.batch, args.workers, args.imgsz, args.device)
            row = normalize_row(v, rec)
            status, errs = validity(row)
            row["validity_status"] = status
            row["validity_errors"] = errs
            rows.append(row)
            write_json(output_dir / "variant_results" / f"{v['variant_key']}_audit_summary.json", row)
        except Exception as exc:
            row = {"variant": v["variant"], "variant_key": v["variant_key"], "loss_key": v["loss_key"], "attention_key": v["attention_key"], "is_run": True, "validity_status": "failed", "validity_errors": repr(exc)}
            rows.append(row)
            write_json(output_dir / "variant_results" / f"{v['variant_key']}_failed.json", row)
            if args.stop_on_failure:
                reports(output_dir, rows)
                make_review_zip(output_dir)
                raise
    reports(output_dir, rows)
    valid_n = sum(1 for r in rows if r.get("validity_status") == "valid")
    make_review_zip(output_dir)
    if valid_n == 0:
        raise RuntimeError("No valid completed runs. Do not use as experiment evidence.")


if __name__ == "__main__":
    main()
