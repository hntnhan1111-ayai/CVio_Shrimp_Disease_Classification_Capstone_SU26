#!/usr/bin/env python3
"""
Stage 1 YOLO classification baseline runner matching the uploaded run-all-models_2.ipynb YOLO protocol.

Purpose:
- Rerun all available YOLO classification baselines using the same core recipe that produced
  strong yolo26m-cls results in run-all-models_2.ipynb.
- Use local processed-images dataset directly.
- Keep YOLO class folders as the original numbered folders: "1. Healthy", "2. BG", ...
  so Ultralytics class indices match the notebook's evaluation method.
- Do not use custom loss, attention, segmentation, kagglehub, or shrimp_scripts.

Expected local dataset:
/home/drnguyenvinh/notebooks/processed-images/processed_images
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

try:
    import torch
    from ultralytics import YOLO
except Exception as exc:  # fail early with readable error
    raise RuntimeError(
        "Missing required package. Install only missing packages, e.g. `python -m pip install ultralytics scikit-learn pandas pillow tqdm`. "
        "Do not reinstall torch if CUDA already works."
    ) from exc


SEED = 42
IMG_SIZE = 224
EPOCHS = 30
PATIENCE = 15
YOLO_BATCH_SIZE = 32  # matches run-all-models_2.ipynb MICRO_BATCH_SIZE/YOLO_BATCH_SIZE
NUM_WORKERS = None  # None means do not pass workers, matching uploaded notebook behavior/Ultralytics default

DATA_DIR = Path("/home/drnguyenvinh/notebooks/processed-images/processed_images")
OUTPUT_DIR = Path("/home/drnguyenvinh/notebooks/stage1_yolo_baseline_match_runall_outputs")
YOLO_DATA_DIR = OUTPUT_DIR / "yolo_dataset"
YOLO_RUNS_DIR = OUTPUT_DIR / "yolo_runs"
REPORT_DIR = OUTPUT_DIR / "reports"
PRED_DIR = OUTPUT_DIR / "predictions"
CM_DIR = OUTPUT_DIR / "confusion_matrices"
CR_DIR = OUTPUT_DIR / "classification_reports"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
LOG_DIR = OUTPUT_DIR / "logs"

CLASS_DIRS = ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]
CLASS_NAMES = ["Healthy", "BG", "WSSV", "WSSV_BG"]
CLASS_TO_IDX = {folder: idx for idx, folder in enumerate(CLASS_DIRS)}
CLASS_DIR_TO_DISPLAY = dict(zip(CLASS_DIRS, CLASS_NAMES))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

YOLO_BASELINES = [
    "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",
    "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
    "yolo26n-cls", "yolo26s-cls", "yolo26m-cls", "yolo26l-cls", "yolo26x-cls",
]

YOLO12_OPTIONAL = ["yolo12n-cls", "yolo12s-cls", "yolo12m-cls", "yolo12l-cls", "yolo12x-cls"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def mkdirs() -> None:
    for d in [OUTPUT_DIR, YOLO_DATA_DIR, YOLO_RUNS_DIR, REPORT_DIR, PRED_DIR, CM_DIR, CR_DIR, MANIFEST_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(".", "_")


def yolo_device_arg() -> int | str:
    return 0 if torch.cuda.is_available() else "cpu"


def shell(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as exc:
        return repr(exc)


def environment_audit() -> None:
    import ultralytics

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": shell(["nvidia-smi"]),
        "dataset_root": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "seed": SEED,
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "yolo_batch_size": YOLO_BATCH_SIZE,
        "workers_argument_passed": NUM_WORKERS,
        "protocol_note": (
            "Matches run-all-models_2.ipynb YOLO baseline: native Ultralytics classification train, "
            "batch=32, epochs=30, patience=15, seed=42, imgsz=224, default optimizer=auto, "
            "cos_lr default False, auto_augment default randaugment, class folders kept as numbered source folders."
        ),
    }
    save_json(OUTPUT_DIR / "environment.json", env)
    print("Environment saved:", OUTPUT_DIR / "environment.json")
    print("Python:", sys.version.split()[0])
    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Ultralytics:", ultralytics.__version__)


def validate_and_manifest_dataset() -> pd.DataFrame:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    rows: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for folder in CLASS_DIRS:
        d = DATA_DIR / folder
        if not d.exists():
            raise FileNotFoundError(f"Missing expected class folder: {d}")
        files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        print(f"{folder}: {len(files)} images")
        for p in files:
            try:
                with Image.open(p) as img:
                    img.verify()
                readable = True
                err = ""
            except Exception as exc:
                readable = False
                err = repr(exc)
                unreadable.append({"path": str(p), "error": err})
            rows.append(
                {
                    "path": str(p),
                    "rel_path": str(p.relative_to(DATA_DIR)),
                    "class_dir": folder,
                    "display_class_name": CLASS_DIR_TO_DISPLAY[folder],
                    "project_label": CLASS_TO_IDX[folder],
                    "readable": readable,
                    "read_error": err,
                    "file_size": p.stat().st_size,
                    "md5": md5_file(p),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) != 1149:
        raise RuntimeError(f"Expected 1149 images, got {len(df)}. Check dataset path.")
    if unreadable:
        pd.DataFrame(unreadable).to_csv(MANIFEST_DIR / "unreadable_images.csv", index=False)
        raise RuntimeError(f"Found unreadable images: {len(unreadable)}. See unreadable_images.csv")

    df.to_csv(MANIFEST_DIR / "source_manifest.csv", index=False)
    counts = df.groupby("class_dir").size().reindex(CLASS_DIRS).reset_index(name="count")
    counts.to_csv(MANIFEST_DIR / "source_class_counts.csv", index=False)
    save_json(
        OUTPUT_DIR / "dataset_audit.json",
        {
            "dataset_root": str(DATA_DIR),
            "total_images": int(len(df)),
            "class_dirs": CLASS_DIRS,
            "class_names": CLASS_NAMES,
            "counts": counts.to_dict(orient="records"),
            "unreadable_count": int(len(unreadable)),
        },
    )
    return df


def create_split(df: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_csv = MANIFEST_DIR / "split_manifest.csv"
    if split_csv.exists() and not force:
        split_df = pd.read_csv(split_csv)
        train_df = split_df[split_df["split"] == "train"].copy()
        val_df = split_df[split_df["split"] == "val"].copy()
        test_df = split_df[split_df["split"] == "test"].copy()
        print("Using existing split_manifest.csv")
    else:
        train_df, tmp_df = train_test_split(
            df,
            test_size=0.30,
            stratify=df["project_label"],
            random_state=SEED,
            shuffle=True,
        )
        val_df, test_df = train_test_split(
            tmp_df,
            test_size=0.50,
            stratify=tmp_df["project_label"],
            random_state=SEED,
            shuffle=True,
        )
        parts = []
        for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
            tmp = frame.copy()
            tmp["split"] = split_name
            parts.append(tmp)
        split_df = pd.concat(parts, ignore_index=True)
        split_df.to_csv(split_csv, index=False)

    # Match uploaded run-all-models_2 split counts exactly.
    expected = {"train": 804, "val": 172, "test": 173}
    actual = split_df["split"].value_counts().to_dict()
    if actual != expected:
        raise RuntimeError(f"Split mismatch. Expected {expected}, got {actual}")

    for split_name, split_frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{split_name}: {len(split_frame)} images")
        print(split_frame["class_dir"].value_counts().reindex(CLASS_DIRS).to_dict())

    paths_by_split = {s: set(split_df.loc[split_df["split"] == s, "path"]) for s in ["train", "val", "test"]}
    assert paths_by_split["train"].isdisjoint(paths_by_split["val"])
    assert paths_by_split["train"].isdisjoint(paths_by_split["test"])
    assert paths_by_split["val"].isdisjoint(paths_by_split["test"])

    md5_by_split = {s: set(split_df.loc[split_df["split"] == s, "md5"]) for s in ["train", "val", "test"]}
    assert md5_by_split["train"].isdisjoint(md5_by_split["val"])
    assert md5_by_split["train"].isdisjoint(md5_by_split["test"])
    assert md5_by_split["val"].isdisjoint(md5_by_split["test"])

    dist = split_df.groupby(["split", "class_dir"]).size().reset_index(name="count")
    dist.to_csv(MANIFEST_DIR / "class_distribution.csv", index=False)
    return split_df, train_df, val_df, test_df


def prepare_yolo_dataset(split_df: pd.DataFrame, force: bool = False) -> None:
    manifest_csv = MANIFEST_DIR / "yolo_copy_manifest.csv"
    if YOLO_DATA_DIR.exists() and manifest_csv.exists() and not force:
        print(f"Using existing YOLO dataset: {YOLO_DATA_DIR}")
        return
    if YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)

    rows = []
    for split_name in ["train", "val", "test"]:
        part = split_df[split_df["split"] == split_name]
        for class_dir in CLASS_DIRS:
            (YOLO_DATA_DIR / split_name / class_dir).mkdir(parents=True, exist_ok=True)
        for _, row in part.iterrows():
            src = Path(row["path"])
            dst = YOLO_DATA_DIR / split_name / row["class_dir"] / src.name
            if dst.exists():
                # Match original collision logic roughly, but deterministic.
                dst = dst.with_name(f"{dst.stem}_{row['md5'][:10]}{dst.suffix}")
            shutil.copy2(src, dst)
            rows.append(
                {
                    "split": split_name,
                    "src": str(src),
                    "dst": str(dst),
                    "class_dir": row["class_dir"],
                    "display_class_name": row["display_class_name"],
                    "project_label": int(row["project_label"]),
                    "md5": row["md5"],
                }
            )
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)
    print(f"Prepared YOLO classification dataset at {YOLO_DATA_DIR}")


def probe_yolo_models(models: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in models:
        weights_name = f"{model_name}.pt"
        try:
            y = YOLO(weights_name)
            task = getattr(y, "task", "")
            available = task == "classify"
            reason = "" if available else f"task={task}, not classify"
            del y
        except Exception as exc:
            available = False
            reason = repr(exc)
        rows.append({"model": model_name, "weights": weights_name, "available": available, "reason": reason})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "model_availability_yolo.csv", index=False)
    return df


def count_params(model) -> float:
    return sum(param.numel() for param in model.parameters()) / 1e6


def build_class_order_audit(yolo_model: YOLO) -> dict[str, Any]:
    names_raw = getattr(yolo_model, "names", {})
    names = {int(k): str(v) for k, v in names_raw.items()}
    yolo_name_to_index = {name: idx for idx, name in names.items()}
    expected_present = all(class_dir in yolo_name_to_index for class_dir in CLASS_DIRS)
    yolo_index_to_project_label = {
        int(yolo_name_to_index[class_dir]): int(CLASS_TO_IDX[class_dir])
        for class_dir in CLASS_DIRS
        if class_dir in yolo_name_to_index
    }
    return {
        "project_class_dirs": CLASS_DIRS,
        "project_class_names": CLASS_NAMES,
        "project_class_to_idx": CLASS_TO_IDX,
        "ultralytics_names": names,
        "ultralytics_name_to_index": yolo_name_to_index,
        "yolo_index_to_project_label": yolo_index_to_project_label,
        "expected_numbered_folders_present_in_model_names": expected_present,
        "audit_passed": bool(expected_present and len(yolo_index_to_project_label) == 4),
        "note": "Metrics are computed in YOLO/ImageFolder index space like run-all-models_2.ipynb; predictions are also exported with project labels.",
    }


def evaluate_yolo_model(yolo_model: YOLO, eval_df: pd.DataFrame, model_key: str, split_name: str, timed: bool = False) -> dict[str, Any]:
    audit = build_class_order_audit(yolo_model)
    if not audit["audit_passed"]:
        raise RuntimeError(f"Class order audit failed for {model_key}: {audit}")

    names = {int(k): str(v) for k, v in yolo_model.names.items()}
    name_to_yolo_idx = {value: int(key) for key, value in names.items()}
    yolo_idx_to_project_label = audit["yolo_index_to_project_label"]
    source_paths = eval_df["path"].tolist()

    if timed:
        _ = yolo_model.predict(source=source_paths[:1], imgsz=IMG_SIZE, device=yolo_device_arg(), verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
    else:
        start = None

    preds = yolo_model.predict(
        source=source_paths,
        imgsz=IMG_SIZE,
        batch=YOLO_BATCH_SIZE,
        device=yolo_device_arg(),
        verbose=False,
    )

    if timed and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start if timed else None

    # Important: exactly the original notebook's metric space.
    y_true_yolo = [name_to_yolo_idx[class_dir] for class_dir in eval_df["class_dir"].tolist()]
    y_pred_yolo = [int(result.probs.top1) for result in preds]

    # Project-order labels for exported predictions and confusion report readability.
    y_true_project = [CLASS_TO_IDX[class_dir] for class_dir in eval_df["class_dir"].tolist()]
    y_pred_project = [yolo_idx_to_project_label[int(result.probs.top1)] for result in preds]

    acc = accuracy_score(y_true_yolo, y_pred_yolo)
    macro_f1 = f1_score(y_true_yolo, y_pred_yolo, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true_yolo, y_pred_yolo)
    precision, recall, _, _ = precision_recall_fscore_support(y_true_yolo, y_pred_yolo, average="macro", zero_division=0)

    cm_project = confusion_matrix(y_true_project, y_pred_project, labels=list(range(4)))
    report_project = classification_report(
        y_true_project,
        y_pred_project,
        labels=list(range(4)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pred_rows = []
    for row, result, yt_yolo, yp_yolo, yt_proj, yp_proj in zip(
        eval_df.to_dict(orient="records"), preds, y_true_yolo, y_pred_yolo, y_true_project, y_pred_project
    ):
        pred_rows.append(
            {
                "path": row["path"],
                "class_dir": row["class_dir"],
                "true_yolo_idx": yt_yolo,
                "pred_yolo_idx": yp_yolo,
                "true_project_label": yt_proj,
                "true_project_class": CLASS_NAMES[yt_proj],
                "pred_project_label": yp_proj,
                "pred_project_class": CLASS_NAMES[yp_proj],
                "pred_ultralytics_name": names[yp_yolo],
                "confidence": float(result.probs.top1conf),
            }
        )

    pd.DataFrame(pred_rows).to_csv(PRED_DIR / f"{model_key}_{split_name}_predictions.csv", index=False)
    pd.DataFrame(cm_project, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(CM_DIR / f"{model_key}_{split_name}_confusion_matrix.csv")
    pd.DataFrame(report_project).T.to_csv(CR_DIR / f"{model_key}_{split_name}_classification_report.csv")

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "cohen_kappa": float(kappa),
        "elapsed": float(elapsed) if elapsed is not None else None,
        "n": int(len(eval_df)),
        "class_order_audit": audit,
        "metrics_space": "ultralytics_yolo_index_space_matching_run_all_models_2",
    }
    save_json(REPORT_DIR / f"{model_key}_{split_name}_metrics.json", metrics)
    return metrics


def run_one_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, force: bool = False) -> dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"Training YOLO classifier: {model_name}")
    print("=" * 90)

    model_key = sanitize_name(model_name)
    model_out_dir = YOLO_RUNS_DIR / model_key
    metrics_json = model_out_dir / "match_run_all_metrics.json"
    if metrics_json.exists() and not force:
        print("SKIP completed:", model_name)
        return json.loads(metrics_json.read_text(encoding="utf-8"))

    weights_name = f"{model_name}.pt"
    status_path = model_out_dir / "status.json"
    model_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(status_path, {"status": "started", "model": model_name, "time": time.ctime()})

    try:
        train_start = time.time()
        yolo = YOLO(weights_name)
        train_kwargs = {
            "data": str(YOLO_DATA_DIR),
            "task": "classify",
            "imgsz": IMG_SIZE,
            "epochs": EPOCHS,
            "batch": YOLO_BATCH_SIZE,
            "patience": PATIENCE,
            "seed": SEED,
            "project": str(YOLO_RUNS_DIR),
            "name": model_key,
            "exist_ok": True,
            "device": yolo_device_arg(),
            "verbose": True,
        }
        if NUM_WORKERS is not None:
            train_kwargs["workers"] = int(NUM_WORKERS)
        save_json(model_out_dir / "train_kwargs.json", train_kwargs)
        yolo.train(**train_kwargs)
        train_time = time.time() - train_start

        best_path = model_out_dir / "weights" / "best.pt"
        if not best_path.exists():
            candidates = sorted(model_out_dir.glob("**/best.pt"))
            if not candidates:
                raise FileNotFoundError(f"Could not locate YOLO best checkpoint for {model_name}")
            best_path = candidates[-1]
        last_path = model_out_dir / "weights" / "last.pt"

        best_yolo = YOLO(str(best_path))
        class_order_audit = build_class_order_audit(best_yolo)
        save_json(model_out_dir / "class_order_audit.json", class_order_audit)
        if not class_order_audit["audit_passed"]:
            raise RuntimeError("Class order audit failed after training.")

        val_metrics = evaluate_yolo_model(best_yolo, val_df, model_key, "val", timed=False)
        test_metrics = evaluate_yolo_model(best_yolo, test_df, model_key, "test", timed=True)
        inf_time = test_metrics["elapsed"]
        params_m = count_params(best_yolo.model)
        size_mb = best_path.stat().st_size / (1024 * 1024)

        result = {
            "Model": model_name,
            "Backend Name": weights_name,
            "Parameters (M)": round(params_m, 2),
            "Model Size (MB)": round(size_mb, 3),
            "Training Time (s)": round(train_time, 1),
            "Val F1-Score": round(val_metrics["macro_f1"], 4),
            "Val Accuracy": round(val_metrics["accuracy"], 4),
            "Test Accuracy": round(test_metrics["accuracy"], 4),
            "Test F1-Score": round(test_metrics["macro_f1"], 4),
            "Cohen Kappa": round(test_metrics["cohen_kappa"], 4),
            "Inference Time (s)": round(inf_time, 2),
            "FPS": round(len(test_df) / max(inf_time, 1e-12), 1),
            "Latency (ms)": round((inf_time / len(test_df)) * 1000, 2),
            "Checkpoint Path": str(best_path),
            "Last Checkpoint Path": str(last_path) if last_path.exists() else "",
            "status": "completed",
            "protocol": "match_run_all_models_2_yolo_baseline",
            "loss": "native_ultralytics_classification_ce_default",
            "attention": "native_yolo_no_extra_attention",
            "auto_augment": "randaugment_default_from_ultralytics_train_args",
        }
        save_json(metrics_json, result)
        save_json(status_path, {"status": "completed", "model": model_name, "time": time.ctime()})
        del yolo, best_yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result

    except Exception as exc:
        err = traceback.format_exc()
        save_json(status_path, {"status": "failed", "model": model_name, "error": repr(exc), "traceback": err})
        print("FAILED:", model_name, repr(exc))
        return {
            "Model": model_name,
            "Backend Name": weights_name,
            "status": "failed",
            "failure_reason": repr(exc),
            "traceback": err,
        }


def collect_results(models: list[str]) -> None:
    rows = []
    for model_name in models:
        p = YOLO_RUNS_DIR / sanitize_name(model_name) / "match_run_all_metrics.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    raw_csv = REPORT_DIR / "match_run_all_yolo_baseline_raw.csv"
    df.to_csv(raw_csv, index=False)

    if df.empty:
        print("No completed/failed model rows found.")
        return

    completed = df[df.get("status", "") == "completed"].copy()
    if not completed.empty:
        completed = completed.sort_values(
            by=["Test F1-Score", "Cohen Kappa", "Test Accuracy", "Latency (ms)", "Model Size (MB)"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
        completed.to_csv(REPORT_DIR / "match_run_all_yolo_baseline_ranking.csv", index=False)
        completed.to_json(REPORT_DIR / "match_run_all_yolo_baseline_ranking.json", orient="records", indent=2)
        completed.to_excel(REPORT_DIR / "match_run_all_yolo_baseline_ranking.xlsx", index=False)
        md = ["# Match run-all-models_2 YOLO Baseline Ranking", "", completed.to_markdown(index=False)]
        (REPORT_DIR / "match_run_all_yolo_baseline_ranking.md").write_text("\n".join(md), encoding="utf-8")
        metrics_cols = [
            "Model", "Parameters (M)", "Model Size (MB)", "Training Time (s)", "Val F1-Score",
            "Test Accuracy", "Test F1-Score", "Cohen Kappa", "Inference Time (s)", "FPS", "Latency (ms)",
            "Checkpoint Path",
        ]
        completed[metrics_cols].to_csv(REPORT_DIR / "match_run_all_yolo_baseline_metrics_table.csv", index=False)
        print("\nRanking:")
        print(completed[["Model", "Test F1-Score", "Test Accuracy", "Cohen Kappa", "Latency (ms)", "FPS", "Model Size (MB)"]].to_string(index=False))

    failed = df[df.get("status", "") != "completed"].copy()
    failed.to_csv(REPORT_DIR / "match_run_all_yolo_failed_or_skipped.csv", index=False)

    save_json(
        OUTPUT_DIR / "final_run_summary.json",
        {
            "planned": len(models),
            "completed": int(len(completed)) if not completed.empty else 0,
            "failed": int(len(failed)) if not failed.empty else 0,
            "output_dir": str(OUTPUT_DIR),
            "ranking_csv": str(REPORT_DIR / "match_run_all_yolo_baseline_ranking.csv"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-split", action="store_true", help="Recreate split_manifest.csv")
    parser.add_argument("--force-yolo-dataset", action="store_true", help="Recreate yolo_dataset")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun models even if metrics JSON exists")
    parser.add_argument("--include-yolo12", action="store_true", help="Also probe/run yolo12*-cls if available")
    parser.add_argument("--only", nargs="*", default=None, help="Optional model names, e.g. yolo26m-cls yolov8m-cls")
    parser.add_argument("--list-only", action="store_true", help="Only print planned models and exit")
    args = parser.parse_args()

    set_seed(SEED)
    mkdirs()
    environment_audit()

    models = list(YOLO_BASELINES)
    if args.include_yolo12:
        models.extend(YOLO12_OPTIONAL)
    if args.only:
        requested = set(args.only)
        models = [m for m in models if m in requested]
        missing = requested - set(models)
        if missing:
            raise ValueError(f"Unknown --only models: {sorted(missing)}")
    print("Planned YOLO baselines:", models)
    if args.list_only:
        return

    df = validate_and_manifest_dataset()
    split_df, train_df, val_df, test_df = create_split(df, force=args.force_split)
    prepare_yolo_dataset(split_df, force=args.force_yolo_dataset)

    availability = probe_yolo_models(models)
    print("\nAvailability:")
    print(availability.to_string(index=False))
    available_models = availability.loc[availability["available"], "model"].tolist()
    skipped = availability.loc[~availability["available"]].copy()
    skipped.to_csv(REPORT_DIR / "model_unavailable_yolo.csv", index=False)

    for model_name in available_models:
        result = run_one_model(model_name, train_df, val_df, test_df, force=args.force_rerun)
        # update partial after every model
        collect_results(models)

    # Include skipped unavailable rows in failed/skipped report
    collect_results(models)
    failed_path = REPORT_DIR / "match_run_all_yolo_failed_or_skipped.csv"
    failed = pd.read_csv(failed_path) if failed_path.exists() and failed_path.stat().st_size > 0 else pd.DataFrame()
    if not skipped.empty:
        skipped_rows = skipped.rename(columns={"model": "Model", "weights": "Backend Name", "reason": "failure_reason"})
        skipped_rows["status"] = "skipped_unavailable"
        failed = pd.concat([failed, skipped_rows], ignore_index=True, sort=False)
        failed.to_csv(failed_path, index=False)

    print("DONE. Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
