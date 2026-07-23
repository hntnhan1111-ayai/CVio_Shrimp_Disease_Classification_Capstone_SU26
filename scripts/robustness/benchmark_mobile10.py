#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from recsra.mobile_corruptions import NOISES, PROTOCOL, apply_corruption, stable_seed

BASELINE_SHA = "b9e30aa76f819126c7e6e9f3d3007a74d0839ad1f95941978a6a214484eb219d"
RECSRA_SHA = "c1652101bb870a0b174b569ac47a70bb8cfafe119577e1ca2fe2de87f57824ff"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def discover_weight(roots: list[Path], expected_sha: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        if sha256_file(p) != expected_sha:
            raise RuntimeError(f"SHA mismatch for {p}")
        return p
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            try:
                if sha256_file(p) == expected_sha:
                    return p.resolve()
            except OSError:
                continue
    raise FileNotFoundError(f"Weight with SHA {expected_sha} was not found")


def dataset_root_valid(root: Path) -> bool:
    return (root / "images" / "test").is_dir() and (root / "labels" / "test").is_dir()


def discover_dataset(roots: list[Path], explicit: str | None) -> tuple[Path, Path]:
    if explicit:
        p = Path(explicit).resolve()
        if p.is_file():
            d = yaml.safe_load(p.read_text())
            root = Path(d.get("path", p.parent))
            if not root.is_absolute():
                root = (p.parent / root).resolve()
            if not dataset_root_valid(root):
                root = p.parent
            if not dataset_root_valid(root):
                raise RuntimeError(f"Dataset root invalid: {root}")
            return p, root
        if dataset_root_valid(p):
            return p / "data.yaml", p
        raise FileNotFoundError(p)

    candidates: list[tuple[int, Path, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for test_images in root.rglob("images/test"):
            ds = test_images.parent.parent
            if not dataset_root_valid(ds):
                continue
            count = sum(1 for p in test_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            yaml_path = ds / "data.yaml"
            score = 1000 - abs(count - 111)
            if yaml_path.is_file():
                score += 50
            candidates.append((score, yaml_path, ds))
    if not candidates:
        raise FileNotFoundError("No dataset root containing images/test and labels/test was found")
    candidates.sort(reverse=True, key=lambda x: (x[0], str(x[2])))
    return candidates[0][1], candidates[0][2]


def write_resolved_yaml(root: Path, original_yaml: Path, out: Path, test_images: Path | None = None, test_labels: Path | None = None) -> None:
    names = {0: "BG", 1: "WSSV"}
    if original_yaml.is_file():
        try:
            src = yaml.safe_load(original_yaml.read_text()) or {}
            names = src.get("names", names)
        except Exception:
            pass
    payload = {
        "path": str(root),
        "train": str(root / "images" / "train"),
        "val": str(root / "images" / "val"),
        "test": str(test_images or (root / "images" / "test")),
        "nc": 2,
        "names": names,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def copy_labels(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.txt"):
        shutil.copy2(p, dst / p.name)


def generate_condition(ds_root: Path, original_yaml: Path, work: Path, noise_id: str, severity: int, seed: int) -> Path:
    condition = work / f"{noise_id}_S{severity}"
    images_out = condition / "images" / "test"
    labels_out = condition / "labels" / "test"
    if images_out.is_dir() and any(images_out.iterdir()) and (condition / "data.yaml").is_file():
        return condition / "data.yaml"
    shutil.rmtree(condition, ignore_errors=True)
    images_out.mkdir(parents=True, exist_ok=True)
    copy_labels(ds_root / "labels" / "test", labels_out)
    images = sorted(p for p in (ds_root / "images" / "test").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError("Test image set is empty")
    for src in images:
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {src}")
        s = stable_seed(src.name, noise_id, severity, seed)
        out = apply_corruption(image, noise_id, severity, s)
        if not cv2.imwrite(str(images_out / src.name), out):
            raise RuntimeError(f"Cannot write {images_out / src.name}")
    write_resolved_yaml(ds_root, original_yaml, condition / "data.yaml", images_out, labels_out)
    return condition / "data.yaml"


def run_pair(python: Path, package: Path, baseline: Path, recsra: Path, data_yaml: Path, output: Path, tag: str) -> tuple[dict, dict]:
    """Evaluate both frozen checkpoints sequentially on physical RTX 4090 GPU 0."""
    condition_dir = output / "metrics_json" / tag
    condition_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for label, model in (("baseline", baseline), ("recsra", recsra)):
        out_json = condition_dir / f"{label}.json"
        log_path = condition_dir / f"{label}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CVIO_EVAL_GPU", "0")
        env["PYTHONPATH"] = str(package) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            str(python), str(package / "eval_worker.py"),
            "--model", str(model),
            "--data", str(data_yaml),
            "--out", str(out_json),
            "--run-name", f"{tag}_{label}",
            "--imgsz", "1536",
            "--iou", "0.55",
            "--conf", "0.0005",
            "--max-det", "600",
            "--workers", "1",
        ]
        print(f"[EVAL] {tag} | {label} | physical GPU {env['CUDA_VISIBLE_DEVICES']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not out_json.is_file():
            raise RuntimeError(
                f"{label} evaluation failed for {tag}; "
                f"returncode={completed.returncode}. See {log_path}"
            )
        results[label] = json.loads(out_json.read_text(encoding="utf-8"))

    return results["baseline"], results["recsra"]


def read_yolo_labels(label: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not label.is_file():
        return rows
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts[:5])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        rows.append((int(cls), x1, y1, x2, y2))
    return rows


def draw_gt(image: np.ndarray, boxes, names) -> np.ndarray:
    out = image.copy()
    for cls, x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 3)
        cv2.putText(out, f"GT {names.get(cls, cls)}", (int(x1), max(20, int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0), 2, cv2.LINE_AA)
    return out


def predict_image(python: Path, package: Path, model: Path, image: Path, gpu: str, out_json: Path) -> dict:
    code = r'''
import json, sys
from ultralytics import YOLO
m=YOLO(sys.argv[1])
r=m.predict(sys.argv[2], imgsz=1536, conf=0.05, iou=0.55, max_det=600, device=0, verbose=False)[0]
rows=[]
if r.boxes is not None:
    for xyxy, conf, cls in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
        rows.append({"xyxy":xyxy,"conf":conf,"cls":int(cls)})
json.dump({"predictions":rows}, open(sys.argv[3],"w"))
'''
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(package) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([str(python), "-c", code, str(model), str(image), str(out_json)], env=env, check=True)
    return json.loads(out_json.read_text())


def draw_pred(image: np.ndarray, preds: dict, names, title: str) -> np.ndarray:
    out = image.copy()
    for item in preds.get("predictions", []):
        x1,y1,x2,y2 = item["xyxy"]
        cls = int(item["cls"])
        conf = float(item["conf"])
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (0,165,255), 3)
        cv2.putText(out, f"{names.get(cls,cls)} {conf:.2f}", (int(x1),max(20,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (20,36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(out, title, (20,36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20,20,20), 1, cv2.LINE_AA)
    return out


def choose_sample(ds_root: Path) -> Path:
    images = sorted(p for p in (ds_root / "images" / "test").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    scored = []
    for p in images:
        label = ds_root / "labels" / "test" / f"{p.stem}.txt"
        count = len(label.read_text().splitlines()) if label.is_file() else 0
        scored.append((count, p.name, p))
    scored.sort(reverse=True)
    return scored[0][2]


def build_samples(top: pd.DataFrame, ds_root: Path, baseline: Path, recsra: Path, package: Path, python: Path, output: Path, seed: int) -> list[dict]:
    sample_src = choose_sample(ds_root)
    names = {0:"BG",1:"WSSV"}
    clean = cv2.imread(str(sample_src))
    h,w = clean.shape[:2]
    label_path = ds_root / "labels" / "test" / f"{sample_src.stem}.txt"
    gt = read_yolo_labels(label_path, w, h)
    clean_gt = draw_gt(clean, gt, names)
    records=[]
    samples_dir=output/"top5_samples"
    samples_dir.mkdir(parents=True,exist_ok=True)
    for _, row in top.iterrows():
        nid=str(row["noise_id"])
        sev=3
        corr=apply_corruption(clean,nid,sev,stable_seed(sample_src.name,nid,sev,seed))
        corr_path=samples_dir/f"{nid}_severity3_input.jpg"
        cv2.imwrite(str(corr_path),corr)
        corr_gt=draw_gt(corr,gt,names)
        bpred=predict_image(python,package,baseline,corr_path,os.environ.get("CVIO_EVAL_GPU","0"),samples_dir/f"{nid}_baseline_pred.json")
        rpred=predict_image(python,package,recsra,corr_path,os.environ.get("CVIO_EVAL_GPU","0"),samples_dir/f"{nid}_recsra_pred.json")
        panels=[
            draw_pred(clean_gt,{"predictions":[]},names,"Clean + ground truth"),
            draw_pred(corr_gt,{"predictions":[]},names,f"{nid} severity 3 + ground truth"),
            draw_pred(corr,bpred,names,"YOLO11s baseline predictions"),
            draw_pred(corr,rpred,names,"YOLO11s-RECSRA predictions"),
        ]
        target_h=min(900,max(p.shape[0] for p in panels))
        resized=[]
        for p in panels:
            scale=target_h/p.shape[0]
            resized.append(cv2.resize(p,(int(p.shape[1]*scale),target_h)))
        montage=np.hstack(resized)
        out_path=samples_dir/f"{nid}_{row['noise_name']}_thesis_sample.jpg"
        cv2.imwrite(str(out_path),montage)
        records.append({"noise_id":nid,"noise_name":row["noise_name"],"severity":sev,"source_image":sample_src.name,"sample_path":str(out_path)})
    return records


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline")
    ap.add_argument("--recsra")
    ap.add_argument("--dataset")
    ap.add_argument("--output",default="outputs/mobile10")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--keep-corrupted",action="store_true")
    ap.add_argument("--device",default="0")
    args=ap.parse_args()

    package=Path(__file__).resolve().parent
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    roots=[Path("/home/drnguyenvinh/notebooks"),Path.cwd()]
    baseline=discover_weight(roots,BASELINE_SHA,args.baseline)
    recsra=discover_weight(roots,RECSRA_SHA,args.recsra)
    original_yaml,ds_root=discover_dataset(roots,args.dataset)
    resolved_yaml=output/"resolved_clean_data.yaml"
    write_resolved_yaml(ds_root,original_yaml,resolved_yaml)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; RTX 4090 evaluation requires CUDA")
    gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not gpu_names:
        raise RuntimeError("No CUDA GPU is available")
    device_index=int(args.device)
    if device_index < 0 or device_index >= len(gpu_names):
        raise RuntimeError(f"Invalid physical GPU index {device_index}; visible GPUs={gpu_names}")
    os.environ["CVIO_EVAL_GPU"] = str(device_index)
    print(f"[GPU MODE] Sequential evaluation on physical GPU {device_index}: {gpu_names[device_index]}", flush=True)
    runtime={
        "python":sys.version,
        "torch":torch.__version__,
        "cuda":torch.version.cuda,
        "gpu_names":gpu_names,
        "baseline":str(baseline),
        "baseline_sha256":sha256_file(baseline),
        "recsra":str(recsra),
        "recsra_sha256":sha256_file(recsra),
        "dataset_root":str(ds_root),
        "test_context":"individual shrimp photographed onshore using a consumer mobile phone",
        "evaluation_hardware":gpu_names[int(args.device)],
        "evaluation_mode":f"single-GPU sequential on physical GPU {args.device}",
    }
    (output/"runtime_and_inputs.json").write_text(json.dumps(runtime,indent=2),encoding="utf-8")
    shutil.copy2(package/"noise_protocol.json",output/"noise_protocol.json")

    rows=[]
    bclean,rclean=run_pair(Path(sys.executable),package,baseline,recsra,resolved_yaml,output,"CLEAN")
    for label,metrics in (("baseline",bclean),("recsra",rclean)):
        rows.append({"condition":"clean","noise_id":"CLEAN","noise_name":"clean","severity":0,"model":label,**metrics})
    work=output/"temporary_corrupted_testsets"
    for nid,item in NOISES.items():
        for sev in range(1,6):
            print(f"[CONDITION] {nid} {item['name']} severity={sev}",flush=True)
            data_yaml=generate_condition(ds_root,original_yaml,work,nid,sev,args.seed)
            bm,rm=run_pair(Path(sys.executable),package,baseline,recsra,data_yaml,output,f"{nid}_S{sev}")
            for label,metrics in (("baseline",bm),("recsra",rm)):
                rows.append({"condition":"corruption","noise_id":nid,"noise_name":item["name"],"severity":sev,"model":label,**metrics})
            if not args.keep_corrupted:
                shutil.rmtree(data_yaml.parent,ignore_errors=True)

    df=pd.DataFrame(rows)
    df.to_csv(output/"all_clean_and_50_condition_metrics.csv",index=False)
    df[df.noise_id.eq("CLEAN")].to_csv(output/"clean_metrics.csv",index=False)
    corr=df[~df.noise_id.eq("CLEAN")].copy()
    summary=(corr.groupby(["noise_id","noise_name","model"],as_index=False)
        .agg(mean_mAP50=("mAP50","mean"),mean_mAP50_95=("mAP50_95","mean"),mean_mAP75=("mAP75","mean"),mean_precision=("precision","mean"),mean_recall=("recall","mean")))
    pivot=summary.pivot(index=["noise_id","noise_name"],columns="model")
    flat=pd.DataFrame(index=pivot.index).reset_index()
    for metric in ["mean_mAP50","mean_mAP50_95","mean_mAP75","mean_precision","mean_recall"]:
        flat[f"baseline_{metric}"]=pivot[(metric,"baseline")].values
        flat[f"recsra_{metric}"]=pivot[(metric,"recsra")].values
        flat[f"delta_{metric}"]=flat[f"recsra_{metric}"]-flat[f"baseline_{metric}"]
    clean_b=float(bclean["mAP50_95"]); clean_r=float(rclean["mAP50_95"])
    flat["baseline_retention_mAP50_95"]=flat["baseline_mean_mAP50_95"]/clean_b
    flat["recsra_retention_mAP50_95"]=flat["recsra_mean_mAP50_95"]/clean_r
    flat["delta_retention_mAP50_95"]=flat["recsra_retention_mAP50_95"]-flat["baseline_retention_mAP50_95"]
    flat["qualifies_recsra_better"]=flat["delta_mean_mAP50_95"]>0
    flat=flat.sort_values(["qualifies_recsra_better","delta_mean_mAP50_95","delta_mean_mAP50","delta_retention_mAP50_95","delta_mean_recall"],ascending=[False,False,False,False,False])
    flat.to_csv(output/"noise_family_summary.csv",index=False)
    top=flat[flat.qualifies_recsra_better].head(5).copy()
    top.to_csv(output/"top5_selected_noise_families.csv",index=False)

    samples=build_samples(top,ds_root,baseline,recsra,package,Path(sys.executable),output,args.seed)
    (output/"top5_sample_metadata.json").write_text(json.dumps(samples,indent=2),encoding="utf-8")

    plt.figure(figsize=(11,6))
    x=np.arange(len(flat))
    plt.bar(x-0.2,flat["baseline_mean_mAP50_95"],width=0.4,label="YOLO11s baseline")
    plt.bar(x+0.2,flat["recsra_mean_mAP50_95"],width=0.4,label="YOLO11s-RECSRA")
    plt.xticks(x,flat["noise_id"],rotation=0)
    plt.ylabel("Mean mAP50-95 across severities 1-5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output/"noise_family_mAP50_95_comparison.png",dpi=220)
    plt.close()

    report=[
        "# RECSRA Mobile-On-Shore Corruption Benchmark",
        "",
        "## Context",
        "Individual shrimp are removed from the pond, placed onshore and photographed with consumer mobile phones.",
        "",
        "## Evaluation protocol",
        "- Two immutable checkpoints verified by SHA-256.",
        "- Baseline and RECSRA were evaluated sequentially on the same RTX 4090 GPU 0.",
        "- Both use imgsz=1536, IoU=0.55, conf=0.0005, max_det=600, batch=1 and FP32.",
        "- Ten mobile-realistic corruption families, five deterministic severity levels each.",
        "- Bounding boxes remain unchanged because no geometric transform is applied.",
        "",
        "## Clean metrics",
        f"- Baseline: mAP50={bclean['mAP50']:.9f}, mAP50-95={bclean['mAP50_95']:.9f}",
        f"- RECSRA: mAP50={rclean['mAP50']:.9f}, mAP50-95={rclean['mAP50_95']:.9f}",
        "",
        "## Top qualifying noise families",
    ]
    if top.empty:
        report.append("No noise family satisfied RECSRA mean mAP50-95 > baseline. No top-five claim is made.")
    else:
        for rank,(_,r) in enumerate(top.iterrows(),1):
            report.append(f"{rank}. {r['noise_id']} {r['noise_name']}: delta mean mAP50-95={r['delta_mean_mAP50_95']:+.9f}")
    (output/"FINAL_REPORT.md").write_text("\n".join(report),encoding="utf-8")

    manifest=[]
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.suffix not in {".zip"} and not p.name.endswith(".sha256"):
            manifest.append({"path":str(p.relative_to(output)),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    pd.DataFrame(manifest).to_csv(output/"FILE_MANIFEST_SHA256.csv",index=False)
    final_zip=output/"RECSRA_mobile10_final_results.zip"
    with zipfile.ZipFile(final_zip,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file() and p!=final_zip and not p.name.endswith(".sha256"):
                z.write(p,p.relative_to(output))
    sha=sha256_file(final_zip)
    (output/f"{final_zip.name}.sha256").write_text(f"{sha}  {final_zip.name}\n")
    print("[FINAL ZIP]",final_zip)
    print("[SHA256]",sha)

if __name__=="__main__":
    main()
