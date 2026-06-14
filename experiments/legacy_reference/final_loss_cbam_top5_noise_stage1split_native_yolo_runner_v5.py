#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, random, shutil, subprocess, sys, time, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

IMAGE_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
CLASS_SETS = [
    ['00_Healthy','01_BG','02_WSSV','03_WSSV_BG'],
    ['1. Healthy','2. BG','3. WSSV','4. WSSV_BG'],
    ['Healthy','BG','WSSV','WSSV_BG'],
]

STAGE1_CE_REFERENCE = dict(
    variant_key='baseline_ce_stage1_reference_metrics_only',
    model='stage1_yolo26m_cls_reference',
    loss_key='native_ultralytics_ce',
    attention_key='none',
    split_name='clean_reference',
    test_macro_f1=0.8902,
    test_accuracy=0.8902,
    cohen_kappa=0.8505,
    status='fixed_stage1_reference_metrics_not_used_for_noisy_eval'
)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'motion_blur', 'defocus_blur', 'low_light', 'contrast_reduction', 'jpeg_compression'
]
SEVERITY_PARAMS = {
    'gaussian_noise': {1: {'sigma': 10}, 2: {'sigma': 30}, 3: {'sigma': 60}},
    'shot_noise': {1: {'scale': 60}, 2: {'scale': 30}, 3: {'scale': 15}},
    'impulse_noise': {1: {'prob': 0.01}, 2: {'prob': 0.03}, 3: {'prob': 0.07}},
    'speckle_noise': {1: {'std': 0.08}, 2: {'std': 0.18}, 3: {'std': 0.35}},
    'motion_blur': {1: {'kernel': 5}, 2: {'kernel': 15}, 3: {'kernel': 25}},
    'defocus_blur': {1: {'radius': 2}, 2: {'radius': 4}, 3: {'radius': 7}},
    'low_light': {1: {'factor': 0.75}, 2: {'factor': 0.50}, 3: {'factor': 0.30}},
    'contrast_reduction': {1: {'factor': 0.70}, 2: {'factor': 0.45}, 3: {'factor': 0.25}},
    'jpeg_compression': {1: {'quality': 70}, 2: {'quality': 40}, 3: {'quality': 20}},
}

def log(x): print(x, flush=True)
def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
def read_json(p: Path, default=None):
    try: return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception: return default

def safe_model_name(x: str) -> str:
    return str(x).replace('/', '_').replace(' ', '_').replace('.', '_')

def run(cmd, cwd: Path, log_path: Path, env=None, check=True):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    e = os.environ.copy(); e.update(env or {})
    log('\n[RUN] ' + ' '.join(map(str, cmd))); log('[CWD] ' + str(cwd)); log('[LOG] ' + str(log_path))
    with log_path.open('w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=str(cwd), env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout
        for line in p.stdout:
            print(line, end='', flush=True); f.write(line)
        rc = p.wait(); f.write(f'\n[returncode] {rc}\n')
    if check and rc:
        raise RuntimeError(f'command failed rc={rc}: {cmd}')
    return rc

def resolve_dataset_root(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    candidates = [p, p/'processed_images', p/'processed-images', p/'processed-images'/'processed_images']
    if p.exists():
        candidates += [x for x in p.glob('*') if x.is_dir()] + [x for x in p.glob('*/*') if x.is_dir()]
    for c in candidates:
        if c.exists() and any(all((c/d).is_dir() for d in dirs) for dirs in CLASS_SETS):
            return c.resolve()
    raise FileNotFoundError(f'Could not resolve dataset root with class folders from: {p}')

def detect_source_classes(root: Path):
    for dirs in CLASS_SETS:
        if all((root/d).is_dir() for d in dirs): return dirs
    raise FileNotFoundError(f'Missing expected class folders under {root}')

def count_images(d: Path):
    return sum(1 for x in d.rglob('*') if x.is_file() and x.suffix.lower() in IMAGE_EXTS)

def find_yolo_root(out: Path) -> Path:
    candidates = [out/'yolo_classification_dataset_seed42', out/'yolo_group_split_hardlink_dataset', out/'yolo_group_split_symlink_dataset']
    for c in candidates:
        if (c/'train').is_dir() and (c/'val').is_dir() and (c/'test').is_dir(): return c
    for c in out.rglob('*'):
        if c.is_dir() and (c/'train').is_dir() and (c/'val').is_dir() and (c/'test').is_dir(): return c
    raise FileNotFoundError(f'Could not find prepared YOLO split folder under {out}')

def find_test_classes(yolo_root: Path):
    test = yolo_root/'test'
    for dirs in CLASS_SETS:
        if all((test/d).is_dir() for d in dirs): return dirs
    dirs = sorted([x.name for x in test.iterdir() if x.is_dir()])
    if len(dirs) == 4: return dirs
    raise FileNotFoundError(f'Expected 4 class dirs under {test}, got {dirs}')

def import_project(project: Path):
    project = Path(project).expanduser().resolve()
    # Critical for this v4 runner: ASL-LDAM and CE-effective losses may live in different
    # copied project folders. Python caches imported shrimp_scripts modules, so unload them
    # before switching project roots.
    for name in list(sys.modules):
        if name == 'shrimp_scripts' or name.startswith('shrimp_scripts.'):
            del sys.modules[name]
    sys.path = [
        p for p in sys.path
        if not Path(str(p)).name.startswith("YOLO26M_")
    ]
    sys.path.insert(0, str(project))
    try:
        import shrimp_scripts.models_yolo as my
        for n in ['register_yolo_safe_globals','register_custom_yolo_safe_globals','register_safe_globals']:
            fn = getattr(my, n, None)
            if callable(fn):
                try: fn()
                except Exception: pass
    except Exception as e:
        log(f'WARN import_project({project}): {e!r}')

def resolve_loss_key(project: Path, requested: str, fallbacks: list[str]) -> str:
    keys=[]
    for k in [requested] + fallbacks:
        if k and k not in keys: keys.append(k)
    files = list((project/'shrimp_scripts').rglob('*.py')) + list((project/'experiments').rglob('*.py'))
    texts=[]
    for f in files:
        try: texts.append(f.read_text(encoding='utf-8', errors='ignore'))
        except Exception: pass
    for k in keys:
        if any(k in t for t in texts): return k
    return requested

def best_pt(run_dir: Path):
    c = list(run_dir.glob('**/weights/best.pt')) + list(run_dir.glob('**/best.pt'))
    c = [x for x in c if x.is_file()]
    if not c: return None
    c.sort(key=lambda x: -x.stat().st_mtime)
    return c[0]

def train_native_ce_baseline(out: Path, yolo_root: Path, args) -> dict:
    from ultralytics import YOLO
    summ = out/'variant_summaries'/'baseline_ce_native_ultralytics.json'
    old = read_json(summ, {})
    if old.get('checkpoint') and Path(old['checkpoint']).is_file() and not args.force:
        log('SKIP existing native CE baseline checkpoint')
        return old
    log('\n'+'='*110); log(f'TRAIN BASELINE CE: native Ultralytics {args.model_name}'); log('='*110)
    run_project = out/'native_ce_yolo_runs'
    model = YOLO(str(args.yolo_weights))
    t0 = time.time()
    train_kwargs = dict(
        data=str(yolo_root), task='classify', imgsz=int(args.imgsz), epochs=int(args.epochs),
        batch=int(args.batch), patience=int(args.patience), seed=int(args.seed), deterministic=bool(args.deterministic),
        project=str(run_project), name=f'baseline_ce_native_ultralytics_{safe_model_name(args.model_name)}_seed{args.seed}', exist_ok=True,
        device=str(args.device), workers=int(args.workers), verbose=True, amp=True, cache=False,
        plots=False, save=True, save_period=-1, optimizer='AdamW', lr0=float(args.lr0), lrf=0.01,
        cos_lr=True, pretrained=True,
    )
    model.train(**train_kwargs)
    ckpt = run_project/f'baseline_ce_native_ultralytics_{safe_model_name(args.model_name)}_seed{args.seed}'/'weights'/'best.pt'
    if not ckpt.is_file(): ckpt = best_pt(run_project)
    row = dict(variant_key='baseline_ce_native_ultralytics', model=args.model_name, loss_key='native_ultralytics_ce',
               attention_key='none', description=f'Baseline CE rerun using native Ultralytics {args.model_name} train()',
               checkpoint=str(ckpt) if ckpt else None, status='trained' if ckpt else 'missing_checkpoint',
               train_elapsed_s=time.time()-t0, train_kwargs=train_kwargs)
    write_json(summ, row)
    return row

def train_custom_variant(project: Path, out: Path, variant: dict, args) -> dict:
    variant_project = Path(variant.get('project_dir') or project).expanduser().resolve()
    import_project(variant_project)
    from shrimp_scripts.dataset import load_yolo_manifest
    from shrimp_scripts.models_yolo import train_yolo_with_fallback, yolo_run_id

    cond = dict(
        condition_key=variant['variant_key'],
        variant_key=variant['variant_key'],
        loss_key=variant['loss_key'],
        attention_key=variant['attention_key'],
        randaugment=True,
        epochs=int(args.epochs),
        seed=int(args.seed),
        experiment_group='final_loss_cbam_top5_noise_v5_stage1split',
        model_name=args.model_name,
    )
    run_id = yolo_run_id(args.model_name, cond)
    summ = out/'variant_summaries'/f"{variant['variant_key']}.json"
    old = read_json(summ, {})
    if old.get('checkpoint') and Path(old['checkpoint']).is_file() and not args.force:
        log(f"SKIP existing {variant['variant_key']}")
        return old

    log('\n'+'='*110)
    log('TRAIN CUSTOM VARIANT '+json.dumps({k:v for k,v in variant.items() if k != 'project_dir'}, default=str))
    log(f"PROJECT={variant_project}")
    log('='*110)

    result = train_yolo_with_fallback(
        args.model_name,
        cond,
        load_yolo_manifest(out),
        out,
        resume=True,
        smoke_test=False,
        progress_enabled=True,
    )

    # Different project copies may save under slightly different folder names. Search the
    # expected run id first, then the whole output/runs tree. Since variants train sequentially,
    # newest best.pt is the correct fallback if the exact run_id path does not exist.
    run_dir = out/'runs'/run_id
    ckpt = best_pt(run_dir)
    if ckpt is None:
        ckpt = best_pt(out/'runs')
    row = {
        **variant,
        'project_dir': str(variant_project),
        'model': args.model_name,
        'condition': cond,
        'run_id': run_id,
        'run_dir': str(run_dir),
        'checkpoint': str(ckpt) if ckpt else None,
        'train_result': result,
        'status': 'trained' if ckpt else 'missing_checkpoint',
    }
    write_json(summ, row)
    return row

def motion_blur_array(arr: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    img = Image.fromarray(arr)
    # simple horizontal kernel via repeated resize/filter fallback: use scipy-free convolution
    kernel = np.zeros((k, k), dtype=np.float32); kernel[k//2, :] = 1.0 / k
    pad = k//2
    padded = np.pad(arr.astype(np.float32), ((pad,pad),(pad,pad),(0,0)), mode='edge')
    out = np.zeros_like(arr, dtype=np.float32)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            out[y,x] = (padded[y:y+k, x:x+k] * kernel[...,None]).sum(axis=(0,1))
    return np.clip(out,0,255).astype(np.uint8)

def disk_blur(img: Image.Image, radius: int) -> Image.Image:
    # PIL Gaussian blur is used as a stable defocus approximation.
    return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))

def apply_corruption(img: Image.Image, corruption: str, severity: int, rng: np.random.Generator) -> Image.Image:
    img = img.convert('RGB')
    arr = np.asarray(img).astype(np.float32)
    p = SEVERITY_PARAMS[corruption][severity]
    if corruption == 'gaussian_noise':
        out = np.clip(arr + rng.normal(0, p['sigma'], arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    if corruption == 'shot_noise':
        scale = float(p['scale'])
        out = rng.poisson(np.clip(arr,0,255) / 255.0 * scale) / scale * 255.0
        return Image.fromarray(np.clip(out,0,255).astype(np.uint8))
    if corruption == 'impulse_noise':
        out = arr.copy(); prob = float(p['prob'])
        mask = rng.random(arr.shape[:2]) < prob
        salt = rng.random(arr.shape[:2]) < 0.5
        out[mask & salt] = 255; out[mask & ~salt] = 0
        return Image.fromarray(np.clip(out,0,255).astype(np.uint8))
    if corruption == 'speckle_noise':
        out = arr + arr * rng.normal(0, float(p['std']), arr.shape)
        return Image.fromarray(np.clip(out,0,255).astype(np.uint8))
    if corruption == 'motion_blur':
        return Image.fromarray(motion_blur_array(arr.astype(np.uint8), int(p['kernel'])))
    if corruption == 'defocus_blur':
        return disk_blur(img, int(p['radius']))
    if corruption == 'low_light':
        return ImageEnhance.Brightness(img).enhance(float(p['factor']))
    if corruption == 'contrast_reduction':
        return ImageEnhance.Contrast(img).enhance(float(p['factor']))
    if corruption == 'jpeg_compression':
        import io
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=int(p['quality'])); buf.seek(0)
        return Image.open(buf).convert('RGB')
    raise ValueError(corruption)

def collect(root: Path, class_dirs: list[str]) -> Tuple[list[Path], list[int]]:
    test = root/'test' if (root/'test').is_dir() else root
    paths=[]; y=[]
    for i, cls in enumerate(class_dirs):
        xs = sorted([x for x in (test/cls).rglob('*') if x.is_file() and x.suffix.lower() in IMAGE_EXTS])
        paths += xs; y += [i]*len(xs)
    return paths, y

def make_corrupted_dataset(clean_test: Path, out: Path, class_dirs: list[str], corruption: str, severity: int, seed: int, force=False, max_examples=4):
    root = out/'top9_corrupted_test_sets'/f'{corruption}_s{severity}'
    dst = root/'test'; done = root/'_DONE.json'
    if done.is_file() and not force: return root
    if dst.exists(): shutil.rmtree(dst)
    rng = np.random.default_rng(seed)
    examples = out/'example_corruptions'/f'{corruption}_s{severity}'
    if examples.exists() and force: shutil.rmtree(examples)
    ex_count=0
    for cls in class_dirs:
        for src in sorted([x for x in (clean_test/cls).rglob('*') if x.is_file() and x.suffix.lower() in IMAGE_EXTS]):
            rel = src.relative_to(clean_test/cls); tgt = dst/cls/rel.with_suffix('.jpg'); tgt.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(src).convert('RGB')
            corr = apply_corruption(im, corruption, severity, rng)
            corr.save(tgt, quality=95)
            if ex_count < max_examples:
                examples.mkdir(parents=True, exist_ok=True)
                canvas = Image.new('RGB', (im.width*2, im.height), 'white')
                canvas.paste(im, (0,0)); canvas.paste(corr, (im.width,0))
                d = ImageDraw.Draw(canvas); d.text((8,8), 'clean', fill='black'); d.text((im.width+8,8), f'{corruption} s{severity}', fill='black')
                canvas.save(examples/f'COMPARE__{cls}__{src.stem}.jpg', quality=95); ex_count += 1
    write_json(done, {'corruption':corruption, 'severity':severity, 'params':SEVERITY_PARAMS[corruption][severity], 'source':str(clean_test), 'class_dirs':class_dirs})
    return root

def evaluate_checkpoint(project: Path, ckpt: Path, root: Path, class_dirs: list[str], split: str, args, outdir: Path):
    import_project(project)
    from ultralytics import YOLO
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix, classification_report
    paths, y_true = collect(root, class_dirs)
    if not paths: raise RuntimeError(f'No test images found under {root}')
    model = YOLO(str(ckpt)); y_pred=[]; rows=[]; t=time.time(); bs=int(args.eval_batch)
    for i in range(0, len(paths), bs):
        batch = paths[i:i+bs]
        res = model.predict(source=[str(p) for p in batch], imgsz=int(args.imgsz), device=str(args.device), batch=len(batch), verbose=False, save=False)
        if len(res) != len(batch): raise RuntimeError(f'Prediction length mismatch: got {len(res)} expected {len(batch)}')
        for pth, r in zip(batch, res):
            probs = r.probs.data.detach().cpu().flatten().tolist(); pred = int(np.argmax(probs)); idx=len(y_pred)
            y_pred.append(pred)
            rows.append({'path':str(pth), 'true_idx':int(y_true[idx]), 'true_class':class_dirs[int(y_true[idx])],
                         'pred_idx':pred, 'pred_class':class_dirs[pred] if pred < len(class_dirs) else str(pred),
                         'correct':pred==int(y_true[idx]), **{f'prob_{j}':float(probs[j]) if j < len(probs) else None for j in range(4)}})
    elapsed = time.time()-t; labels=list(range(len(class_dirs)))
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/f'predictions_{split}.csv', index=False)
    pd.DataFrame(confusion_matrix(y_true,y_pred,labels=labels), index=class_dirs, columns=class_dirs).to_csv(outdir/f'confusion_matrix_{split}.csv')
    write_json(outdir/f'classification_report_{split}.json', classification_report(y_true,y_pred,labels=labels,target_names=class_dirs,output_dict=True,zero_division=0))
    m = {'split_name':split, 'checkpoint':str(ckpt), 'eval_root':str(root), 'n':len(y_true),
         'test_accuracy':float(accuracy_score(y_true,y_pred)),
         'test_macro_f1':float(f1_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)),
         'macro_precision':float(precision_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)),
         'macro_recall':float(recall_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)),
         'cohen_kappa':float(cohen_kappa_score(y_true,y_pred,labels=labels)),
         'fps':float(len(y_true)/elapsed) if elapsed>0 else None,
         'true_counts':{class_dirs[i]:int(np.sum(np.array(y_true)==i)) for i in labels},
         'pred_counts':{class_dirs[i]:int(np.sum(np.array(y_pred)==i)) for i in labels}}
    write_json(outdir/f'metrics_{split}.json', m)
    return m

def zip_results(out: Path, zip_path: Path):
    if zip_path.exists(): zip_path.unlink()
    skip = {'weights','__pycache__','yolo_classification_dataset_seed42','yolo_group_split_hardlink_dataset','yolo_group_split_symlink_dataset','top9_corrupted_test_sets'}
    ok = {'.csv','.json','.txt','.log','.md','.py','.sh','.xlsx'}
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if not p.is_file(): continue
            if any(part in skip for part in p.parts): continue
            if p.suffix.lower() in {'.pt','.pth','.onnx','.engine','.cache'}: continue
            if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'} and 'example_corruptions' not in p.parts: continue
            if p.suffix.lower() not in ok and 'example_corruptions' not in p.parts: continue
            if p.stat().st_size > 50_000_000: continue
            z.write(p, p.relative_to(out.parent))
    log(f'ZIP={zip_path} size_MB={zip_path.stat().st_size/1024/1024:.3f}')

def main():
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    output_root = Path(
        os.environ.get("OUTPUT_DIR", project_root / "runs")
    ).expanduser()
    legacy_projects = project_root / "legacy_projects"
    default_asl_project = os.environ.get(
        "ASL_PROJECT_DIR",
        str(legacy_projects / "YOLO26M_CE_VS_ASL_LDAM_NOISY_TEST_EVAL_project"),
    )
    default_ce_project = os.environ.get(
        "CE_PROJECT_DIR",
        str(legacy_projects / "YOLO26M_CE_TOP10_CUSTOM_LOSS_CBAM_WORK_project"),
    )
    default_data_dir = os.environ.get(
        "DATA_DIR",
        str(project_root / "datasets" / "processed-images"),
    )
    ap = argparse.ArgumentParser()
    ap.add_argument('--project_dir', default=default_asl_project)
    ap.add_argument('--asl_project_dir', default=default_asl_project)
    ap.add_argument('--ce_project_dir', default=default_ce_project)
    ap.add_argument('--dataset_root', default=default_data_dir)
    ap.add_argument(
        '--output_dir',
        default=str(output_root / 'legacy_yolo_v5_outputs'),
    )
    ap.add_argument(
        '--patch_script',
        default=str(project_root / 'patch_yolo_attention_injection_v2.py'),
    )
    ap.add_argument('--model_name', default='yolo26m-cls')
    ap.add_argument('--yolo_weights', default='', help='Local .pt path or Ultralytics model name. If empty, uses <model_name>.pt')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--patience', type=int, default=15); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--imgsz', type=int, default=224)
    ap.add_argument('--device', default='0'); ap.add_argument('--eval_batch', type=int, default=32); ap.add_argument('--lr0', type=float, default=0.00125)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--asl_ldam_loss_key', default='asl_ldam_margin'); ap.add_argument('--ce_effective_loss_key', default='ce_effective_num_weighted_cbam')
    ap.add_argument('--severities', default='1,2,3')
    ap.add_argument('--corruptions', default='impulse_noise,gaussian_noise,contrast_reduction,defocus_blur,low_light')
    ap.add_argument('--max_examples_per_corruption', type=int, default=4)
    ap.add_argument('--stage1_split_note', default='ShrimpXNet-style fixed random stratified image-level 70/15/15 split, seed 42; not group-safe')
    ap.add_argument('--skip_prepare', action='store_true'); ap.add_argument('--skip_patch', action='store_true'); ap.add_argument('--skip_train', action='store_true')
    ap.add_argument('--force', action='store_true'); ap.add_argument('--force_corruptions_rebuild', action='store_true')
    ap.add_argument(
        '--zip_path',
        default=str(output_root / 'legacy_yolo_v5_results_only.zip'),
    )
    args = ap.parse_args()
    if not args.yolo_weights:
        args.yolo_weights = f'{args.model_name}.pt'
    random.seed(args.seed); np.random.seed(args.seed)
    project = Path(args.project_dir).expanduser().resolve()
    asl_project = Path(args.asl_project_dir).expanduser().resolve()
    ce_project = Path(args.ce_project_dir).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve(); log_dir = out/'command_logs'; out.mkdir(parents=True, exist_ok=True)
    ds = resolve_dataset_root(args.dataset_root)
    for label, p in [('project_dir', project), ('asl_project_dir', asl_project), ('ce_project_dir', ce_project)]:
        if not (p/'shrimp_scripts'/'models_yolo.py').is_file():
            raise FileNotFoundError(f'{label} missing shrimp_scripts/models_yolo.py: {p}')
    env = {'PYTHONPATH':str(asl_project)+os.pathsep+str(ce_project)+os.pathsep+os.environ.get('PYTHONPATH',''), 'SHRIMP_DATASET_ROOT':str(ds), 'PYTHONUNBUFFERED':'1',
           'CUDA_VISIBLE_DEVICES':str(args.device), 'TORCH_CUDNN_V8_API_ENABLED':'1', 'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True,max_split_size_mb:128'}
    import_project(asl_project)
    write_json(out/'source_dataset_audit.json', {'dataset_root':str(ds), 'class_dirs':detect_source_classes(ds), 'counts':{d:count_images(ds/d) for d in detect_source_classes(ds)}})
    write_json(out/'fixed_stage1_ce_reference.json', STAGE1_CE_REFERENCE)
    run([sys.executable, '-m', 'compileall', 'shrimp_scripts', 'experiments'], asl_project, log_dir/'000_compileall_asl_project_before.log', env)
    run([sys.executable, '-m', 'compileall', 'shrimp_scripts', 'experiments'], ce_project, log_dir/'001_compileall_ce_project_before.log', env)
    if not args.skip_prepare:
        run([sys.executable, 'shrimp_scripts/run_01_prepare_dataset.py', '--output_dir', str(out), '--resume', '--progress'], asl_project, log_dir/'010_prepare_dataset.log', env)
    if not args.skip_patch:
        patch = Path(args.patch_script).expanduser().resolve()
        if not patch.is_file(): raise FileNotFoundError(f'Missing patch script: {patch}')
        run([sys.executable, str(patch), '--project_dir', str(asl_project), '--run_compileall'], project_root, log_dir/'020_patch_attention_cbam_asl_project.log', env)
        run([sys.executable, str(patch), '--project_dir', str(ce_project), '--run_compileall'], project_root, log_dir/'021_patch_attention_cbam_ce_project.log', env)
    yolo_root = find_yolo_root(out); class_dirs = find_test_classes(yolo_root); clean_root = yolo_root
    write_json(out/'prepared_yolo_dataset_audit.json', {'yolo_root':str(yolo_root), 'class_dirs':class_dirs, 'test_counts':{c:count_images(yolo_root/'test'/c) for c in class_dirs}})
    asl = resolve_loss_key(asl_project, args.asl_ldam_loss_key, ['asl_ldam_margin','asl_ldam','asl_ldam_loss'])
    ce = resolve_loss_key(ce_project, args.ce_effective_loss_key, ['ce_effective_num_weighted_cbam','ce_effective_num_weighted','effective_num_weighted_ce','class_balanced_effective_ce'])
    custom_variants = [
        {'variant_key':'asl_ldam_margin','loss_key':asl,'attention_key':'none_baseline','description':'ASL-LDAM, no CBAM','project_dir':str(asl_project)},
        {'variant_key':'asl_ldam_margin_cbam','loss_key':asl,'attention_key':'cbam','description':'ASL-LDAM + CBAM','project_dir':str(asl_project)},
        # Top-1 valid row from the 152-run AUTO_RESUME ranking:
        # ASL-LDAM + SimAM gated residual + DCFR texture, test Macro-F1≈0.9054 on the previous clean test.
        {'variant_key':'asl_ldam_simam_dcfr_top1','loss_key':asl,'attention_key':'simam_gated_residual__dcfr_texture','description':'Top-1 ASL-LDAM + SimAM gated residual + DCFR texture from 152-run ranking','project_dir':str(asl_project)},
        {'variant_key':'ce_effective_num_weighted','loss_key':ce,'attention_key':'none_baseline','description':'CE effective-number weighted custom loss, no CBAM','project_dir':str(ce_project)},
        {'variant_key':'ce_effective_num_weighted_cbam','loss_key':ce,'attention_key':'cbam','description':'CE effective-number weighted custom loss + CBAM','project_dir':str(ce_project)},
    ]
    write_json(out/'focused_run_plan.json', {
        'baseline':'native CE rerun + fixed Stage1 reference metrics',
        'custom_variants':custom_variants,
        'corruptions':args.corruptions.split(','),
        'severities':args.severities.split(','),
        'resolved_ce_loss_key':ce,
        'resolved_asl_loss_key':asl,
        'model_name':args.model_name,
        'stage1_split_note':args.stage1_split_note,
        'asl_project_dir':str(asl_project),
        'ce_project_dir':str(ce_project),
    })
    summaries=[]
    if not args.skip_train:
        try: summaries.append(train_native_ce_baseline(out, yolo_root, args))
        except Exception as e:
            row={'variant_key':'baseline_ce_native_ultralytics','status':'failed_train','error':repr(e)}; summaries.append(row); write_json(out/'variant_summaries'/'baseline_ce_native_ultralytics.json', row); log(f'FAILED TRAIN baseline CE: {e!r}')
        for v in custom_variants:
            try: summaries.append(train_custom_variant(project,out,v,args))
            except Exception as e:
                row={**v,'status':'failed_train','error':repr(e)}; summaries.append(row); write_json(out/'variant_summaries'/f"{v['variant_key']}.json", row); log(f"FAILED TRAIN {v['variant_key']}: {e!r}")
    else:
        summaries=[read_json(out/'variant_summaries'/'baseline_ce_native_ultralytics.json', {'variant_key':'baseline_ce_native_ultralytics','status':'missing_summary'})]
        summaries += [read_json(out/'variant_summaries'/f"{v['variant_key']}.json", {**v,'status':'missing_summary'}) for v in custom_variants]
    eval_roots = [('clean', clean_root, 'clean', 0, {})]
    corruptions = [x.strip() for x in args.corruptions.split(',') if x.strip()]
    severities = [int(x.strip()) for x in args.severities.split(',') if x.strip()]
    for corruption in corruptions:
        if corruption not in CORRUPTIONS: raise ValueError(f'Unknown corruption {corruption}')
        for severity in severities:
            root = make_corrupted_dataset(yolo_root/'test', out, class_dirs, corruption, severity, args.seed + 1000*severity + 17*CORRUPTIONS.index(corruption), args.force_corruptions_rebuild, args.max_examples_per_corruption)
            eval_roots.append((f'{corruption}_s{severity}', root, corruption, severity, SEVERITY_PARAMS[corruption][severity]))
    rows=[dict(STAGE1_CE_REFERENCE)]
    for s in summaries:
        ck=s.get('checkpoint')
        if not ck or not Path(str(ck)).is_file(): rows.append({**s,'status':s.get('status','missing_checkpoint')}); continue
        clean_f1 = None; clean_acc = None; clean_kappa = None
        for split, root, corruption, severity, params in eval_roots:
            try:
                m = evaluate_checkpoint(project, Path(str(ck)), root, class_dirs, split, args, out/'evaluations'/s['variant_key'])
                if split == 'clean':
                    clean_f1 = m['test_macro_f1']; clean_acc = m['test_accuracy']; clean_kappa = m['cohen_kappa']
                rows.append({**s, 'status':'evaluated', 'corruption':corruption, 'severity':severity, 'severity_params':json.dumps(params), **m,
                             'drop_macro_f1_vs_own_clean': None if clean_f1 is None else float(m['test_macro_f1']-clean_f1),
                             'drop_accuracy_vs_own_clean': None if clean_acc is None else float(m['test_accuracy']-clean_acc),
                             'drop_kappa_vs_own_clean': None if clean_kappa is None else float(m['cohen_kappa']-clean_kappa)})
            except Exception as e:
                rows.append({**s, 'split_name':split, 'corruption':corruption, 'severity':severity, 'severity_params':json.dumps(params), 'status':'failed_eval', 'error':repr(e)})
                log(f"FAILED EVAL {s.get('variant_key')} {split}: {e!r}")
    df = pd.DataFrame(rows)
    for c in ['test_macro_f1','test_accuracy','cohen_kappa']:
        if c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
    # Compare to baseline CE under the same split/corruption/severity.
    base = df[df.get('variant_key',pd.Series(dtype=str)).eq('baseline_ce_native_ultralytics')][['split_name','test_macro_f1','test_accuracy','cohen_kappa']].rename(columns={'test_macro_f1':'baseline_ce_same_split_macro_f1','test_accuracy':'baseline_ce_same_split_accuracy','cohen_kappa':'baseline_ce_same_split_kappa'})
    if not base.empty:
        df = df.merge(base, on='split_name', how='left')
        df['delta_macro_f1_vs_baseline_ce_same_split'] = df['test_macro_f1'] - df['baseline_ce_same_split_macro_f1']
        df['delta_accuracy_vs_baseline_ce_same_split'] = df['test_accuracy'] - df['baseline_ce_same_split_accuracy']
        df['delta_kappa_vs_baseline_ce_same_split'] = df['cohen_kappa'] - df['baseline_ce_same_split_kappa']
    df['delta_macro_f1_vs_fixed_stage1_clean_ce'] = df.get('test_macro_f1', pd.Series(dtype=float)) - STAGE1_CE_REFERENCE['test_macro_f1']
    reports = out/'final_reports'; reports.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports/'final_clean_top9_noise_summary_compared_to_baseline_ce.csv', index=False)
    keep = [c for c in ['variant_key','loss_key','attention_key','split_name','corruption','severity','test_macro_f1','test_accuracy','cohen_kappa','drop_macro_f1_vs_own_clean','delta_macro_f1_vs_baseline_ce_same_split','delta_accuracy_vs_baseline_ce_same_split','delta_kappa_vs_baseline_ce_same_split','status'] if c in df.columns]
    df[keep].to_csv(reports/'compact_summary_table.csv', index=False)
    corr = df[(df.get('status')=='evaluated') & (df.get('corruption')!='clean')].copy()
    if not corr.empty:
        robust = corr.groupby(['variant_key','loss_key','attention_key'], dropna=False).agg(mean_corrupted_macro_f1=('test_macro_f1','mean'), worst_corrupted_macro_f1=('test_macro_f1','min'), mean_delta_f1_vs_ce_same_split=('delta_macro_f1_vs_baseline_ce_same_split','mean'), mean_accuracy=('test_accuracy','mean'), mean_kappa=('cohen_kappa','mean')).reset_index().sort_values('mean_corrupted_macro_f1', ascending=False)
        robust.to_csv(reports/'robustness_ranked_summary.csv', index=False)
    try:
        with pd.ExcelWriter(reports/'final_clean_top9_noise_summary_compared_to_baseline_ce.xlsx', engine='openpyxl') as w:
            pd.DataFrame([STAGE1_CE_REFERENCE]).to_excel(w,'Fixed_Stage1_CE_Ref', index=False)
            df.to_excel(w,'All_Evals', index=False)
            df[keep].to_excel(w,'Compact', index=False)
            if not corr.empty: robust.to_excel(w,'Robustness_Ranking', index=False)
    except Exception as e: log(f'WARN xlsx failed: {e!r}')
    log('\nFINAL COMPACT SUMMARY\n'+'='*120); log(df[keep].to_string(index=False))
    zip_results(out, Path(args.zip_path).expanduser().resolve())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
