"""Generate 4 Colab-ready Group-A improvement notebooks from the combined-modules source."""
import json, copy, re
from pathlib import Path

SRC = Path(r"D:\My_Projects\LLM_Scio\CVio_Shrimp_Disease_Classification_Capstone_SU26\yolov11n_attention\yolov11n_grouped_attention\aip491-01-yolo-seg-11n-combined-modules.ipynb")
OUT = Path(r"D:\My_Projects\LLM_Scio\CVio_Shrimp_Disease_Classification_Capstone_SU26\yolov11n_attention\yolov11n_simam_NhomA")

nb_orig = json.loads(SRC.read_text(encoding='utf-8'))

def colab(t):
    return t.replace('/kaggle/working/', '/content/')

def prep(src_nb):
    nb = copy.deepcopy(src_nb)
    for c in nb['cells']:
        c['source'] = [colab(''.join(c['source']))]
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    return nb

def get_src(nb, i):
    return ''.join(nb['cells'][i]['source'])

def put_src(nb, i, text):
    nb['cells'][i]['source'] = [text]

def sub(nb, i, old, new, use_regex=False):
    src = get_src(nb, i)
    if use_regex:
        result = re.sub(old, new, src, count=1, flags=re.DOTALL)
        put_src(nb, i, result)
    else:
        if old not in src:
            print(f"  WARN cell {i}: pattern not found -> {repr(old[:80])}")
            return nb
        put_src(nb, i, src.replace(old, new, 1))
    return nb

GPU_CELL = {
    "cell_type": "code", "metadata": {}, "execution_count": None,
    "outputs": [], "id": "gpu_check_nhom_a",
    "source": [
        "# Verify T4 GPU — Runtime > Change runtime type > GPU > T4\n",
        "import subprocess\n",
        "out = subprocess.run(['nvidia-smi'], capture_output=True, text=True)\n",
        "print(out.stdout if out.returncode == 0 else\n",
        "      'No GPU detected. Enable T4: Runtime > Change runtime type > T4 GPU')\n",
    ]
}

NEW_EXPS = (
    'EXPERIMENTS = [\n'
    '    {\n'
    '        "key": "baseline",\n'
    '        "name": "Baseline YOLO11n-seg",\n'
    '        "model_type": "baseline",\n'
    '    },\n'
    '    {\n'
    '        "key": "simam_ca",\n'
    '        "name": "YOLO11n-seg + SimAM + CA",\n'
    '        "model_type": "attention",\n'
    '        "yaml": MODEL_YAML_PATHS["simam_ca"],\n'
    '    },\n'
    ']'
)

def apply_common(nb, root, run_name, title, desc):
    # Cell 0: title + description
    put_src(nb, 0, f"# {title}\n\n{desc}\n")
    # Cell 19: EXPERIMENT_ROOT_STR, RUN_BASE_NAME, trim EXPERIMENTS to baseline+simam_ca
    sub(nb, 19,
        'EXPERIMENT_ROOT_STR = "/content/yolo11n_combined_modules_group_run"',
        f'EXPERIMENT_ROOT_STR = "/content/{root}"')
    sub(nb, 19,
        'RUN_BASE_NAME = "combined_modules_group_run"',
        f'RUN_BASE_NAME = "{run_name}"')
    # Regex replace full EXPERIMENTS list (DOTALL to match across lines)
    sub(nb, 19, r'EXPERIMENTS = \[.*?\]\n\nprint', NEW_EXPS + '\n\nprint', use_regex=True)
    # Insert GPU check cell right after title (index 1); all subsequent indices shift +1
    nb['cells'].insert(1, copy.deepcopy(GPU_CELL))
    return nb

def save(nb, fname):
    path = OUT / fname
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"  Saved: {fname}  ({path.stat().st_size // 1024} KB)")


# ─── A1: Stronger Augmentation ───────────────────────────────────────────────
print("Creating A1: augmentation...")
nb = prep(nb_orig)
apply_common(nb,
    root="yolo11n_simam_augmentation_run",
    run_name="simam_augmentation_run",
    title="YOLO11n SimAM+CA — Nhóm A1: Augmentation Mạnh Hơn",
    desc=(
        "So sánh Baseline và SimAM+CA khi dùng augmentation mạnh hơn phù hợp với ảnh tôm dưới nước:\n"
        "flipud=0.3, scale=0.5, hsv_s=0.5, hsv_v=0.4, degrees=10, translate=0.1, erasing=0.15.\n"
        "Kỳ vọng: +1-3% mAP50 nhờ đa dạng hóa quang học và góc chụp."
    )
)
# After GPU insert: original cell 21 → now at index 22
sub(nb, 22, '"erasing": 0.0,', '"erasing": 0.15,')
sub(nb, 22,
    '"flipud": 0.0,\n    "hsv_h": 0.01,\n    "hsv_s": 0.35,\n    "hsv_v": 0.20,\n'
    '    "degrees": 0.0,\n    "translate": 0.05,\n    "scale": 0.20,',
    '"flipud": 0.3,\n    "hsv_h": 0.05,\n    "hsv_s": 0.50,\n    "hsv_v": 0.40,\n'
    '    "degrees": 10.0,\n    "translate": 0.10,\n    "scale": 0.50,')
save(nb, "yolov11n_simam_augmentation.ipynb")

# ─── A2: Epochs 150 + Cosine LR ──────────────────────────────────────────────
print("Creating A2: epochs + cosine LR...")
nb = prep(nb_orig)
apply_common(nb,
    root="yolo11n_simam_epochs_cosine_run",
    run_name="simam_epochs_cosine_run",
    title="YOLO11n SimAM+CA — Nhóm A2: Epochs 150 + Cosine LR",
    desc=(
        "Tăng epochs 100→150, giảm patience 30→20, bật cos_lr=True và warmup_epochs=8.\n"
        "Cosine schedule giúp model hội tụ tốt hơn ở phase cuối; patience nhỏ hơn tránh lãng phí compute.\n"
        "Kỳ vọng: ổn định training, val loss thấp hơn, +1-2% mAP50."
    )
)
# Original cell 21 → index 22 after GPU insert
sub(nb, 22,
    '        epochs=100,\n        batch=16,\n        patience=30,',
    '        epochs=150,\n        batch=16,\n        patience=20,\n'
    '        cos_lr=True,\n        warmup_epochs=8,')
save(nb, "yolov11n_simam_epochs_cosine.ipynb")

# ─── A3: Test-Time Augmentation ──────────────────────────────────────────────
print("Creating A3: TTA...")
nb = prep(nb_orig)
apply_common(nb,
    root="yolo11n_simam_tta_run",
    run_name="simam_tta_run",
    title="YOLO11n SimAM+CA — Nhóm A3: Test-Time Augmentation (TTA)",
    desc=(
        "Bật augment=True trong tất cả val() calls để kích hoạt TTA (flip ngang + multi-scale ensemble).\n"
        "TTA không yêu cầu retrain — chỉ thay đổi inference. Kỳ vọng: +2-4% mAP50 ngay lập tức."
    )
)
# Original cell 21 → index 22 after GPU insert
sub(nb, 22,
    '    # YOLO val-like reports\n'
    '    full_val = best_model.val(data=str(yaml_path), split="val", imgsz=640, plots=True, verbose=False)\n'
    '    full_test = best_model.val(data=str(yaml_path), split="test", imgsz=640, plots=True, verbose=False)\n'
    '    labeled_val = best_model.val(data=str(labeled_eval_yaml), split="val", imgsz=640, plots=False, verbose=False)\n'
    '    labeled_test = best_model.val(data=str(labeled_eval_yaml), split="test", imgsz=640, plots=False, verbose=False)',
    '    # YOLO val-like reports (Test-Time Augmentation enabled: augment=True)\n'
    '    full_val = best_model.val(data=str(yaml_path), split="val", imgsz=640, augment=True, plots=True, verbose=False)\n'
    '    full_test = best_model.val(data=str(yaml_path), split="test", imgsz=640, augment=True, plots=True, verbose=False)\n'
    '    labeled_val = best_model.val(data=str(labeled_eval_yaml), split="val", imgsz=640, augment=True, plots=False, verbose=False)\n'
    '    labeled_test = best_model.val(data=str(labeled_eval_yaml), split="test", imgsz=640, augment=True, plots=False, verbose=False)')
save(nb, "yolov11n_simam_tta.ipynb")

# ─── A4: Prototype Masks nm=64 ───────────────────────────────────────────────
print("Creating A4: nm=64...")
nb = prep(nb_orig)
apply_common(nb,
    root="yolo11n_simam_nm64_run",
    run_name="simam_nm64_run",
    title="YOLO11n SimAM+CA — Nhóm A4: Prototype Masks nm=64",
    desc=(
        "Tăng số prototype masks Segment head từ nm=32 lên nm=64 trong YAML của SimAM+CA.\n"
        "Nhiều prototype basis vectors hơn → mask head có capacity lớn hơn → biên giới mask sắc nét hơn.\n"
        "Kỳ vọng: cải thiện mAP50-95 (mask quality) +1-2%, cost tính toán tăng nhẹ."
    )
)
# Original cell 17 → index 18 after GPU insert
sub(nb, 18,
    '  - [[24, 26, 28], 1, Segment, [nc, 32, 256]]"""),\n    "eca_simam"',
    '  - [[24, 26, 28], 1, Segment, [nc, 64, 256]]"""),\n    "eca_simam"')
save(nb, "yolov11n_simam_nm64.ipynb")

print("\nAll 4 Group-A notebooks created successfully!")
