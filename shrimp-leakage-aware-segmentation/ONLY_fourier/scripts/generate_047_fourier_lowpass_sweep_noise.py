import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '06_lowfreq_flatten_s100_b0p30_no_aug_hook_off.ipynb'
OUTPUT = ROOT / '047_fourier_lowpass_sweep_noise_kaggle_seed42.ipynb'


def source(nb, index):
    return ''.join(nb['cells'][index].get('source', []))


def set_source(nb, index, text):
    nb['cells'][index]['source'] = text.splitlines(keepends=True)


def replace_once(text, old, new):
    if old not in text:
        raise RuntimeError(f'Expected template text was not found: {old[:120]!r}')
    return text.replace(old, new, 1)


nb = json.loads(BASE.read_text(encoding='utf-8'))

set_source(nb, 0, """# ONLY Fourier - Pure Low-Pass Sweep and Noise Robustness\n\nThis Kaggle notebook uses the old `shrimpDisHandSegV2` dataset and the established strict no-augmentation, hook-off protocol.\n\nTraining rows:\n- no-Fourier control\n- pure Gaussian low-pass sigma 25\n- pure Gaussian low-pass sigma 50\n- pure Gaussian low-pass sigma 100\n\nNoise evaluation order:\n1. corrupt the original test image;\n2. apply the matching low-pass Fourier transform for Fourier rows;\n3. run YOLO inference.\n\nThe no-Fourier control receives the noisy image directly.\n""")
set_source(nb, 1, """## Outputs\n\nThe notebook writes training metrics, paper rows, split manifests, checkpoints, and a noise-evaluation CSV under `/kaggle/working/shrimp_only_fourier_lowpass_sweep_noise_kaggle_seed42/`.\n\nThe final checklist is intentionally self-contained and does not reference optional variables from earlier display cells.\n""")

controls = source(nb, 3)
controls = controls.replace("EXECUTION_PLAN = 'only_fourier_06_lowfreq_flatten_s100_b0p30_no_aug_hook_off'", "EXECUTION_PLAN = 'only_fourier_047_lowpass_sweep_noise_kaggle_seed42'")
controls = controls.replace("EXPERIMENT_ROOT = WORK_DIR / 'shrimp_only_fourier_lowfreq_flatten_s100_b0p30_no_aug_hook_off'", "EXPERIMENT_ROOT = WORK_DIR / 'shrimp_only_fourier_lowpass_sweep_noise_kaggle_seed42'")
set_source(nb, 3, controls)

core = source(nb, 17)
core = core.replace("FOURIER_METHOD = 'lowfreq_flatten'", "FOURIER_METHOD = 'lowpass'")
core = core.replace("FOURIER_POLICY = 'lowfreq_flatten_s100_b0p30_all_splits'", "FOURIER_POLICY = 'lowpass_sweep_all_splits'")
core = core.replace("FOURIER_PARAMS = {\"beta\": 0.3, \"sigma\": 100}", "FOURIER_PARAMS = {'sigma': 50}")
core = core.replace("'preprocessing_policy': 'lowfreq_flatten',", "'preprocessing_policy': exp.get('preprocessing_policy', 'fourier_lowpass' if exp.get('fourier_enabled') else 'none'),")
core = core.replace("'fourier_policy': 'lowfreq_flatten_s100_b0p30_all_splits',", "'fourier_policy': exp.get('fourier_policy', 'none'),")
core = core.replace("'fourier_sigma': FOURIER_SIGMA,", "'fourier_sigma': exp.get('sigma') if exp.get('fourier_enabled') else None,")
core = core.replace("'fourier_alpha': FOURIER_ALPHA,", "'fourier_alpha': None,")
core = core.replace("'fourier_params': json.dumps(FOURIER_PARAMS, sort_keys=True),", "'fourier_params': json.dumps({'sigma': exp.get('sigma')} if exp.get('fourier_enabled') else {}, sort_keys=True),")
core = core.replace("fourier_transform_summary = apply_fourier_transform_to_dataset(dataset_dir)", "fourier_transform_summary = apply_fourier_transform_to_dataset(dataset_dir) if exp.get('fourier_enabled') else {'strategy': 'fourier_disabled'}")
core = core.replace("def apply_fourier_tensor_transform(tensor):\n    height, width = tensor.shape[-2:]\n    method = FOURIER_METHOD\n    params = FOURIER_PARAMS\n", "def apply_fourier_tensor_transform(tensor):\n    height, width = tensor.shape[-2:]\n    method = FOURIER_METHOD\n    params = FOURIER_PARAMS\n\n    if method == 'lowpass':\n        low = fft_filter_image(tensor, fourier_lowpass_mask(height, width, float(params['sigma']), tensor.device))\n        return torch.clamp(low, 0, 255)\n")
core = core.replace("    dataset_dir = copy_dataset_for_experiment(exp['key'])\n    fourier_transform_summary", "    global FOURIER_METHOD, FOURIER_PARAMS, FOURIER_POLICY, FOURIER_SIGMA, FOURIER_ALPHA\n    if exp.get('fourier_enabled'):\n        FOURIER_METHOD = 'lowpass'\n        FOURIER_PARAMS = {'sigma': float(exp['sigma'])}\n        FOURIER_POLICY = exp['fourier_policy']\n        FOURIER_SIGMA = float(exp['sigma'])\n        FOURIER_ALPHA = 0.0\n    dataset_dir = copy_dataset_for_experiment(exp['key'])\n    fourier_transform_summary")

experiment_block = """EXPERIMENTS = [\n    {\n        'key': 'no_fourier_control',\n        'name': 'No-Fourier control | no augmentation | hook off',\n        'fourier_enabled': False,\n        'preprocessing_policy': 'none',\n        'fourier_policy': 'none',\n    },\n    {\n        'key': 'lowpass_sigma25',\n        'name': 'Pure Fourier low-pass | sigma 25 | no augmentation | hook off',\n        'fourier_enabled': True,\n        'sigma': 25,\n        'preprocessing_policy': 'fourier_lowpass',\n        'fourier_policy': 'lowpass_gaussian_sigma25_all_splits',\n    },\n    {\n        'key': 'lowpass_sigma50',\n        'name': 'Pure Fourier low-pass | sigma 50 | no augmentation | hook off',\n        'fourier_enabled': True,\n        'sigma': 50,\n        'preprocessing_policy': 'fourier_lowpass',\n        'fourier_policy': 'lowpass_gaussian_sigma50_all_splits',\n    },\n    {\n        'key': 'lowpass_sigma100',\n        'name': 'Pure Fourier low-pass | sigma 100 | no augmentation | hook off',\n        'fourier_enabled': True,\n        'sigma': 100,\n        'preprocessing_policy': 'fourier_lowpass',\n        'fourier_policy': 'lowpass_gaussian_sigma100_all_splits',\n    },\n]\n"""
core = re.sub(r"EXPERIMENTS = \[.*?\n\]\n\nNO_AUG_TRAIN_ARGS", experiment_block + "\nNO_AUG_TRAIN_ARGS", core, count=1, flags=re.S)
set_source(nb, 17, core)

loop = source(nb, 19)
loop = re.sub(r"EXPERIMENTS = \[.*?\n\]\n\npaper_rows", "EXPERIMENTS = [\n    {'key': 'no_fourier_control', 'name': 'No-Fourier control | no augmentation | hook off', 'fourier_enabled': False, 'preprocessing_policy': 'none', 'fourier_policy': 'none'},\n    {'key': 'lowpass_sigma25', 'name': 'Pure Fourier low-pass | sigma 25 | no augmentation | hook off', 'fourier_enabled': True, 'sigma': 25, 'preprocessing_policy': 'fourier_lowpass', 'fourier_policy': 'lowpass_gaussian_sigma25_all_splits'},\n    {'key': 'lowpass_sigma50', 'name': 'Pure Fourier low-pass | sigma 50 | no augmentation | hook off', 'fourier_enabled': True, 'sigma': 50, 'preprocessing_policy': 'fourier_lowpass', 'fourier_policy': 'lowpass_gaussian_sigma50_all_splits'},\n    {'key': 'lowpass_sigma100', 'name': 'Pure Fourier low-pass | sigma 100 | no augmentation | hook off', 'fourier_enabled': True, 'sigma': 100, 'preprocessing_policy': 'fourier_lowpass', 'fourier_policy': 'lowpass_gaussian_sigma100_all_splits'},\n]\n\npaper_rows", loop, count=1, flags=re.S)
set_source(nb, 19, loop)

for cell in nb['cells']:
    if 'source' not in cell:
        continue
    text = ''.join(cell['source'])
    replacements = {
        'only_fourier_lowfreq_flatten_s100_b0p30_no_aug_hook_off_summary.csv': 'only_fourier_lowpass_sweep_noise_summary.csv',
        'only_fourier_lowfreq_flatten_s100_b0p30_no_aug_hook_off_paper_row.csv': 'only_fourier_lowpass_sweep_noise_paper_row.csv',
        'only_fourier_lowfreq_flatten_s100_b0p30_no_aug_hook_off_partial.csv': 'only_fourier_lowpass_sweep_noise_partial.csv',
        'train_args_fourier_highpass_s50_a0p30_no_aug_hook_off.json': 'train_args_lowpass_sweep_noise.json',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    cell['source'] = text.splitlines(keepends=True)

checklist = """from pathlib import Path

print('Download checklist:')
for path in [
    REPORT_DIR / 'only_fourier_lowpass_sweep_noise_summary.csv',
    REPORT_DIR / 'only_fourier_lowpass_sweep_noise_paper_row.csv',
    REPORT_DIR / 'only_fourier_lowpass_sweep_noise_partial.csv',
    REPORT_DIR / 'train_args_lowpass_sweep_noise.json',
    REPORT_DIR / 'split_manifests',
]:
    print(f'{path} | exists={path.exists()}')

if 'summary_df' in globals() and not summary_df.empty:
    display(summary_df[['experiment', 'name', 'best_pt', 'run_path']])
else:
    print('Training summary is not available.')
"""
set_source(nb, 23, checklist)

noise_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## Noise Evaluation: Noise Before Fourier\n',
        '\n',
        'This section corrupts each held-out test image first. For low-pass models, the matching Gaussian low-pass transform is then applied to the noisy image before inference. The no-Fourier control receives the noisy image directly.\n',
    ],
}

noise_code = r'''# Noise evaluation: noise first, then matching low-pass Fourier, then inference.
import gc
import shutil
import time

NOISE_ROOT = EXPERIMENT_ROOT / 'lowpass_noise_test_sets'
NOISE_REPORT_DIR = EXPERIMENT_ROOT / 'lowpass_noise_reports'
if NOISE_ROOT.exists():
    shutil.rmtree(NOISE_ROOT)
if NOISE_REPORT_DIR.exists():
    shutil.rmtree(NOISE_REPORT_DIR)
NOISE_ROOT.mkdir(parents=True, exist_ok=True)
NOISE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

NOISE_CONDITIONS = [
    {'name': 'clean_reference', 'type': 'clean', 'severity': 0},
    {'name': 'gaussian_sigma20', 'type': 'gaussian', 'severity': 20},
    {'name': 'salt_pepper_p003', 'type': 'salt_pepper', 'severity': 0.03},
    {'name': 'color_cast_blue40', 'type': 'color_cast', 'severity': 40},
    {'name': 'low_contrast_f050', 'type': 'low_contrast', 'severity': 0.50},
    {'name': 'motion_blur_k11', 'type': 'motion_blur', 'severity': 11},
]

def corrupt_noise_image(image, condition, rng):
    kind, severity = condition['type'], condition['severity']
    if kind == 'clean':
        return image.copy()
    if kind == 'gaussian':
        noise = rng.normal(0.0, float(severity), image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == 'salt_pepper':
        output = image.copy()
        mask = rng.random(image.shape[:2])
        output[mask < float(severity) / 2.0] = 0
        output[mask > 1.0 - float(severity) / 2.0] = 255
        return output
    if kind == 'color_cast':
        output = image.astype(np.int32)
        output[:, :, 0] = np.clip(output[:, :, 0] + int(severity), 0, 255)
        return output.astype(np.uint8)
    if kind == 'low_contrast':
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip(mean + (image.astype(np.float32) - mean) * float(severity), 0, 255).astype(np.uint8)
    if kind == 'motion_blur':
        k = int(severity)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        angle = float(rng.uniform(0.0, 180.0))
        matrix = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (k, k))
        kernel /= kernel.sum() if kernel.sum() > 0 else 1.0
        return cv2.filter2D(image, -1, kernel)
    raise ValueError(kind)

def copy_split(source_root, destination_root, split):
    for subdir in ['images', 'labels']:
        shutil.copytree(Path(source_root) / split / subdir, Path(destination_root) / split / subdir, dirs_exist_ok=True)

def write_noise_yaml(root):
    content = dict(yaml.safe_load((Path(base_path) / 'data.yaml').read_text(encoding='utf-8')))
    content['path'] = str(root)
    content['train'] = str(Path(root) / 'train' / 'images')
    content['val'] = str(Path(root) / 'valid' / 'images')
    content['test'] = str(Path(root) / 'test' / 'images')
    path = Path(root) / 'data.yaml'
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding='utf-8')
    return path

def build_noise_condition(index, condition):
    destination = NOISE_ROOT / condition['name']
    for split in ['train', 'valid']:
        copy_split(base_path, destination, split)
    (destination / 'test' / 'images').mkdir(parents=True, exist_ok=True)
    (destination / 'test' / 'labels').mkdir(parents=True, exist_ok=True)
    for image_index, source_path in enumerate(sorted(Path(base_path, 'test', 'images').glob('*'))):
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        rng = np.random.default_rng(42 + index * 100000 + image_index)
        transformed = corrupt_noise_image(image, condition, rng)
        cv2.imwrite(str(destination / 'test' / 'images' / source_path.name), transformed)
        shutil.copy2(Path(base_path, 'test', 'labels', f'{source_path.stem}.txt'), destination / 'test' / 'labels' / f'{source_path.stem}.txt')
    return destination, write_noise_yaml(destination)

NOISE_DATASETS = {c['name']: build_noise_condition(i, c) for i, c in enumerate(NOISE_CONDITIONS)}

def prepare_lowpass_noise(condition_name, source_root, sigma):
    destination = NOISE_ROOT / f'lowpass_sigma{int(sigma)}' / condition_name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_root, destination)
    global FOURIER_METHOD, FOURIER_PARAMS, FOURIER_SIGMA, FOURIER_ALPHA
    FOURIER_METHOD = 'lowpass'
    FOURIER_PARAMS = {'sigma': float(sigma)}
    FOURIER_POLICY = f'lowpass_gaussian_sigma{int(sigma)}_test_after_noise'
    FOURIER_SIGMA = float(sigma)
    FOURIER_ALPHA = 0.0
    apply_fourier_transform_to_image_dir(destination / 'test' / 'images')
    return destination, write_noise_yaml(destination)

noise_rows = []
for model_row in summary_df.to_dict('records'):
    experiment = str(model_row['experiment'])
    checkpoint = Path(model_row['best_pt'])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = YOLO(str(checkpoint))
    sigma = float(experiment.split('lowpass_sigma')[-1]) if 'lowpass_sigma' in experiment else None
    for condition in NOISE_CONDITIONS:
        raw_root, raw_yaml = NOISE_DATASETS[condition['name']]
        if sigma is None:
            eval_root, eval_yaml = raw_root, raw_yaml
            fourier_at_inference = False
        else:
            eval_root, eval_yaml = prepare_lowpass_noise(condition['name'], raw_root, sigma)
            fourier_at_inference = True
        started = time.time()
        labeled_dir, labeled_yaml, _ = make_labeled_only_eval_dataset(eval_root, f'noise_{experiment}_{condition["name"]}')
        healthy_dir, healthy_yaml, _ = make_healthy_only_eval_dataset(eval_root, f'noise_{experiment}_{condition["name"]}')
        full = model.val(data=str(eval_yaml), split='test', imgsz=TRAIN_IMGSZ, plots=False, verbose=False)
        labeled = model.val(data=str(labeled_yaml), split='test', imgsz=TRAIN_IMGSZ, plots=False, verbose=False)
        count = count_prediction_errors(model, labeled_dir / 'test' / 'images', labeled_dir / 'test' / 'labels')
        healthy = healthy_false_positive_summary(model, healthy_dir / 'test' / 'images')
        labeled_map50 = metric_value(labeled, 'seg.map50')
        noise_rows.append({
            'experiment': experiment,
            'best_pt': str(checkpoint),
            'noise_condition': condition['name'],
            'noise_type': condition['type'],
            'noise_severity': condition['severity'],
            'fourier_at_inference': fourier_at_inference,
            'lowpass_sigma': sigma,
            'noise_order': 'noise_then_fourier_then_inference' if fourier_at_inference else 'noise_then_inference',
            'noisy_full_test_mask_map50': metric_value(full, 'seg.map50'),
            'noisy_full_test_mask_map50_95': metric_value(full, 'seg.map'),
            'noisy_labeled_test_mask_map50': labeled_map50,
            'noisy_labeled_test_mask_map50_95': metric_value(labeled, 'seg.map'),
            'noisy_healthy_fp_rate': healthy['healthy_mask_fp_rate'],
            'noisy_disease_miss_rate': count['disease_mask_miss_rate'],
            'noisy_healthy_aware_score': healthy_aware_score(labeled_map50, count, healthy),
            'noise_eval_time_sec': round(time.time() - started, 2),
        })
        pd.DataFrame(noise_rows).to_csv(NOISE_REPORT_DIR / 'lowpass_noise_evaluation_partial.csv', index=False)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

noise_df = pd.DataFrame(noise_rows)
noise_csv = NOISE_REPORT_DIR / 'lowpass_noise_evaluation_summary.csv'
noise_df.to_csv(noise_csv, index=False)
print('Saved low-pass noise evaluation:', noise_csv)
display(noise_df)
'''
final_check = """from pathlib import Path

print('Final download checklist:')
final_paths = [
    REPORT_DIR / 'only_fourier_lowpass_sweep_noise_summary.csv',
    REPORT_DIR / 'only_fourier_lowpass_sweep_noise_paper_row.csv',
    REPORT_DIR / 'only_fourier_lowpass_sweep_noise_partial.csv',
    REPORT_DIR / 'train_args_lowpass_sweep_noise.json',
    REPORT_DIR / 'split_manifests',
    NOISE_REPORT_DIR / 'lowpass_noise_evaluation_summary.csv',
    NOISE_REPORT_DIR / 'lowpass_noise_evaluation_partial.csv',
]
for path in final_paths:
    print(f'{path} | exists={path.exists()}')
"""
nb['cells'].extend([
    noise_md,
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': noise_code.splitlines(keepends=True)},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## Final Download Checklist\n']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': final_check.splitlines(keepends=True)},
])

OUTPUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'Wrote {OUTPUT}')
