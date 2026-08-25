import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ONLY_FOURIER = ROOT / "shrimp-leakage-aware-segmentation" / "ONLY_fourier"
SOURCE = ONLY_FOURIER / "012_factorial_A_H_highpass_s50_a0p10.ipynb"
OUTPUT = ONLY_FOURIER / "048_w1_train_aug_test_tta_kaggle_seed42.ipynb"


def source_text(cell):
    return "".join(cell.get("source", []))


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        cell["source"][-1] += "\n"


def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"][:22]

set_source(
    cells[0],
    """# 048 - Train W1, Offline Augmented-Test Evaluation, and TTA

This Kaggle notebook first trains the canonical W1 model because no checkpoint is assumed to exist. It then evaluates:

- **M0:** standard W1 reference on the original grouped test split;
- **M1:** six offline geometric test views, with geometry applied before W1;
- **M2:** single-view custom-evaluator control;
- **M3:** six-view W1 TTA with inverse mapping and mask fusion.

Standard AP uses Ultralytics validation confidence behavior (`conf=0.001`, supplied explicitly for auditability). Operational healthy-negative, disease-miss, and count diagnostics use `conf=0.25`.
""",
)

set_source(
    cells[1],
    """## Primary Outputs

- strict W1 training summary and paper row;
- seed-42 grouped split manifest, summary, and fingerprint;
- trained W1 `best.pt`;
- M0/M1 standard Ultralytics metrics, including per-view and pooled rows;
- M2/M3 custom-evaluator metrics and per-class rows;
- healthy-negative, disease-miss, count, and healthy-aware diagnostics;
- preprocessing/inference/fusion timing records;
- compact reports-and-checkpoint ZIP package.
""",
)

controls = source_text(cells[3])
controls = controls.replace(
    "EXECUTION_PLAN = 'only_fourier_12_factorial_A_H_highpass_s50_a0p10'",
    "EXECUTION_PLAN = '048_w1_train_aug_test_tta_kaggle_seed42'",
)
controls = controls.replace(
    "EXPERIMENT_ROOT = WORK_DIR / 'shrimp_only_fourier_factorial_A_H_highpass_s50_a0p10'",
    "EXPERIMENT_ROOT = WORK_DIR / 'shrimp_048_w1_train_aug_test_tta_seed42'",
)
controls = controls.replace(
    "PINNED_ULTRALYTICS_VERSION = '8.4.62'",
    """PINNED_ULTRALYTICS_VERSION = '8.4.62'

# Post-training evaluation controls.
RUN_M0_STANDARD_REFERENCE = True
RUN_M1_OFFLINE_AUG_TEST = not SMOKE_RUN
RUN_M2_SINGLE_VIEW_CONTROL = not SMOKE_RUN
RUN_M3_SIX_VIEW_TTA = not SMOKE_RUN
EXPECTED_SPLIT_FINGERPRINT = '1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb'
STANDARD_AP_CONF = 0.001
DIAGNOSTIC_CONF = 0.25
VAL_BATCH = 16
TTA_INFERENCE_BATCH = 1
""",
)
set_source(cells[3], controls)

download = source_text(cells[7])
download = download.replace("ROBOFLOW_API_KEY_DIRECT = ''", "ROBOFLOW_API_KEY_DIRECT = 'KOEk0qLzBFDc7zfyxtgs'")
set_source(cells[7], download)

helpers = source_text(cells[17])
single_mode = """ABLATION_MODES = [
    {
        'mode': 'W1',
        'enabled': True,
        'fourier_enabled': True,
        'clean_light_aug_enabled': False,
        'default_hook_enabled': False,
        'key': 'W1_fourier_highpass_s50_a0p10_no_aug_hook_off',
        'name': 'W1 | Fourier high-pass s50 a0.10 | all YOLO aug zero | hook off',
    },
]

EXPERIMENTS"""
helpers, replaced = re.subn(
    r"ABLATION_MODES = \[.*?\]\n\nEXPERIMENTS",
    single_mode,
    helpers,
    count=1,
    flags=re.S,
)
if replaced != 1:
    raise RuntimeError("Could not replace ABLATION_MODES in source notebook")
helpers = helpers.replace(
    "train_args_factorial_A_H_highpass_s50_a0p10.json",
    "train_args_048_w1_train_aug_test_tta_seed42.json",
)
helpers = helpers.replace(
    "Saved factorial A-H train args:",
    "Saved strict W1 train args:",
)
set_source(cells[17], helpers)

set_source(cells[18], "## Train Strict W1 and Evaluate the Canonical Split\n")

runner = source_text(cells[19])
runner = runner.replace("Enabled A-H modes:", "Enabled strict W1 training modes:")
runner = runner.replace(
    "only_fourier_factorial_A_H_highpass_s50_a0p10_partial.csv",
    "048_w1_train_partial.csv",
)
runner = runner.replace(
    "only_fourier_factorial_A_H_highpass_s50_a0p10_summary.csv",
    "048_w1_train_summary.csv",
)
runner = runner.replace("Saved baseline summary CSV:", "Saved W1 training summary CSV:")
set_source(cells[19], runner)

set_source(cells[20], "## Compact W1 Training Row\n")
paper = source_text(cells[21])
paper = paper.replace(
    "only_fourier_factorial_A_H_highpass_s50_a0p10_paper_row.csv",
    "048_w1_train_paper_row.csv",
)
paper = paper.replace("Saved compact baseline paper row:", "Saved compact W1 training paper row:")
paper = paper.replace(
    "A-H factorial high-pass s50 a0.10 grouped-specimen checkpoint by validation healthy-aware score:",
    "Strict W1 grouped-specimen checkpoint:",
)
set_source(cells[21], paper)


resolve_source = r'''# Resolve the newly trained W1 checkpoint and its Fourier-transformed dataset.
from pathlib import Path
import json
import os
import time

EVAL_REPORT_DIR = REPORT_DIR / 'w1_aug_test_tta'
EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

if summary_df.empty:
    raise RuntimeError('W1 training produced no summary row.')

w1_row = summary_df.sort_values(
    ['healthy_aware_labeled_val_mask_map50', 'labeled_val_mask_map50'],
    ascending=False,
).iloc[0]
W1_CHECKPOINT = Path(w1_row['best_pt'])
W1_EXPERIMENT_KEY = str(w1_row['experiment'])
W1_DATASET_DIR = EXPERIMENT_ROOT / W1_EXPERIMENT_KEY / 'dataset'
W1_DATA_YAML = W1_DATASET_DIR / 'data.yaml'

if not W1_CHECKPOINT.exists():
    raise FileNotFoundError(f'Missing newly trained W1 checkpoint: {W1_CHECKPOINT}')
if not W1_DATASET_DIR.exists():
    raise FileNotFoundError(f'Missing Fourier-transformed W1 dataset: {W1_DATASET_DIR}')

actual_fingerprint = str(w1_row.get('split_fingerprint', ''))
if not SMOKE_RUN and actual_fingerprint != EXPECTED_SPLIT_FINGERPRINT:
    raise RuntimeError(
        'Split fingerprint mismatch. Refusing post-training comparison. '
        f'Expected {EXPECTED_SPLIT_FINGERPRINT}, got {actual_fingerprint}'
    )

TEST_VIEWS = [
    'original',
    'horizontal_flip',
    'vertical_flip',
    'rotate_90',
    'rotate_180',
    'rotate_270',
]

EVALUATION_CONTRACT = {
    'checkpoint': str(W1_CHECKPOINT),
    'dataset': 'shrimpdishandsegv2-v1',
    'seed': 42,
    'split_policy': 'stratified_grouped_specimen',
    'split_fingerprint': actual_fingerprint,
    'ultralytics_version': ultralytics.__version__,
    'imgsz': TRAIN_IMGSZ,
    'w1_sigma': FOURIER_SIGMA,
    'w1_alpha': FOURIER_ALPHA,
    'standard_ap_conf': STANDARD_AP_CONF,
    'diagnostic_conf': DIAGNOSTIC_CONF,
    'nms_iou': 0.70,
    'max_det': 300,
    'test_views': TEST_VIEWS,
    'processing_order': 'geometric_view_then_w1_then_yolo',
}
(EVAL_REPORT_DIR / '048_evaluation_contract.json').write_text(
    json.dumps(EVALUATION_CONTRACT, indent=2), encoding='utf-8'
)

print('W1 checkpoint:', W1_CHECKPOINT)
print('W1 transformed dataset:', W1_DATASET_DIR)
print('Verified split fingerprint:', actual_fingerprint)
print('Evaluation modes:', {
    'M0': RUN_M0_STANDARD_REFERENCE,
    'M1': RUN_M1_OFFLINE_AUG_TEST,
    'M2': RUN_M2_SINGLE_VIEW_CONTROL,
    'M3': RUN_M3_SIX_VIEW_TTA,
})
'''


standard_helpers_source = r'''# Standard Ultralytics evaluation and offline augmented-test helpers.
from collections import defaultdict
import gc
import os
import shutil


def list_images_flat(image_dir):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(Path(image_dir).glob(f'*{ext}'))
    return sorted(path for path in paths if path.is_file())


def label_for_image(image_path):
    return Path(image_path).parent.parent / 'labels' / f'{Path(image_path).stem}.txt'


def label_has_instances(label_path):
    label_path = Path(label_path)
    return label_path.exists() and any(line.strip() for line in label_path.read_text().splitlines())


def write_image_list_yaml(image_paths, key):
    list_dir = EVAL_REPORT_DIR / 'image_lists'
    list_dir.mkdir(parents=True, exist_ok=True)
    list_path = list_dir / f'{key}.txt'
    list_path.write_text('\n'.join(str(Path(path).resolve()) for path in image_paths) + '\n', encoding='utf-8')

    cfg = yaml.safe_load(W1_DATA_YAML.read_text(encoding='utf-8'))
    cfg['train'] = str(W1_DATASET_DIR / 'train' / 'images')
    cfg['val'] = str(W1_DATASET_DIR / 'valid' / 'images')
    cfg['test'] = str(list_path.resolve())
    yaml_path = list_dir / f'{key}.yaml'
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return yaml_path


def metric_or_nan(metrics, family, attribute):
    obj = getattr(metrics, family, None)
    if obj is None or not hasattr(obj, attribute):
        return float('nan')
    try:
        return float(getattr(obj, attribute))
    except Exception:
        return float('nan')


def per_class_rows(metrics, mode, view, scope, class_names):
    rows = []
    box_maps = list(getattr(metrics.box, 'maps', [])) if getattr(metrics, 'box', None) is not None else []
    mask_maps = list(getattr(metrics.seg, 'maps', [])) if getattr(metrics, 'seg', None) is not None else []
    for class_id, class_name in enumerate(class_names):
        rows.append({
            'mode': mode,
            'view': view,
            'scope': scope,
            'class_id': class_id,
            'class_name': class_name,
            'box_map50_95': float(box_maps[class_id]) if class_id < len(box_maps) else float('nan'),
            'mask_map50_95': float(mask_maps[class_id]) if class_id < len(mask_maps) else float('nan'),
        })
    return rows


def fixed_threshold_diagnostics(model, image_paths, conf=DIAGNOSTIC_CONF):
    image_paths = [Path(path) for path in image_paths]
    if not image_paths:
        return {}
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=TRAIN_IMGSZ,
        conf=conf,
        iou=0.70,
        max_det=300,
        batch=VAL_BATCH,
        stream=True,
        verbose=False,
    )
    healthy_images = healthy_with_mask_fp = healthy_masks = 0
    disease_images = disease_mask_misses = 0
    labeled_mask_errors = []
    labeled_mask_exact = []
    pred_masks_total = gt_total = 0

    for image_path, result in zip(image_paths, predictions):
        label_path = label_for_image(image_path)
        gt_count = len([line for line in label_path.read_text().splitlines() if line.strip()]) if label_path.exists() else 0
        pred_count = len(result.masks) if result.masks is not None else 0
        pred_masks_total += pred_count
        gt_total += gt_count
        if gt_count == 0:
            healthy_images += 1
            healthy_with_mask_fp += int(pred_count > 0)
            healthy_masks += pred_count
        else:
            disease_images += 1
            disease_mask_misses += int(pred_count == 0)
            labeled_mask_errors.append(abs(pred_count - gt_count) / max(1, gt_count))
            labeled_mask_exact.append(float(pred_count == gt_count))

    healthy_fp_rate = healthy_with_mask_fp / healthy_images if healthy_images else float('nan')
    healthy_fp_per_image = healthy_masks / healthy_images if healthy_images else float('nan')
    disease_miss_rate = disease_mask_misses / disease_images if disease_images else float('nan')
    mask_count_mae = float(np.mean(labeled_mask_errors)) if labeled_mask_errors else float('nan')
    mask_count_exact = float(np.mean(labeled_mask_exact)) if labeled_mask_exact else float('nan')
    return {
        'diagnostic_conf': conf,
        'healthy_images': healthy_images,
        'healthy_test_mask_fp_rate': healthy_fp_rate,
        'healthy_test_fp_masks_per_image': healthy_fp_per_image,
        'disease_images': disease_images,
        'labeled_test_disease_mask_miss_rate': disease_miss_rate,
        'labeled_test_mask_count_mae': mask_count_mae,
        'labeled_test_mask_count_exact': mask_count_exact,
        'gt_instances': gt_total,
        'pred_masks_at_diagnostic_conf': pred_masks_total,
    }


def standard_eval_row(model, mode, view, image_dir, class_names, fourier_seconds=float('nan')):
    image_paths = list_images_flat(image_dir)
    labeled_paths = [path for path in image_paths if label_has_instances(label_for_image(path))]
    if not image_paths or not labeled_paths:
        raise RuntimeError(f'Incomplete evaluation view {view}: all={len(image_paths)}, labeled={len(labeled_paths)}')

    full_yaml = write_image_list_yaml(image_paths, f'{mode}_{view}_full')
    labeled_yaml = write_image_list_yaml(labeled_paths, f'{mode}_{view}_labeled')

    full_metrics = model.val(
        data=str(full_yaml), split='test', imgsz=TRAIN_IMGSZ, batch=VAL_BATCH,
        conf=STANDARD_AP_CONF, iou=0.70, max_det=300, plots=False, verbose=False,
    )
    labeled_metrics = model.val(
        data=str(labeled_yaml), split='test', imgsz=TRAIN_IMGSZ, batch=VAL_BATCH,
        conf=STANDARD_AP_CONF, iou=0.70, max_det=300, plots=False, verbose=False,
    )
    diagnostics = fixed_threshold_diagnostics(model, image_paths)
    labeled_map50 = metric_or_nan(labeled_metrics, 'seg', 'map50')
    healthy_aware = (
        labeled_map50
        - COUNT_PENALTY_WEIGHT * diagnostics['labeled_test_mask_count_mae']
        - DISEASE_MISS_PENALTY_WEIGHT * diagnostics['labeled_test_disease_mask_miss_rate']
        - HEALTHY_FP_PENALTY_WEIGHT * diagnostics['healthy_test_mask_fp_rate']
    )
    row = {
        'mode': mode,
        'view': view,
        'evaluator': 'ultralytics_val',
        'ap_conf': STANDARD_AP_CONF,
        'images': len(image_paths),
        'labeled_images': len(labeled_paths),
        'healthy_images': len(image_paths) - len(labeled_paths),
        'fourier_precompute_seconds': fourier_seconds,
        'full_test_box_precision': metric_or_nan(full_metrics, 'box', 'mp'),
        'full_test_box_recall': metric_or_nan(full_metrics, 'box', 'mr'),
        'full_test_box_map50': metric_or_nan(full_metrics, 'box', 'map50'),
        'full_test_box_map50_95': metric_or_nan(full_metrics, 'box', 'map'),
        'full_test_mask_precision': metric_or_nan(full_metrics, 'seg', 'mp'),
        'full_test_mask_recall': metric_or_nan(full_metrics, 'seg', 'mr'),
        'full_test_mask_map50': metric_or_nan(full_metrics, 'seg', 'map50'),
        'full_test_mask_map50_95': metric_or_nan(full_metrics, 'seg', 'map'),
        'labeled_test_box_map50': metric_or_nan(labeled_metrics, 'box', 'map50'),
        'labeled_test_box_map50_95': metric_or_nan(labeled_metrics, 'box', 'map'),
        'labeled_test_mask_map50': labeled_map50,
        'labeled_test_mask_map50_95': metric_or_nan(labeled_metrics, 'seg', 'map'),
        **diagnostics,
        'healthy_aware_labeled_test_mask_map50': healthy_aware,
    }
    class_rows = (
        per_class_rows(full_metrics, mode, view, 'full', class_names)
        + per_class_rows(labeled_metrics, mode, view, 'labeled', class_names)
    )
    del full_metrics, labeled_metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, class_rows


def apply_test_view(image, view):
    if view == 'original':
        return image.copy()
    if view == 'horizontal_flip':
        return cv2.flip(image, 1)
    if view == 'vertical_flip':
        return cv2.flip(image, 0)
    if view == 'rotate_90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if view == 'rotate_180':
        return cv2.rotate(image, cv2.ROTATE_180)
    if view == 'rotate_270':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(view)


def transform_normalized_polygon(points, view):
    points = np.asarray(points, dtype=float).copy()
    x, y = points[:, 0].copy(), points[:, 1].copy()
    if view == 'original':
        pass
    elif view == 'horizontal_flip':
        points[:, 0], points[:, 1] = 1.0 - x, y
    elif view == 'vertical_flip':
        points[:, 0], points[:, 1] = x, 1.0 - y
    elif view == 'rotate_90':
        points[:, 0], points[:, 1] = 1.0 - y, x
    elif view == 'rotate_180':
        points[:, 0], points[:, 1] = 1.0 - x, 1.0 - y
    elif view == 'rotate_270':
        points[:, 0], points[:, 1] = y, 1.0 - x
    else:
        raise ValueError(view)
    return points.clip(0.0, 1.0)


def transform_segmentation_label(source_label, destination_label, view):
    source_label = Path(source_label)
    destination_label = Path(destination_label)
    output_lines = []
    if source_label.exists():
        for line in source_label.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 7 or (len(parts) - 1) % 2:
                raise ValueError(f'Expected YOLO polygon label, got: {source_label} :: {line}')
            class_id = parts[0]
            points = np.asarray([float(value) for value in parts[1:]], dtype=float).reshape(-1, 2)
            transformed = transform_normalized_polygon(points, view)
            coords = ' '.join(f'{value:.8f}' for value in transformed.reshape(-1))
            output_lines.append(f'{class_id} {coords}')
    destination_label.write_text('\n'.join(output_lines) + ('\n' if output_lines else ''), encoding='utf-8')


def hardlink_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except Exception:
        shutil.copy2(source, destination)


def build_w1_test_views():
    root = EXPERIMENT_ROOT / 'offline_test_views_w1'
    if root.exists():
        shutil.rmtree(root)
    source_images = list_images_flat(Path(base_path) / 'test' / 'images')
    source_labels = Path(base_path) / 'test' / 'labels'
    canonical_w1_images = W1_DATASET_DIR / 'test' / 'images'
    timings = {}

    for view in TEST_VIEWS:
        view_images = root / view / 'images'
        view_labels = root / view / 'labels'
        view_images.mkdir(parents=True, exist_ok=True)
        view_labels.mkdir(parents=True, exist_ok=True)

        if view == 'original':
            started = time.perf_counter()
            for image_path in source_images:
                canonical_image = canonical_w1_images / image_path.name
                if not canonical_image.exists():
                    raise FileNotFoundError(canonical_image)
                shutil.copy2(canonical_image, view_images / image_path.name)
                shutil.copy2(source_labels / f'{image_path.stem}.txt', view_labels / f'{image_path.stem}.txt')
            timings[view] = float('nan')  # Canonical original-view W1 was created during training.
            continue

        started = time.perf_counter()
        image_batch, path_batch, batch_shape = [], [], None

        def flush_fourier_batch():
            nonlocal image_batch, path_batch, batch_shape
            if image_batch:
                apply_fourier_batch(
                    image_batch, path_batch, sigma=FOURIER_SIGMA,
                    alpha=FOURIER_ALPHA, device=fourier_device()
                )
            image_batch, path_batch, batch_shape = [], [], None

        for image_path in source_images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            transformed = apply_test_view(image, view)
            output_name = f'{view}__{image_path.name}'
            output_image = view_images / output_name
            transform_segmentation_label(
                source_labels / f'{image_path.stem}.txt',
                view_labels / f'{Path(output_name).stem}.txt',
                view,
            )
            if batch_shape is not None and transformed.shape != batch_shape:
                flush_fourier_batch()
            image_batch.append(transformed)
            path_batch.append(output_image)
            batch_shape = transformed.shape
            if len(image_batch) >= FOURIER_BATCH_SIZE:
                flush_fourier_batch()
        flush_fourier_batch()
        timings[view] = time.perf_counter() - started

    pooled_images = root / 'pooled' / 'images'
    pooled_labels = root / 'pooled' / 'labels'
    pooled_images.mkdir(parents=True, exist_ok=True)
    pooled_labels.mkdir(parents=True, exist_ok=True)
    for view in TEST_VIEWS:
        for image_path in list_images_flat(root / view / 'images'):
            output_name = image_path.name if view != 'original' else f'original__{image_path.name}'
            hardlink_or_copy(image_path, pooled_images / output_name)
            hardlink_or_copy(label_for_image(image_path), pooled_labels / f'{Path(output_name).stem}.txt')

    expected = len(source_images) * len(TEST_VIEWS)
    actual = len(list_images_flat(pooled_images))
    if actual != expected:
        raise RuntimeError(f'Expected {expected} pooled images, found {actual}')
    (EVAL_REPORT_DIR / '048_offline_view_fourier_timing.json').write_text(
        json.dumps(timings, indent=2), encoding='utf-8'
    )
    return root, timings
'''


standard_run_source = r'''# Run M0 and M1 with standard Ultralytics validation AP.
data_cfg = yaml.safe_load(W1_DATA_YAML.read_text(encoding='utf-8'))
names_obj = data_cfg.get('names', {})
if isinstance(names_obj, dict):
    CLASS_NAMES = [names_obj[key] for key in sorted(names_obj, key=lambda value: int(value))]
else:
    CLASS_NAMES = list(names_obj)

w1_model = YOLO(str(W1_CHECKPOINT))
standard_rows = []
standard_class_rows = []

if RUN_M0_STANDARD_REFERENCE:
    row, class_rows = standard_eval_row(
        w1_model, 'M0_standard_w1_reference', 'original', W1_DATASET_DIR / 'test' / 'images', CLASS_NAMES
    )
    standard_rows.append(row)
    standard_class_rows.extend(class_rows)
    print('M0 complete')
    display(pd.DataFrame([row]))

OFFLINE_VIEW_ROOT = None
offline_fourier_timings = {}
if RUN_M1_OFFLINE_AUG_TEST:
    OFFLINE_VIEW_ROOT, offline_fourier_timings = build_w1_test_views()

if RUN_M1_OFFLINE_AUG_TEST:
    for view in TEST_VIEWS:
        row, class_rows = standard_eval_row(
            w1_model,
            'M1_offline_augmented_test',
            view,
            OFFLINE_VIEW_ROOT / view / 'images',
            CLASS_NAMES,
            fourier_seconds=offline_fourier_timings.get(view, float('nan')),
        )
        standard_rows.append(row)
        standard_class_rows.extend(class_rows)
        print(f'M1 view complete: {view}')

    row, class_rows = standard_eval_row(
        w1_model,
        'M1_offline_augmented_test',
        'pooled_six_views',
        OFFLINE_VIEW_ROOT / 'pooled' / 'images',
        CLASS_NAMES,
        fourier_seconds=sum(offline_fourier_timings.values()),
    )
    standard_rows.append(row)
    standard_class_rows.extend(class_rows)

standard_metrics_df = pd.DataFrame(standard_rows)
standard_per_class_df = pd.DataFrame(standard_class_rows)
standard_metrics_csv = EVAL_REPORT_DIR / '048_w1_standard_and_aug_test_metrics.csv'
standard_per_class_csv = EVAL_REPORT_DIR / '048_w1_standard_and_aug_test_per_class.csv'
standard_metrics_df.to_csv(standard_metrics_csv, index=False)
standard_per_class_df.to_csv(standard_per_class_csv, index=False)
print('Saved:', standard_metrics_csv)
print('Saved:', standard_per_class_csv)
display(standard_metrics_df)
'''


tta_helpers_source = r'''# Memory-conscious custom evaluator and W1 TTA helpers.
from collections import defaultdict
import pickle

TTA_CONFIG = {
    'transforms': TEST_VIEWS,
    'candidate_conf': STANDARD_AP_CONF,
    'diagnostic_conf': DIAGNOSTIC_CONF,
    'iou_nms': 0.70,
    'max_det': 300,
    'matching_metric': 'mask_iou',
    'matching_iou_threshold': 0.50,
    'matching_assignment': 'greedy_best_match',
    'one_to_one': True,
    'class_aware_matching': True,
    'minimum_support': 4,
    'total_views': 6,
    'mask_fusion': 'average_binary_masks',
    'mask_threshold': 0.50,
    'final_mask_nms_iou': 0.50,
}
(EVAL_REPORT_DIR / '048_tta_config.json').write_text(json.dumps(TTA_CONFIG, indent=2), encoding='utf-8')


def inverse_mask(mask, view, original_hw):
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    if view == 'horizontal_flip':
        mask = cv2.flip(mask, 1)
    elif view == 'vertical_flip':
        mask = cv2.flip(mask, 0)
    elif view == 'rotate_90':
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif view == 'rotate_180':
        mask = cv2.rotate(mask, cv2.ROTATE_180)
    elif view == 'rotate_270':
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif view != 'original':
        raise ValueError(view)
    height, width = original_hw
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def boolean_mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def mask_box(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)


def pack_instance(instance):
    mask = np.asarray(instance['mask'], dtype=bool)
    return {
        'class_id': int(instance['class_id']),
        'confidence': float(instance.get('confidence', 1.0)),
        'support': int(instance.get('support', 1)),
        'shape': tuple(mask.shape),
        'packed_mask': np.packbits(mask.reshape(-1)),
        'box': mask_box(mask),
    }


POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def packed_mask_iou(instance_a, instance_b):
    if tuple(instance_a['shape']) != tuple(instance_b['shape']):
        raise ValueError('Packed masks must have equal shapes')
    left = instance_a['packed_mask']
    right = instance_b['packed_mask']
    intersection = POPCOUNT[np.bitwise_and(left, right)].sum(dtype=np.uint64)
    union = POPCOUNT[np.bitwise_or(left, right)].sum(dtype=np.uint64)
    return float(intersection / union) if union else 0.0


def box_iou_xyxy(box_a, box_b):
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def predict_tta_view(model, image, view, original_hw):
    transformed = apply_test_view(image, view)
    fourier_started = time.perf_counter()
    transformed = fourier_highpass_boost_image_cpu(
        transformed, sigma=FOURIER_SIGMA, alpha=FOURIER_ALPHA
    )
    fourier_seconds = time.perf_counter() - fourier_started

    inference_started = time.perf_counter()
    result = model.predict(
        source=transformed,
        imgsz=TRAIN_IMGSZ,
        conf=TTA_CONFIG['candidate_conf'],
        iou=TTA_CONFIG['iou_nms'],
        max_det=TTA_CONFIG['max_det'],
        agnostic_nms=False,
        verbose=False,
    )[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    instances = []
    if result.masks is not None and result.boxes is not None:
        raw_masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        view_height, view_width = transformed.shape[:2]
        for raw_mask, class_id, score in zip(raw_masks, classes, scores):
            view_mask = cv2.resize(
                raw_mask, (view_width, view_height), interpolation=cv2.INTER_NEAREST
            ) >= 0.5
            instances.append({
                'mask': inverse_mask(view_mask, view, original_hw),
                'class_id': int(class_id),
                'confidence': float(score),
                'view': view,
            })
    del result, transformed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return instances, fourier_seconds, inference_seconds


def fused_track_mask(track, threshold):
    accumulator = np.zeros_like(track[0]['mask'], dtype=np.uint16)
    for instance in track:
        accumulator += instance['mask'].astype(np.uint16)
    return (accumulator / len(track)) >= threshold


def fuse_tta_views(view_predictions, transforms, minimum_support):
    tracks = []
    for view in transforms:
        instances = view_predictions.get(view, [])
        candidates = []
        for track_id, track in enumerate(tracks):
            reference = fused_track_mask(track, TTA_CONFIG['mask_threshold'])
            for instance_id, instance in enumerate(instances):
                if TTA_CONFIG['class_aware_matching'] and track[0]['class_id'] != instance['class_id']:
                    continue
                overlap = boolean_mask_iou(reference, instance['mask'])
                if overlap >= TTA_CONFIG['matching_iou_threshold']:
                    candidates.append((overlap, track_id, instance_id))

        used_tracks, used_instances = set(), set()
        for _, track_id, instance_id in sorted(candidates, reverse=True):
            if track_id in used_tracks or instance_id in used_instances:
                continue
            tracks[track_id].append(instances[instance_id])
            used_tracks.add(track_id)
            used_instances.add(instance_id)
        tracks.extend([[instance] for instance_id, instance in enumerate(instances) if instance_id not in used_instances])

    fused = []
    for track in tracks:
        if len(track) < minimum_support:
            continue
        fused_mask = fused_track_mask(track, TTA_CONFIG['mask_threshold'])
        if not fused_mask.any():
            continue
        fused.append({
            'mask': fused_mask,
            'class_id': track[0]['class_id'],
            'confidence': float(np.mean([instance['confidence'] for instance in track])),
            'support': len(track),
        })

    kept = []
    for prediction in sorted(fused, key=lambda item: item['confidence'], reverse=True):
        suppress = any(
            prediction['class_id'] == old['class_id']
            and boolean_mask_iou(prediction['mask'], old['mask']) >= TTA_CONFIG['final_mask_nms_iou']
            for old in kept
        )
        if not suppress:
            kept.append(prediction)
    return kept


def load_gt_instances(label_path, image_hw):
    height, width = image_hw
    instances = []
    label_path = Path(label_path)
    if not label_path.exists():
        return instances
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        class_id = int(float(parts[0]))
        points = np.asarray(parts[1:], dtype=float).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        instances.append(pack_instance({'class_id': class_id, 'confidence': 1.0, 'mask': mask > 0}))
    return instances


def ap_from_pr(recall, precision):
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[1.0, precision, 0.0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    grid = np.linspace(0.0, 1.0, 101)
    values = np.interp(grid, mrec, mpre)
    return float(np.trapezoid(values, grid))


def evaluate_one_threshold(records, class_id, iou_threshold, metric_type):
    total_gt = sum(sum(gt['class_id'] == class_id for gt in record['gt']) for record in records)
    ranked = []
    for image_id, record in enumerate(records):
        ranked.extend(
            (prediction['confidence'], image_id, prediction)
            for prediction in record['pred'] if prediction['class_id'] == class_id
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    matched = defaultdict(set)
    true_positive, false_positive = [], []
    for _, image_id, prediction in ranked:
        candidates = []
        for gt_id, gt in enumerate(records[image_id]['gt']):
            if gt['class_id'] != class_id or gt_id in matched[image_id]:
                continue
            overlap = (
                packed_mask_iou(prediction, gt)
                if metric_type == 'mask'
                else box_iou_xyxy(prediction['box'], gt['box'])
            )
            candidates.append((overlap, gt_id))
        best_iou, best_gt = max(candidates, default=(0.0, -1))
        hit = best_iou >= iou_threshold
        true_positive.append(float(hit))
        false_positive.append(float(not hit))
        if hit:
            matched[image_id].add(best_gt)

    tp_sum = float(sum(true_positive))
    fp_sum = float(sum(false_positive))
    precision = tp_sum / max(tp_sum + fp_sum, 1e-12)
    recall = tp_sum / max(total_gt, 1e-12)
    if total_gt and ranked:
        tp_curve = np.cumsum(true_positive)
        fp_curve = np.cumsum(false_positive)
        recall_curve = tp_curve / total_gt
        precision_curve = tp_curve / np.maximum(tp_curve + fp_curve, 1e-12)
        average_precision = ap_from_pr(recall_curve, precision_curve)
    else:
        average_precision = 0.0 if total_gt else float('nan')
    return {
        'precision': precision,
        'recall': recall,
        'ap': average_precision,
        'gt': total_gt,
        'pred': len(ranked),
    }


def evaluate_custom_ap(records, class_names, mode, scope):
    thresholds = np.arange(0.50, 0.96, 0.05)
    per_class = []
    for class_id, class_name in enumerate(class_names):
        row = {'mode': mode, 'scope': scope, 'class_id': class_id, 'class_name': class_name}
        for metric_type, suffix in (('box', 'box'), ('mask', 'mask')):
            evaluations = [
                evaluate_one_threshold(records, class_id, float(threshold), metric_type)
                for threshold in thresholds
            ]
            row.update({
                f'{suffix}_precision': evaluations[0]['precision'],
                f'{suffix}_recall': evaluations[0]['recall'],
                f'{suffix}_map50': evaluations[0]['ap'],
                f'{suffix}_map50_95': float(np.nanmean([item['ap'] for item in evaluations])),
            })
            if metric_type == 'mask':
                row['gt_instances'] = evaluations[0]['gt']
                row['pred_instances'] = evaluations[0]['pred']
        per_class.append(row)

    frame = pd.DataFrame(per_class)
    valid = frame[frame['gt_instances'] > 0]
    overall = {
        'mode': mode,
        'scope': scope,
        'evaluator': 'custom_tta_101_point_ap',
        'ap_conf': TTA_CONFIG['candidate_conf'],
        'box_precision': float(valid['box_precision'].mean()),
        'box_recall': float(valid['box_recall'].mean()),
        'box_map50': float(valid['box_map50'].mean()),
        'box_map50_95': float(valid['box_map50_95'].mean()),
        'mask_precision': float(valid['mask_precision'].mean()),
        'mask_recall': float(valid['mask_recall'].mean()),
        'mask_map50': float(valid['mask_map50'].mean()),
        'mask_map50_95': float(valid['mask_map50_95'].mean()),
    }
    return overall, per_class


def custom_diagnostics(records, threshold=DIAGNOSTIC_CONF):
    healthy_images = healthy_with_fp = healthy_masks = 0
    disease_images = disease_misses = 0
    count_errors = []
    count_exact = []
    for record in records:
        predictions = [prediction for prediction in record['pred'] if prediction['confidence'] >= threshold]
        gt_count = len(record['gt'])
        pred_count = len(predictions)
        if gt_count == 0:
            healthy_images += 1
            healthy_with_fp += int(pred_count > 0)
            healthy_masks += pred_count
        else:
            disease_images += 1
            disease_misses += int(pred_count == 0)
            count_errors.append(abs(pred_count - gt_count) / max(1, gt_count))
            count_exact.append(float(pred_count == gt_count))
    return {
        'diagnostic_conf': threshold,
        'healthy_images': healthy_images,
        'healthy_test_mask_fp_rate': healthy_with_fp / healthy_images if healthy_images else float('nan'),
        'healthy_test_fp_masks_per_image': healthy_masks / healthy_images if healthy_images else float('nan'),
        'disease_images': disease_images,
        'labeled_test_disease_mask_miss_rate': disease_misses / disease_images if disease_images else float('nan'),
        'labeled_test_mask_count_mae': float(np.mean(count_errors)) if count_errors else float('nan'),
        'labeled_test_mask_count_exact': float(np.mean(count_exact)) if count_exact else float('nan'),
    }
'''


tta_run_source = r'''# Run M2 and M3 in one pass over the original test images.
tta_rows = []
tta_per_class_rows = []
tta_timing_rows = []

if RUN_M2_SINGLE_VIEW_CONTROL or RUN_M3_SIX_VIEW_TTA:
    tta_model = YOLO(str(W1_CHECKPOINT))
    source_test_images = list_images_flat(Path(base_path) / 'test' / 'images')
    source_test_labels = Path(base_path) / 'test' / 'labels'
    records_m2, records_m3 = [], []

    for image_index, image_path in enumerate(source_test_images, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)

        view_predictions = {}
        fourier_seconds = inference_seconds = 0.0
        original_fourier_seconds = original_inference_seconds = 0.0
        for view in TEST_VIEWS:
            instances, view_fourier_seconds, view_inference_seconds = predict_tta_view(
                tta_model, image, view, image.shape[:2]
            )
            view_predictions[view] = instances
            fourier_seconds += view_fourier_seconds
            inference_seconds += view_inference_seconds
            if view == 'original':
                original_fourier_seconds = view_fourier_seconds
                original_inference_seconds = view_inference_seconds

        gt = load_gt_instances(source_test_labels / f'{image_path.stem}.txt', image.shape[:2])

        if RUN_M2_SINGLE_VIEW_CONTROL:
            fusion_started = time.perf_counter()
            single_predictions = fuse_tta_views(view_predictions, ['original'], minimum_support=1)
            single_fusion_seconds = time.perf_counter() - fusion_started
            records_m2.append({
                'path': str(image_path),
                'gt': gt,
                'pred': [pack_instance(prediction) for prediction in single_predictions],
            })
            tta_timing_rows.append({
                'mode': 'M2_single_view_custom_control',
                'image': image_path.name,
                'fourier_seconds': original_fourier_seconds,
                'inference_seconds': original_inference_seconds,
                'fusion_seconds': single_fusion_seconds,
                'total_seconds': original_fourier_seconds + original_inference_seconds + single_fusion_seconds,
            })

        if RUN_M3_SIX_VIEW_TTA:
            fusion_started = time.perf_counter()
            fused_predictions = fuse_tta_views(
                view_predictions, TEST_VIEWS, minimum_support=TTA_CONFIG['minimum_support']
            )
            fusion_seconds = time.perf_counter() - fusion_started
            records_m3.append({
                'path': str(image_path),
                'gt': gt,
                'pred': [pack_instance(prediction) for prediction in fused_predictions],
            })
            tta_timing_rows.append({
                'mode': 'M3_six_view_w1_tta',
                'image': image_path.name,
                'fourier_seconds': fourier_seconds,
                'inference_seconds': inference_seconds,
                'fusion_seconds': fusion_seconds,
                'total_seconds': fourier_seconds + inference_seconds + fusion_seconds,
            })

        del view_predictions, image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if image_index % 10 == 0 or image_index == len(source_test_images):
            print(f'TTA inference: {image_index}/{len(source_test_images)}')
            with open(EVAL_REPORT_DIR / '048_tta_partial_records.pkl', 'wb') as stream:
                pickle.dump({'M2': records_m2, 'M3': records_m3}, stream, protocol=pickle.HIGHEST_PROTOCOL)

    for mode, records in (
        ('M2_single_view_custom_control', records_m2),
        ('M3_six_view_w1_tta', records_m3),
    ):
        if not records:
            continue
        labeled_records = [record for record in records if record['gt']]
        full_metrics, full_per_class = evaluate_custom_ap(records, CLASS_NAMES, mode, 'full')
        labeled_metrics, labeled_per_class = evaluate_custom_ap(labeled_records, CLASS_NAMES, mode, 'labeled')
        diagnostics = custom_diagnostics(records)
        healthy_aware = (
            labeled_metrics['mask_map50']
            - COUNT_PENALTY_WEIGHT * diagnostics['labeled_test_mask_count_mae']
            - DISEASE_MISS_PENALTY_WEIGHT * diagnostics['labeled_test_disease_mask_miss_rate']
            - HEALTHY_FP_PENALTY_WEIGHT * diagnostics['healthy_test_mask_fp_rate']
        )
        timing_frame = pd.DataFrame([row for row in tta_timing_rows if row['mode'] == mode])
        row = {
            'mode': mode,
            'evaluator': 'custom_tta_101_point_ap',
            'ap_conf': STANDARD_AP_CONF,
            'images': len(records),
            'labeled_images': len(labeled_records),
            'full_test_box_precision': full_metrics['box_precision'],
            'full_test_box_recall': full_metrics['box_recall'],
            'full_test_box_map50': full_metrics['box_map50'],
            'full_test_box_map50_95': full_metrics['box_map50_95'],
            'full_test_mask_precision': full_metrics['mask_precision'],
            'full_test_mask_recall': full_metrics['mask_recall'],
            'full_test_mask_map50': full_metrics['mask_map50'],
            'full_test_mask_map50_95': full_metrics['mask_map50_95'],
            'labeled_test_box_map50': labeled_metrics['box_map50'],
            'labeled_test_box_map50_95': labeled_metrics['box_map50_95'],
            'labeled_test_mask_map50': labeled_metrics['mask_map50'],
            'labeled_test_mask_map50_95': labeled_metrics['mask_map50_95'],
            **diagnostics,
            'healthy_aware_labeled_test_mask_map50': healthy_aware,
            'mean_fourier_ms_per_original': 1000.0 * timing_frame['fourier_seconds'].mean(),
            'mean_inference_ms_per_original': 1000.0 * timing_frame['inference_seconds'].mean(),
            'mean_fusion_ms_per_original': 1000.0 * timing_frame['fusion_seconds'].mean(),
            'mean_total_ms_per_original': 1000.0 * timing_frame['total_seconds'].mean(),
        }
        tta_rows.append(row)
        tta_per_class_rows.extend(full_per_class + labeled_per_class)

tta_metrics_df = pd.DataFrame(tta_rows)
tta_per_class_df = pd.DataFrame(tta_per_class_rows)
tta_timing_df = pd.DataFrame(tta_timing_rows)
tta_metrics_csv = EVAL_REPORT_DIR / '048_w1_tta_metrics.csv'
tta_per_class_csv = EVAL_REPORT_DIR / '048_w1_tta_per_class.csv'
tta_timing_csv = EVAL_REPORT_DIR / '048_w1_tta_timing.csv'
tta_metrics_df.to_csv(tta_metrics_csv, index=False)
tta_per_class_df.to_csv(tta_per_class_csv, index=False)
tta_timing_df.to_csv(tta_timing_csv, index=False)
print('Saved:', tta_metrics_csv)
print('Saved:', tta_per_class_csv)
print('Saved:', tta_timing_csv)
display(tta_metrics_df)
'''


summary_source = r'''# Consolidate all mode rows without implying identical evaluator implementations.
combined_frames = []
if 'standard_metrics_df' in globals() and not standard_metrics_df.empty:
    combined_frames.append(standard_metrics_df)
if 'tta_metrics_df' in globals() and not tta_metrics_df.empty:
    combined_frames.append(tta_metrics_df)

combined_metrics_df = pd.concat(combined_frames, ignore_index=True, sort=False) if combined_frames else pd.DataFrame()
combined_metrics_csv = EVAL_REPORT_DIR / '048_w1_all_evaluation_modes.csv'
combined_metrics_df.to_csv(combined_metrics_csv, index=False)

comparison_notes = {
    'M0_vs_M1': 'Valid standard Ultralytics AP comparison; M1 views are correlated copies, so per-view rows are primary.',
    'M2_vs_M3': 'Valid matched custom-evaluator comparison for the effect of six-view TTA.',
    'M0_vs_M2': 'Evaluator calibration only; custom mask rasterization and AP integration can create a delta.',
    'M0_vs_M3': 'Descriptive only unless the M0-to-M2 evaluator delta is disclosed.',
}
(EVAL_REPORT_DIR / '048_comparison_boundaries.json').write_text(
    json.dumps(comparison_notes, indent=2), encoding='utf-8'
)
print('Saved:', combined_metrics_csv)
display(combined_metrics_df)
'''


checklist_source = r'''# Final download checklist: only references variables created unconditionally above.
import zipfile

training_summary_csv = REPORT_DIR / '048_w1_train_summary.csv'
training_paper_csv = REPORT_DIR / '048_w1_train_paper_row.csv'
training_partial_csv = REPORT_DIR / '048_w1_train_partial.csv'
train_args_path = REPORT_DIR / 'train_args_048_w1_train_aug_test_tta_seed42.json'

required_paths = [
    training_summary_csv,
    training_paper_csv,
    training_partial_csv,
    train_args_path,
    REPORT_DIR / 'split_manifests',
    W1_CHECKPOINT,
    EVAL_REPORT_DIR,
]

print('Final download checklist')
for index, path in enumerate(required_paths, 1):
    print(f'{index}. {path} | exists={Path(path).exists()}')

package_path = WORK_DIR / '048_w1_train_aug_test_tta_outputs.zip'
with zipfile.ZipFile(package_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for report_path in REPORT_DIR.rglob('*'):
        if report_path.is_file() and report_path.suffix.lower() != '.pkl':
            archive.write(report_path, arcname=str(Path('reports') / report_path.relative_to(REPORT_DIR)))
    archive.write(W1_CHECKPOINT, arcname='weights/w1_best.pt')
    results_csv = Path(w1_row['run_path']) / 'results.csv'
    if results_csv.exists():
        archive.write(results_csv, arcname='training/results.csv')

print('Compact reports-and-checkpoint package:', package_path)
print('Package size MB:', round(package_path.stat().st_size / (1024 * 1024), 2))
print('On Kaggle, download this ZIP from /kaggle/working after the run completes.')
'''


cells.extend(
    [
        markdown_cell("## Resolve the Newly Trained W1 Model\n"),
        code_cell(resolve_source),
        markdown_cell("## M0 and M1: Standard AP Evaluation\n"),
        code_cell(standard_helpers_source),
        code_cell(standard_run_source),
        markdown_cell("## M2 and M3: Matched Custom TTA Evaluation\n"),
        code_cell(tta_helpers_source),
        code_cell(tta_run_source),
        markdown_cell("## Consolidated Evaluation Table\n"),
        code_cell(summary_source),
        markdown_cell("## Final Download Checklist\n"),
        code_cell(checklist_source),
    ]
)

notebook["cells"] = cells
notebook.setdefault("metadata", {})
notebook["metadata"].setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
notebook["metadata"].setdefault("language_info", {"name": "python", "version": "3.x"})
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Generated {OUTPUT}")
