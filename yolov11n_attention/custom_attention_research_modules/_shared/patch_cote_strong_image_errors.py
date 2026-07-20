"""Patch CoTE strong notebook to print image-level class prediction errors."""

from pathlib import Path

import nbformat


NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "custom_attention_research_modules"
    / "cote_gate"
    / "cote_gate_strong_augmentation.ipynb"
)


FUNCTION_BLOCK = r'''

def _safe_metric_key(name):
    key = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(name))
    key = '_'.join(part for part in key.split('_') if part)
    return key or 'class'


def _read_label_class_ids(label_path):
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    class_ids = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_ids.append(int(float(parts[0])))
        except Exception:
            continue
    return class_ids


def _prediction_class_ids(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    try:
        return [int(v) for v in result.boxes.cls.detach().cpu().tolist()]
    except Exception:
        return []


def image_level_class_error_summary(model, images_dir, labels_dir, conf=PREDICT_CONF_FOR_COUNT):
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(Path(images_dir).glob(f'*{ext}'))
    image_paths = sorted(image_paths)

    per_class = {
        cid: {
            'class_id': cid,
            'class_name': class_names[cid] if cid < len(class_names) else f'Unknown({cid})',
            'gt_images': 0,
            'pred_images': 0,
            'miss_images': 0,
            'fp_images': 0,
            'wrong_images': 0,
        }
        for cid in range(len(class_names))
    }

    if not image_paths:
        return {
            'images': 0,
            'any_wrong_images': 0,
            'any_wrong_rate': float('nan'),
            'healthy_images': 0,
            'healthy_wrong_images': 0,
            'healthy_wrong_rate': float('nan'),
            'diseased_images': 0,
            'class_rows': list(per_class.values()),
        }

    results = model.predict(source=[str(p) for p in image_paths], imgsz=TRAIN_IMGSZ, conf=conf, verbose=False)
    any_wrong_images = 0
    healthy_images = 0
    healthy_wrong_images = 0
    diseased_images = 0

    for image_path, result in zip(image_paths, results):
        gt_classes = _read_label_class_ids(Path(labels_dir) / f'{image_path.stem}.txt')
        pred_classes = _prediction_class_ids(result)
        gt_set = set(gt_classes)
        pred_set = set(pred_classes)

        if gt_classes:
            diseased_images += 1
        else:
            healthy_images += 1
            if pred_classes:
                healthy_wrong_images += 1

        if gt_set != pred_set:
            any_wrong_images += 1

        for cid, row in per_class.items():
            has_gt = cid in gt_set
            has_pred = cid in pred_set
            if has_gt:
                row['gt_images'] += 1
            if has_pred:
                row['pred_images'] += 1
            if has_gt and not has_pred:
                row['miss_images'] += 1
            if has_pred and not has_gt:
                row['fp_images'] += 1
            if has_gt != has_pred:
                row['wrong_images'] += 1

    return {
        'images': len(image_paths),
        'any_wrong_images': any_wrong_images,
        'any_wrong_rate': any_wrong_images / len(image_paths),
        'healthy_images': healthy_images,
        'healthy_wrong_images': healthy_wrong_images,
        'healthy_wrong_rate': healthy_wrong_images / healthy_images if healthy_images else float('nan'),
        'diseased_images': diseased_images,
        'class_rows': list(per_class.values()),
    }


def flatten_image_level_class_errors(prefix, summary):
    flat = {
        f'{prefix}_image_total': summary['images'],
        f'{prefix}_image_wrong_total': summary['any_wrong_images'],
        f'{prefix}_image_wrong_rate': summary['any_wrong_rate'],
        f'{prefix}_healthy_images': summary['healthy_images'],
        f'{prefix}_healthy_wrong_images': summary['healthy_wrong_images'],
        f'{prefix}_healthy_wrong_rate': summary['healthy_wrong_rate'],
        f'{prefix}_diseased_images': summary['diseased_images'],
    }
    for row in summary['class_rows']:
        key = _safe_metric_key(row['class_name'])
        flat[f'{prefix}_{key}_gt_images'] = row['gt_images']
        flat[f'{prefix}_{key}_pred_images'] = row['pred_images']
        flat[f'{prefix}_{key}_miss_images'] = row['miss_images']
        flat[f'{prefix}_{key}_fp_images'] = row['fp_images']
        flat[f'{prefix}_{key}_wrong_images'] = row['wrong_images']
    return flat


def print_image_level_class_errors(title, summary):
    print(f'\n{title}')
    print(f"  Images: {summary['images']}")
    print(f"  Images with wrong predicted class set: {summary['any_wrong_images']} ({summary['any_wrong_rate']:.4f})")
    print(f"  Healthy images predicted wrong: {summary['healthy_wrong_images']} / {summary['healthy_images']} ({summary['healthy_wrong_rate']:.4f})")
    df = pd.DataFrame(summary['class_rows'])
    if not df.empty:
        display(df[['class_id', 'class_name', 'gt_images', 'pred_images', 'miss_images', 'fp_images', 'wrong_images']])
'''


def patch_notebook() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)

    cell17 = nb.cells[17].source
    if "def image_level_class_error_summary" not in cell17:
        marker = "\ndef healthy_false_positive_summary(model, images_dir, conf=PREDICT_CONF_FOR_COUNT):"
        cell17 = cell17.replace(marker, FUNCTION_BLOCK + marker, 1)

    if "full_test_class_errors = image_level_class_error_summary" not in cell17:
        old = """        healthy_test_fp = healthy_false_positive_summary(
            best_model,
            healthy_eval_dir / 'test' / 'images',
        )
        labeled_test_map50 = metric_value(labeled_test, 'seg.map50')
"""
        new = """        healthy_test_fp = healthy_false_positive_summary(
            best_model,
            healthy_eval_dir / 'test' / 'images',
        )
        full_test_class_errors = image_level_class_error_summary(
            best_model,
            dataset_dir / 'test' / 'images',
            dataset_dir / 'test' / 'labels',
        )
        labeled_test_class_errors = image_level_class_error_summary(
            best_model,
            labeled_eval_dir / 'test' / 'images',
            labeled_eval_dir / 'test' / 'labels',
        )
        print_image_level_class_errors('Full test image-level class error summary', full_test_class_errors)
        print_image_level_class_errors('Labeled-only test image-level class error summary', labeled_test_class_errors)
        labeled_test_map50 = metric_value(labeled_test, 'seg.map50')
"""
        cell17 = cell17.replace(old, new, 1)

    if "**flatten_image_level_class_errors('full_test'" not in cell17:
        old = """            'healthy_test_avg_fp_confidence': healthy_test_fp['healthy_avg_fp_confidence'],
            'healthy_aware_labeled_test_mask_map50': healthy_aware_score(labeled_test_map50, labeled_test_count, healthy_test_fp),
        })
"""
        new = """            'healthy_test_avg_fp_confidence': healthy_test_fp['healthy_avg_fp_confidence'],
            **flatten_image_level_class_errors('full_test', full_test_class_errors),
            **flatten_image_level_class_errors('labeled_test', labeled_test_class_errors),
            'healthy_aware_labeled_test_mask_map50': healthy_aware_score(labeled_test_map50, labeled_test_count, healthy_test_fp),
        })
"""
        cell17 = cell17.replace(old, new, 1)

    nb.cells[17].source = cell17

    cell19 = nb.cells[19].source
    if "full_test_image_wrong_total" not in cell19:
        old = """    'healthy_test_mask_fp_rate',
    'healthy_aware_labeled_test_mask_map50',
]
"""
        new = """    'healthy_test_mask_fp_rate',
    'full_test_image_wrong_total',
    'full_test_image_wrong_rate',
    'full_test_healthy_wrong_images',
    'full_test_bg_wrong_images',
    'full_test_wssv_wrong_images',
    'labeled_test_image_wrong_total',
    'labeled_test_image_wrong_rate',
    'labeled_test_bg_wrong_images',
    'labeled_test_wssv_wrong_images',
    'healthy_aware_labeled_test_mask_map50',
]
"""
        cell19 = cell19.replace(old, new, 1)
    nb.cells[19].source = cell19

    cell20 = nb.cells[20].source
    if "image-level class prediction errors" not in cell20:
        cell20 += (
            "\n\nThe notebook now also prints image-level class prediction errors "
            "for the full test set, the labeled-only test set, and healthy-negative images."
        )
    nb.cells[20].source = cell20

    cell21 = nb.cells[21].source
    if "Image-level class error summaries" not in cell21:
        cell21 = cell21.replace(
            "import random\nfrom pathlib import Path\n",
            "import random\nfrom pathlib import Path\nimport pandas as pd\n",
            1,
        )
        old = """print('Healthy test false-positive masks per image:', healthy_test_fp['healthy_fp_masks_per_image'])


def draw_yolo_segmentation_labels"""
        new = """print('Healthy test false-positive masks per image:', healthy_test_fp['healthy_fp_masks_per_image'])

print('\\nImage-level class error summaries at prediction conf:', PREDICT_CONF_FOR_COUNT)
full_test_class_errors = image_level_class_error_summary(
    model_inference,
    selected_dataset_dir / 'test' / 'images',
    selected_dataset_dir / 'test' / 'labels',
)
labeled_test_class_errors = image_level_class_error_summary(
    model_inference,
    selected_labeled_yaml.parent / 'test' / 'images',
    selected_labeled_yaml.parent / 'test' / 'labels',
)
print_image_level_class_errors('Full test image-level class error summary', full_test_class_errors)
print_image_level_class_errors('Labeled-only test image-level class error summary', labeled_test_class_errors)

full_error_csv = REPORT_DIR / f'{EXPERIMENT_NAME}_full_test_image_class_errors.csv'
labeled_error_csv = REPORT_DIR / f'{EXPERIMENT_NAME}_labeled_test_image_class_errors.csv'
pd.DataFrame(full_test_class_errors['class_rows']).to_csv(full_error_csv, index=False)
pd.DataFrame(labeled_test_class_errors['class_rows']).to_csv(labeled_error_csv, index=False)
print('Saved full-test image class errors:', full_error_csv)
print('Saved labeled-test image class errors:', labeled_error_csv)


def draw_yolo_segmentation_labels"""
        cell21 = cell21.replace(old, new, 1)

    nb.cells[21].source = cell21

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    nbformat.write(nb, NOTEBOOK)
    print(f"patched {NOTEBOOK}")


if __name__ == "__main__":
    patch_notebook()
