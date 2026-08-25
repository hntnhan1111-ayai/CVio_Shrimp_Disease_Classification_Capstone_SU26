import ast
import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE_NB = (
    ROOT
    / "shrimp-leakage-aware-segmentation"
    / "notebooks"
    / "baseline"
    / "yolo11n_new_dataset_mixed_split_baseline_kaggle.ipynb"
)
FOURIER_SOURCE_NB = (
    ROOT
    / "shrimp-leakage-aware-segmentation"
    / "ONLY_fourier"
    / "training_log"
    / "11to20"
    / "015_fourier_attention_interaction"
    / "015_modeBCD_fouirer-hev-aug-attention.ipynb"
)
OUT_NB = (
    ROOT
    / "shrimp-leakage-aware-segmentation"
    / "notebooks"
    / "new_dataset_candidates"
    / "yolo11n_new_dataset_candidate_transfer_panel_kaggle.ipynb"
)


def source_of(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def clean_cell(cell, source=None):
    result = copy.deepcopy(cell)
    if source is not None:
        result["source"] = source
    result["metadata"] = {
        key: value
        for key, value in result.get("metadata", {}).items()
        if key not in {"execution", "trusted"}
    }
    if result.get("cell_type") == "code":
        result["outputs"] = []
        result["execution_count"] = None
    else:
        result.pop("outputs", None)
        result.pop("execution_count", None)
    return result


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


TITLE = """# YOLO11n New-Dataset Candidate Transfer Panel

Kaggle-ready candidate notebook for the expanded Roboflow dataset.

This notebook preserves the accepted new-data baseline contract and tests whether
the strongest old-dataset recipes transfer to the expanded dataset.

Defined modes:

| Mode | Enabled | Fourier | Augmentation | Hidden hook |
|---|---:|---|---|---|
| M0 | False | off | clean-light | on |
| M1 | True | off | strong | off |
| M2 | True | W1 high-pass | clean-light | on |
| M3 | True | W1 high-pass | strong | off |

W1 is fixed offline preprocessing applied to train, validation, and test images:
high-pass boost with `sigma=50` and `alpha=0.10`. Therefore the effective
training order for Fourier modes is `W1 -> online YOLO augmentation`.

The baseline mode remains available but is disabled because the accepted seed-42
baseline has already been trained. Its recorded metrics are included as a
reference, not inserted as a newly executed row.
"""


OUTPUTS = """## Outputs to Preserve

After the run, preserve:

- `reports/new_dataset_candidate_transfer_panel_seed42_summary.csv`
- `reports/new_dataset_candidate_transfer_panel_seed42_paper_row.csv`
- `reports/new_dataset_candidate_transfer_panel_seed42_partial.csv`
- `reports/train_args_new_dataset_candidate_transfer_panel_seed42.json`
- `reports/reference_new_dataset_baseline_seed42.json`
- `reports/split_manifests/`
- `reports/checkpoints/*_best.pt`
- `reports/results_csv/*_results.csv`
- `/kaggle/working/new_dataset_candidate_transfer_panel_seed42_reports.zip`

Candidate selection is based on validation healthy-aware score. Test metrics are
reported only after training and are not used to choose the checkpoint.
"""


RUN_CONTROLS = """# Run controls
from pathlib import Path

EXECUTION_PLAN = 'yolo11n_new_dataset_candidate_transfer_panel'
SMOKE_RUN = False
SEEDS = [42]
ULTRALYTICS_VERSION = '8.4.62'

YOLO_MODELS = ['yolo11n-seg.pt']
INCLUDE_M_MODELS = False
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

ROBOFLOW_WORKSPACE = 'lets-try-this'
ROBOFLOW_PROJECT = 'shrimpdisbd-tigershrimp_mrtudat'
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = 'yolo26'
NAMING_CONVENTION = '<disease>-<shrimpid>-img-<number>'

EXPECTED_SPLIT_FINGERPRINT = 'acd89cbdbf72795b5a0d3719550cb0df32e2312fbe56754e1b3a2b2c38552ec8'

REFERENCE_NEW_DATASET_BASELINE = {
    'mode': 'M0',
    'training_seed': 42,
    'augmentation_policy': 'clean_light_yolo_aug',
    'default_hook_enabled': True,
    'labeled_test_mask_map50': 0.5636756665266707,
    'labeled_test_mask_map50_95': 0.23313935238571667,
    'healthy_test_mask_fp_rate': 0.2682926829268293,
    'labeled_test_disease_mask_miss_rate': 0.09166666666666666,
    'healthy_aware_labeled_val_mask_map50': 0.5217857761776155,
    'healthy_aware_labeled_test_mask_map50': 0.502054731567321,
    'split_fingerprint': EXPECTED_SPLIT_FINGERPRINT,
}

print('EXECUTION_PLAN:', EXECUTION_PLAN)
print('SMOKE_RUN:', SMOKE_RUN)
print('SEEDS:', SEEDS)
print('YOLO_MODELS:', YOLO_MODELS)
print('Expected split fingerprint:', EXPECTED_SPLIT_FINGERPRINT)
print('Roboflow project:', f'{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/v{ROBOFLOW_VERSION}')
"""


MODE_CONFIG = """FOURIER_SIGMA = 50
FOURIER_ALPHA = 0.10
FOURIER_BATCH_SIZE = 4
USE_GPU_FOURIER = True
_FOURIER_LOWPASS_CACHE = {}

CANDIDATE_MODES = [
    {
        'mode': 'M0',
        'enabled': False,
        'key': 'M0_clean_light_no_fourier_hook_on',
        'name': 'M0 | clean-light | no Fourier | hook on',
        'fourier_enabled': False,
        'fourier_transform': 'none',
        'fourier_sigma': None,
        'fourier_alpha': None,
        'clean_light_aug_enabled': True,
        'strong_aug_enabled': False,
        'default_hook_enabled': True,
    },
    {
        'mode': 'M1',
        'enabled': True,
        'key': 'M1_strong_aug_no_fourier_hook_off',
        'name': 'M1 | strong augmentation | no Fourier | hook off',
        'fourier_enabled': False,
        'fourier_transform': 'none',
        'fourier_sigma': None,
        'fourier_alpha': None,
        'clean_light_aug_enabled': False,
        'strong_aug_enabled': True,
        'default_hook_enabled': False,
    },
    {
        'mode': 'M2',
        'enabled': True,
        'key': 'M2_clean_light_W1_hook_on',
        'name': 'M2 | clean-light | W1 high-pass | hook on',
        'fourier_enabled': True,
        'fourier_transform': 'highpass_boost',
        'fourier_sigma': 50,
        'fourier_alpha': 0.10,
        'clean_light_aug_enabled': True,
        'strong_aug_enabled': False,
        'default_hook_enabled': True,
    },
    {
        'mode': 'M3',
        'enabled': True,
        'key': 'M3_strong_aug_W1_hook_off',
        'name': 'M3 | strong augmentation | W1 high-pass | hook off',
        'fourier_enabled': True,
        'fourier_transform': 'highpass_boost',
        'fourier_sigma': 50,
        'fourier_alpha': 0.10,
        'clean_light_aug_enabled': False,
        'strong_aug_enabled': True,
        'default_hook_enabled': False,
    },
]
EXPERIMENTS = [mode for mode in CANDIDATE_MODES if mode.get('enabled', True)]

CLEAN_LIGHT_AUG_TRAIN_ARGS = {
    'auto_augment': None,
    'erasing': 0.0,
    'mosaic': 0.0,
    'mixup': 0.0,
    'cutmix': 0.0,
    'copy_paste': 0.0,
    'fliplr': 0.5,
    'flipud': 0.0,
    'hsv_h': 0.01,
    'hsv_s': 0.35,
    'hsv_v': 0.20,
    'degrees': 0.0,
    'translate': 0.05,
    'scale': 0.20,
    'shear': 0.0,
    'perspective': 0.0,
    'multi_scale': 0.0,
    'bgr': 0.0,
}

STRONG_AUG_TRAIN_ARGS = {
    'auto_augment': None,
    'erasing': 0.15,
    'mosaic': 0.0,
    'mixup': 0.0,
    'cutmix': 0.0,
    'copy_paste': 0.0,
    'fliplr': 0.5,
    'flipud': 0.3,
    'hsv_h': 0.05,
    'hsv_s': 0.50,
    'hsv_v': 0.40,
    'degrees': 10.0,
    'translate': 0.10,
    'scale': 0.50,
    'shear': 0.0,
    'perspective': 0.0,
    'multi_scale': 0.0,
    'bgr': 0.0,
}


def train_args_for_mode(exp):
    if exp.get('strong_aug_enabled'):
        return dict(STRONG_AUG_TRAIN_ARGS)
    if exp.get('clean_light_aug_enabled'):
        return dict(CLEAN_LIGHT_AUG_TRAIN_ARGS)
    raise ValueError(f"Mode {exp.get('mode')} has no declared augmentation policy.")


def augmentation_policy_name(exp):
    if exp.get('strong_aug_enabled'):
        return 'strong_yolo_aug_path15'
    if exp.get('clean_light_aug_enabled'):
        return 'clean_light_yolo_aug'
    return 'undefined'
"""


RUN_EXPERIMENT = """def run_experiment(exp):
    print('\\n' + '#' * 90)
    print(f"Starting experiment: {exp['name']}")
    print('#' * 90)

    dataset_dir = copy_dataset_for_experiment(exp['key'])
    if exp.get('fourier_enabled'):
        fourier_transform_summary = apply_fourier_transform_to_dataset(dataset_dir, exp)
    else:
        fourier_transform_summary = {
            'strategy': 'fourier_disabled',
            'fourier_enabled': False,
            'transform': 'none',
            'sigma': None,
            'alpha': None,
        }
        print('Fourier disabled for this mode; copied dataset images remain original.')

    remove_yolo_label_caches(dataset_dir)
    yaml_path = dataset_dir / 'data.yaml'
    write_data_yaml(dataset_dir, yaml_path)
    labeled_eval_dir, labeled_eval_yaml, _ = make_labeled_only_eval_dataset(dataset_dir, exp['key'])
    healthy_eval_dir, healthy_eval_yaml, _ = make_healthy_only_eval_dataset(dataset_dir, exp['key'])

    split_counts = {}
    actual_image_files = {}
    for split in ['train', 'valid', 'test']:
        split_counts[split] = count_labeled_images(dataset_dir / split / 'labels')
        image_dir = dataset_dir / split / 'images'
        actual_image_files[split] = sum(
            1
            for ext in IMAGE_EXTENSIONS
            for image_path in image_dir.glob(f'*{ext}')
            if image_path.is_file()
        )
        print(
            f'{exp["key"]} {split}: {split_counts[split]} | '
            f'actual_image_files={actual_image_files[split]}'
        )

    training_seed = int(globals().get('CURRENT_RUN_SEED', 42))
    run_name = (
        f'{RUN_BASE_NAME}_seed{training_seed}_{exp["key"]}'
        + ('_smoke' if SMOKE_RUN else '')
    )
    hook_disabled = not bool(exp.get('default_hook_enabled'))
    configure_ultralytics_albumentations_hook(disable=hook_disabled)

    train_args = train_args_for_mode(exp)
    yolo = YOLO(YOLO_MODEL)
    start = time.time()
    yolo.train(
        data=str(yaml_path),
        task='segment',
        imgsz=TRAIN_IMGSZ,
        epochs=TRAIN_EPOCHS,
        batch=TRAIN_BATCH,
        patience=TRAIN_PATIENCE,
        seed=training_seed,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        plots=not SMOKE_RUN,
        verbose=True,
        **train_args,
    )
    train_time_min = (time.time() - start) / 60

    run_path = RUNS_DIR / run_name
    best_path = run_path / 'weights' / 'best.pt'
    if not best_path.exists():
        raise FileNotFoundError(f'Missing trained checkpoint: {best_path}')
    best_model = YOLO(str(best_path))

    full_val = best_model.val(
        data=str(yaml_path),
        split='val',
        imgsz=TRAIN_IMGSZ,
        plots=not SMOKE_RUN,
        verbose=False,
    )
    labeled_val = best_model.val(
        data=str(labeled_eval_yaml),
        split='val',
        imgsz=TRAIN_IMGSZ,
        plots=False,
        verbose=False,
    )
    labeled_val_count = count_prediction_errors(
        best_model,
        labeled_eval_dir / 'valid' / 'images',
        labeled_eval_dir / 'valid' / 'labels',
    )
    healthy_val_fp = healthy_false_positive_summary(
        best_model,
        healthy_eval_dir / 'valid' / 'images',
    )
    labeled_val_map50 = metric_value(labeled_val, 'seg.map50')
    val_score = healthy_aware_score(
        labeled_val_map50,
        labeled_val_count,
        healthy_val_fp,
    )

    row = {
        'experiment': exp['key'],
        'mode': exp.get('mode'),
        'name': exp['name'],
        'model': YOLO_MODEL,
        'run_name': run_name,
        'run_path': str(run_path),
        'best_pt': str(best_path),
        'smoke_run': SMOKE_RUN,
        'training_seed': training_seed,
        'fourier_enabled': bool(exp.get('fourier_enabled')),
        'strong_aug_enabled': bool(exp.get('strong_aug_enabled')),
        'clean_light_aug_enabled': bool(exp.get('clean_light_aug_enabled')),
        'default_hook_enabled': bool(exp.get('default_hook_enabled')),
        'baseline_augmentation_policy': augmentation_policy_name(exp),
        'hidden_albumentations_disabled': hook_disabled,
        'preprocessing_policy': 'offline_fourier_highpass_all_splits'
        if exp.get('fourier_enabled')
        else 'none',
        'fourier_train_order': 'offline_W1_then_online_YOLO_augmentation'
        if exp.get('fourier_enabled')
        else 'not_applicable',
        'fourier_policy': fourier_policy_name(exp),
        'fourier_transform': exp.get('fourier_transform', 'none'),
        'fourier_sigma': exp.get('fourier_sigma')
        if exp.get('fourier_enabled')
        else None,
        'fourier_alpha': exp.get('fourier_alpha')
        if exp.get('fourier_enabled')
        else None,
        'ultralytics_version': ultralytics.__version__,
        'train_epochs_requested': TRAIN_EPOCHS,
        'train_patience': TRAIN_PATIENCE,
        'train_batch': TRAIN_BATCH,
        'train_imgsz': TRAIN_IMGSZ,
        'actual_train_image_files': actual_image_files.get('train'),
        'actual_valid_image_files': actual_image_files.get('valid'),
        'actual_test_image_files': actual_image_files.get('test'),
        'fourier_transform_summary': json.dumps(fourier_transform_summary),
        'train_args': json.dumps(train_args, sort_keys=True),
        'train_time_min': round(train_time_min, 2),
        'full_val_box_map50': metric_value(full_val, 'box.map50'),
        'full_val_mask_map50': metric_value(full_val, 'seg.map50'),
        'labeled_val_box_map50': metric_value(labeled_val, 'box.map50'),
        'labeled_val_mask_map50': labeled_val_map50,
        'labeled_val_mask_map50_95': metric_value(labeled_val, 'seg.map'),
        'labeled_val_gt_instances': labeled_val_count['gt_total'],
        'labeled_val_pred_boxes': labeled_val_count['pred_box_total'],
        'labeled_val_pred_masks': labeled_val_count['pred_mask_total'],
        'labeled_val_mask_count_mae': labeled_val_count['mask_count_mae'],
        'labeled_val_disease_box_miss_rate': labeled_val_count['disease_box_miss_rate'],
        'labeled_val_disease_mask_miss_rate': labeled_val_count['disease_mask_miss_rate'],
        'healthy_val_images': healthy_val_fp['healthy_images'],
        'healthy_val_mask_fp_rate': healthy_val_fp['healthy_mask_fp_rate'],
        'healthy_val_fp_masks_per_image': healthy_val_fp['healthy_fp_masks_per_image'],
        'healthy_aware_labeled_val_mask_map50': val_score,
    }

    if RUN_TEST_EVALUATION:
        full_test = best_model.val(
            data=str(yaml_path),
            split='test',
            imgsz=TRAIN_IMGSZ,
            plots=True,
            verbose=False,
        )
        labeled_test = best_model.val(
            data=str(labeled_eval_yaml),
            split='test',
            imgsz=TRAIN_IMGSZ,
            plots=False,
            verbose=False,
        )
        test_count = count_prediction_errors(
            best_model,
            dataset_dir / 'test' / 'images',
            dataset_dir / 'test' / 'labels',
        )
        labeled_test_count = count_prediction_errors(
            best_model,
            labeled_eval_dir / 'test' / 'images',
            labeled_eval_dir / 'test' / 'labels',
        )
        healthy_test_fp = healthy_false_positive_summary(
            best_model,
            healthy_eval_dir / 'test' / 'images',
        )
        labeled_test_map50 = metric_value(labeled_test, 'seg.map50')
        row.update(
            {
                'full_test_box_map50': metric_value(full_test, 'box.map50'),
                'full_test_box_map50_95': metric_value(full_test, 'box.map'),
                'full_test_mask_map50': metric_value(full_test, 'seg.map50'),
                'full_test_mask_map50_95': metric_value(full_test, 'seg.map'),
                'labeled_test_box_map50': metric_value(labeled_test, 'box.map50'),
                'labeled_test_mask_map50': labeled_test_map50,
                'labeled_test_mask_map50_95': metric_value(labeled_test, 'seg.map'),
                'test_gt_instances': test_count['gt_total'],
                'test_pred_boxes': test_count['pred_box_total'],
                'test_pred_masks': test_count['pred_mask_total'],
                'test_mask_count_mae': test_count['mask_count_mae'],
                'labeled_test_gt_instances': labeled_test_count['gt_total'],
                'labeled_test_pred_boxes': labeled_test_count['pred_box_total'],
                'labeled_test_pred_masks': labeled_test_count['pred_mask_total'],
                'labeled_test_mask_count_mae': labeled_test_count['mask_count_mae'],
                'labeled_test_mask_count_exact': labeled_test_count['mask_count_exact'],
                'labeled_test_disease_box_miss_rate': labeled_test_count[
                    'disease_box_miss_rate'
                ],
                'labeled_test_disease_mask_miss_rate': labeled_test_count[
                    'disease_mask_miss_rate'
                ],
                'healthy_test_images': healthy_test_fp['healthy_images'],
                'healthy_test_mask_fp_rate': healthy_test_fp['healthy_mask_fp_rate'],
                'healthy_test_box_fp_rate': healthy_test_fp['healthy_box_fp_rate'],
                'healthy_test_fp_masks_total': healthy_test_fp[
                    'healthy_fp_masks_total'
                ],
                'healthy_test_fp_masks_per_image': healthy_test_fp[
                    'healthy_fp_masks_per_image'
                ],
                'healthy_test_avg_fp_confidence': healthy_test_fp[
                    'healthy_avg_fp_confidence'
                ],
                'healthy_aware_labeled_test_mask_map50': healthy_aware_score(
                    labeled_test_map50,
                    labeled_test_count,
                    healthy_test_fp,
                ),
            }
        )

    row.update(read_best_epoch_from_results(run_path))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_copy = CHECKPOINT_DIR / f"{exp['mode']}_{exp['key']}_best.pt"
    shutil.copy2(best_path, checkpoint_copy)
    row['preserved_best_pt'] = str(checkpoint_copy)

    source_results_csv = run_path / 'results.csv'
    if source_results_csv.exists():
        results_copy = RESULTS_CSV_DIR / f"{exp['mode']}_{exp['key']}_results.csv"
        shutil.copy2(source_results_csv, results_copy)
        row['preserved_results_csv'] = str(results_copy)

    del yolo, best_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


train_args_json = REPORT_DIR / 'train_args_new_dataset_candidate_transfer_panel_seed42.json'
train_args_payload = {
    'candidate_modes': CANDIDATE_MODES,
    'enabled_modes': [mode['mode'] for mode in EXPERIMENTS],
    'clean_light_aug_train_args': CLEAN_LIGHT_AUG_TRAIN_ARGS,
    'strong_aug_train_args': STRONG_AUG_TRAIN_ARGS,
    'fourier_sigma': FOURIER_SIGMA,
    'fourier_alpha': FOURIER_ALPHA,
    'fourier_batch_size': FOURIER_BATCH_SIZE,
    'expected_split_fingerprint': EXPECTED_SPLIT_FINGERPRINT,
    'train_epochs_requested': TRAIN_EPOCHS,
    'train_patience': TRAIN_PATIENCE,
    'train_batch': TRAIN_BATCH,
    'train_imgsz': TRAIN_IMGSZ,
}
train_args_json.write_text(
    json.dumps(train_args_payload, indent=2),
    encoding='utf-8',
)
print('Saved candidate transfer train args:', train_args_json)
"""


RUN_CELL = """from collections import defaultdict
import gc
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from IPython.display import display

EXPERIMENT_ROOT = Path('/kaggle/working/shrimp_yolo11n_new_dataset_candidate_transfer_panel')
RUNS_DIR = Path('/kaggle/working/runs/segment')
REPORT_DIR = EXPERIMENT_ROOT / 'reports'
CHECKPOINT_DIR = REPORT_DIR / 'checkpoints'
RESULTS_CSV_DIR = REPORT_DIR / 'results_csv'
for output_dir in [REPORT_DIR, CHECKPOINT_DIR, RESULTS_CSV_DIR]:
    output_dir.mkdir(parents=True, exist_ok=True)

TRAIN_IMGSZ = 320 if SMOKE_RUN else 640
TRAIN_EPOCHS = 1 if SMOKE_RUN else 100
TRAIN_BATCH = 8 if SMOKE_RUN else 16
TRAIN_PATIENCE = 1 if SMOKE_RUN else 30
RUN_TEST_EVALUATION = not SMOKE_RUN
PREDICT_CONF_FOR_COUNT = 0.25


def split_records():
    records = []
    for split_name in ['train', 'valid', 'test']:
        for image_path in image_files_in_split(split_name):
            group_key, disease, shrimp_id, img_num = parse_shrimp_group_key(
                image_path.name
            )
            label_path = (
                base_path
                / split_name
                / 'labels'
                / f'{image_path.stem}.txt'
            )
            label_lines = (
                [
                    line.strip()
                    for line in label_path.read_text(encoding='utf-8').splitlines()
                    if line.strip()
                ]
                if label_path.exists()
                else []
            )
            class_ids = sorted(
                {
                    int(float(line.split()[0]))
                    for line in label_lines
                    if line.split()
                }
            )
            meta = SOURCE_META.get(image_path.name, {})
            records.append(
                {
                    'split': split_name,
                    'image': image_path.name,
                    'source_subset': meta.get('source_subset', 'unknown'),
                    'naming_convention_matched': meta.get(
                        'naming_convention_matched',
                        group_key is not None,
                    ),
                    'group_key': group_key or f'unparsed::{image_path.stem}',
                    'disease': str(disease).lower(),
                    'label_stratum': disease_stratum_for_image(image_path),
                    'shrimp_id': shrimp_id,
                    'mask_instances': len(label_lines),
                    'is_labeled': bool(label_lines),
                    'class_ids': ','.join(str(cid) for cid in class_ids),
                    'has_bg_mask': int(0 in class_ids),
                    'has_wssv_mask': int(1 in class_ids),
                    'is_coinfection': int(len(class_ids) >= 2),
                }
            )
    return pd.DataFrame(records)


def save_split_artifacts(seed=42):
    df = split_records().sort_values(['split', 'image']).reset_index(drop=True)
    manifest_dir = REPORT_DIR / 'split_manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'merged_mixed_split_seed{seed}'
    manifest_csv = manifest_dir / f'{prefix}_manifest.csv'
    summary_csv = manifest_dir / f'{prefix}_summary.csv'
    fingerprint_txt = manifest_dir / f'{prefix}_fingerprint.txt'
    df.to_csv(manifest_csv, index=False)

    summary_rows = []
    split_hashes = []
    for split in ['train', 'valid', 'test']:
        part = df[df['split'] == split]
        split_hash = hashlib.sha256(
            '\\n'.join(part['image'].tolist()).encode()
        ).hexdigest()
        split_hashes.append(split_hash)
        summary_rows.append(
            {
                'split': split,
                'images': len(part),
                'specimens_or_unique_groups': part['group_key'].nunique(),
                'convention_grouped_images': int(
                    (part['source_subset'] == 'convention_grouped_stratified').sum()
                ),
                'unmatched_random_images': int(
                    (part['source_subset'] == 'unmatched_stratified_random').sum()
                ),
                'labeled_images': int(part['is_labeled'].sum()),
                'healthy_empty_images': int((~part['is_labeled']).sum()),
                'bg_mask_images': int(part['has_bg_mask'].sum()),
                'wssv_mask_images': int(part['has_wssv_mask'].sum()),
                'coinfection_images': int(part['is_coinfection'].sum()),
                'mask_instances': int(part['mask_instances'].sum()),
                'split_sha256': split_hash,
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    fingerprint = hashlib.sha256(
        '||'.join(split_hashes).encode()
    ).hexdigest()
    fingerprint_txt.write_text(fingerprint + '\\n', encoding='utf-8')

    parsed = df[df['naming_convention_matched']]
    parsed_sets = {
        split: set(parsed.loc[parsed['split'] == split, 'group_key'])
        for split in ['train', 'valid', 'test']
    }
    overlaps = {
        f'{left}_{right}': len(parsed_sets[left] & parsed_sets[right])
        for left, right in [
            ('train', 'valid'),
            ('train', 'test'),
            ('valid', 'test'),
        ]
    }
    if any(overlaps.values()):
        raise RuntimeError(
            f'Convention-matched specimen leakage detected: {overlaps}'
        )

    print('Saved mixed split manifest:', manifest_csv)
    print('Saved mixed split summary:', summary_csv)
    print('Split fingerprint:', fingerprint)
    print('Convention-matched group overlap:', overlaps)
    return {
        'split_manifest_csv': str(manifest_csv),
        'split_summary_csv': str(summary_csv),
        'split_fingerprint': fingerprint,
    }


def split_metadata(model_name, seed):
    df = split_records()
    row = {
        'split_policy': 'merged_convention_grouped_plus_unmatched_stratified_random',
        'model': model_name,
        'seed': seed,
        'train_images': int((df['split'] == 'train').sum()),
        'valid_images': int((df['split'] == 'valid').sum()),
        'test_images': int((df['split'] == 'test').sum()),
        'train_specimens_or_unique_groups': int(
            df[df['split'] == 'train']['group_key'].nunique()
        ),
        'valid_specimens_or_unique_groups': int(
            df[df['split'] == 'valid']['group_key'].nunique()
        ),
        'test_specimens_or_unique_groups': int(
            df[df['split'] == 'test']['group_key'].nunique()
        ),
    }
    for split in ['train', 'valid', 'test']:
        part = df[df['split'] == split]
        row[f'{split}_convention_grouped_images'] = int(
            (part['source_subset'] == 'convention_grouped_stratified').sum()
        )
        row[f'{split}_unmatched_random_images'] = int(
            (part['source_subset'] == 'unmatched_stratified_random').sum()
        )
        row[f'{split}_labeled_images'] = int(part['is_labeled'].sum())
        row[f'{split}_healthy_empty_images'] = int((~part['is_labeled']).sum())
        row[f'{split}_bg_mask_images'] = int(part['has_bg_mask'].sum())
        row[f'{split}_wssv_mask_images'] = int(part['has_wssv_mask'].sum())
        row[f'{split}_coinfection_images'] = int(part['is_coinfection'].sum())
        row[f'{split}_mask_instances'] = int(part['mask_instances'].sum())
    return row


def count_model_params_million(checkpoint_name):
    yolo_for_count = YOLO(checkpoint_name)
    try:
        params = sum(parameter.numel() for parameter in yolo_for_count.model.parameters())
        return round(params / 1_000_000, 3)
    finally:
        del yolo_for_count
        gc.collect()


print('Defined candidate modes:', [mode['mode'] for mode in CANDIDATE_MODES])
print('Enabled candidate modes:', [mode['mode'] for mode in EXPERIMENTS])
assert not next(mode for mode in CANDIDATE_MODES if mode['mode'] == 'M0')['enabled']
assert [mode['mode'] for mode in EXPERIMENTS] == ['M1', 'M2', 'M3']

reference_json = REPORT_DIR / 'reference_new_dataset_baseline_seed42.json'
reference_json.write_text(
    json.dumps(REFERENCE_NEW_DATASET_BASELINE, indent=2),
    encoding='utf-8',
)

paper_rows = []
for seed in SEEDS:
    CURRENT_RUN_SEED = seed
    split_artifacts = save_split_artifacts(seed)
    actual_fingerprint = split_artifacts['split_fingerprint']
    if actual_fingerprint != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(
            'Regenerated split fingerprint does not match the accepted baseline. '
            f'Expected {EXPECTED_SPLIT_FINGERPRINT}, got {actual_fingerprint}. '
            'Stop before training and audit the downloaded dataset/version.'
        )
    print('Split fingerprint matches the accepted new-data baseline.')

    for model_name in YOLO_MODELS:
        YOLO_MODEL = model_name
        params_million = count_model_params_million(YOLO_MODEL)
        for experiment in EXPERIMENTS:
            exp = dict(experiment)
            exp['key'] = (
                f'merged_mixed_split_{Path(YOLO_MODEL).stem}_{experiment["key"]}'
            )
            exp['name'] = (
                f'Merged mixed split | {Path(YOLO_MODEL).stem} | '
                f'{experiment["name"]}'
            )
            meta = split_metadata(YOLO_MODEL, seed)
            meta.update(split_artifacts)
            meta['params_million'] = params_million

            result = run_experiment(exp)
            result.update(meta)
            paper_rows.append(result)

            partial_df = pd.DataFrame(paper_rows)
            partial_path = (
                REPORT_DIR
                / 'new_dataset_candidate_transfer_panel_seed42_partial.csv'
            )
            partial_df.to_csv(partial_path, index=False)
            print('Saved recovery CSV:', partial_path)
            display(partial_df)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

summary_df = pd.DataFrame(paper_rows)
summary_csv = REPORT_DIR / 'new_dataset_candidate_transfer_panel_seed42_summary.csv'
summary_df.to_csv(summary_csv, index=False)
print('Saved candidate transfer summary:', summary_csv)

print('Candidate ranking by validation healthy-aware score:')
display(
    summary_df.sort_values(
        'healthy_aware_labeled_val_mask_map50',
        ascending=False,
    )
)
"""


PAPER_ROW = """paper_cols = [
    'split_policy',
    'mode',
    'model',
    'seed',
    'training_seed',
    'params_million',
    'ultralytics_version',
    'fourier_enabled',
    'fourier_policy',
    'fourier_transform',
    'fourier_sigma',
    'fourier_alpha',
    'fourier_train_order',
    'strong_aug_enabled',
    'clean_light_aug_enabled',
    'default_hook_enabled',
    'baseline_augmentation_policy',
    'hidden_albumentations_disabled',
    'train_epochs_requested',
    'train_patience',
    'train_batch',
    'train_imgsz',
    'train_images',
    'valid_images',
    'test_images',
    'train_labeled_images',
    'valid_labeled_images',
    'test_labeled_images',
    'train_healthy_empty_images',
    'valid_healthy_empty_images',
    'test_healthy_empty_images',
    'train_convention_grouped_images',
    'valid_convention_grouped_images',
    'test_convention_grouped_images',
    'train_unmatched_random_images',
    'valid_unmatched_random_images',
    'test_unmatched_random_images',
    'split_manifest_csv',
    'split_fingerprint',
    'epochs_ran',
    'best_epoch_by_mask_map50',
    'train_time_min',
    'full_test_mask_map50',
    'full_test_mask_map50_95',
    'labeled_test_mask_map50',
    'labeled_test_mask_map50_95',
    'healthy_test_mask_fp_rate',
    'healthy_test_fp_masks_per_image',
    'labeled_test_disease_mask_miss_rate',
    'labeled_test_mask_count_mae',
    'healthy_aware_labeled_val_mask_map50',
    'healthy_aware_labeled_test_mask_map50',
    'preserved_best_pt',
    'preserved_results_csv',
    'run_path',
]

available_paper_cols = [column for column in paper_cols if column in summary_df.columns]
paper_table = summary_df[available_paper_cols].copy()

test_delta_specs = [
    (
        'delta_labeled_test_map50_vs_reference',
        'labeled_test_mask_map50',
        'labeled_test_mask_map50',
    ),
    (
        'delta_healthy_fp_rate_vs_reference',
        'healthy_test_mask_fp_rate',
        'healthy_test_mask_fp_rate',
    ),
    (
        'delta_healthy_aware_test_vs_reference',
        'healthy_aware_labeled_test_mask_map50',
        'healthy_aware_labeled_test_mask_map50',
    ),
]
for output_column, result_column, reference_key in test_delta_specs:
    if result_column in paper_table.columns:
        paper_table[output_column] = (
            paper_table[result_column]
            - REFERENCE_NEW_DATASET_BASELINE[reference_key]
        )
    else:
        # Smoke runs intentionally skip test evaluation.
        paper_table[output_column] = float('nan')

paper_table_csv = (
    REPORT_DIR / 'new_dataset_candidate_transfer_panel_seed42_paper_row.csv'
)
paper_table.to_csv(paper_table_csv, index=False)
print('Saved compact candidate transfer paper row:', paper_table_csv)
display(paper_table)

print('Selection view: validation healthy-aware score first; test deltas are report-only.')
selection_columns = [
    'mode',
    'fourier_enabled',
    'baseline_augmentation_policy',
    'default_hook_enabled',
    'healthy_aware_labeled_val_mask_map50',
    'labeled_test_mask_map50',
    'healthy_test_mask_fp_rate',
    'labeled_test_disease_mask_miss_rate',
    'healthy_aware_labeled_test_mask_map50',
    'delta_labeled_test_map50_vs_reference',
    'delta_healthy_fp_rate_vs_reference',
]
selection_columns = [
    column for column in selection_columns if column in paper_table.columns
]
display(
    paper_table.sort_values(
        'healthy_aware_labeled_val_mask_map50',
        ascending=False,
    )[selection_columns]
)
"""


FINAL_DOWNLOAD = """from pathlib import Path
import shutil

print('Kaggle output checklist')
required_globals = ['REPORT_DIR', 'summary_df', 'paper_table_csv']
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    print('Training/report cells did not complete. Missing:', missing_globals)
else:
    report_dir = Path(REPORT_DIR)
    expected_outputs = [
        report_dir / 'new_dataset_candidate_transfer_panel_seed42_summary.csv',
        report_dir / 'new_dataset_candidate_transfer_panel_seed42_paper_row.csv',
        report_dir / 'new_dataset_candidate_transfer_panel_seed42_partial.csv',
        report_dir / 'train_args_new_dataset_candidate_transfer_panel_seed42.json',
        report_dir / 'reference_new_dataset_baseline_seed42.json',
        report_dir / 'split_manifests',
        report_dir / 'checkpoints',
        report_dir / 'results_csv',
    ]
    for index, output_path in enumerate(expected_outputs, start=1):
        status = 'OK' if output_path.exists() else 'MISSING'
        print(f'{index}. [{status}] {output_path}')

    archive_base = Path('/kaggle/working/new_dataset_candidate_transfer_panel_seed42_reports')
    archive_path = shutil.make_archive(
        str(archive_base),
        'zip',
        root_dir=str(report_dir),
    )
    print('Reports/checkpoints archive:', archive_path)
    print('Download the zip from the Kaggle Output pane.')
"""


def build_training_cell(baseline_source, fourier_source):
    source = baseline_source
    source = source.replace(
        "import csv\nimport gc\nimport math\nimport shutil\nimport time",
        "import csv\nimport gc\nimport json\nimport math\nimport shutil\nimport time",
    )
    source = source.replace(
        "import cv2\nimport pandas as pd\nimport yaml",
        "import cv2\nimport numpy as np\nimport pandas as pd\nimport torch\nimport ultralytics\nimport yaml",
    )
    source = source.replace(
        "EXPERIMENT_ROOT = Path('/kaggle/working/shrimp_yolo11n_new_dataset_mixed_split_baseline')",
        "EXPERIMENT_ROOT = Path('/kaggle/working/shrimp_yolo11n_new_dataset_candidate_transfer_panel')",
    )
    source = source.replace(
        "REPORT_DIR = EXPERIMENT_ROOT / 'reports'\nREPORT_DIR.mkdir(parents=True, exist_ok=True)",
        "REPORT_DIR = EXPERIMENT_ROOT / 'reports'\n"
        "CHECKPOINT_DIR = REPORT_DIR / 'checkpoints'\n"
        "RESULTS_CSV_DIR = REPORT_DIR / 'results_csv'\n"
        "for output_dir in [REPORT_DIR, CHECKPOINT_DIR, RESULTS_CSV_DIR]:\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)",
    )

    source, count = re.subn(
        r"EXPERIMENTS = \[.*?\n\n\ndef configure_ultralytics_albumentations_hook",
        MODE_CONFIG + "\n\n\ndef configure_ultralytics_albumentations_hook",
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Could not replace baseline experiment configuration.")

    fourier_start = fourier_source.index("def fourier_device():")
    fourier_end = fourier_source.index(
        "def configure_ultralytics_albumentations_hook"
    )
    fourier_helpers = fourier_source[fourier_start:fourier_end].rstrip()
    source = source.replace(
        "def count_labeled_images(label_dir):",
        fourier_helpers + "\n\n\ndef count_labeled_images(label_dir):",
        1,
    )

    run_start = source.index("def run_experiment(exp):")
    source = source[:run_start] + RUN_EXPERIMENT
    return source


def build_dataset_download(source):
    source = source.replace(
        "ROBOFLOW_API_KEY_DIRECT = ''  # Keep empty. Use a Kaggle Secret named ROBOFLOW_API_KEY.",
        "ROBOFLOW_API_KEY_DIRECT = 'KOEk0qLzBFDc7zfyxtgs'",
    )
    source = source.replace(
        "raise RuntimeError('Missing ROBOFLOW_API_KEY. Add it to Kaggle Secrets; do not paste it into the notebook.')",
        "raise RuntimeError('Missing ROBOFLOW_API_KEY.')",
    )
    return source


def validate_notebook(notebook):
    code_cells = [
        source_of(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code_cells):
        ast.parse(source, filename=f"code_cell_{index}")

    combined = "\n".join(code_cells)
    required_tokens = [
        "EXPECTED_SPLIT_FINGERPRINT",
        "actual_fingerprint != EXPECTED_SPLIT_FINGERPRINT",
        "'mode': 'M0'",
        "'enabled': False",
        "'mode': 'M1'",
        "'mode': 'M2'",
        "'mode': 'M3'",
        "apply_fourier_transform_to_dataset",
        "STRONG_AUG_TRAIN_ARGS",
        "CLEAN_LIGHT_AUG_TRAIN_ARGS",
        "TRAIN_BATCH = 8 if SMOKE_RUN else 16",
        "TRAIN_PATIENCE = 1 if SMOKE_RUN else 30",
        "new_dataset_candidate_transfer_panel_seed42_partial.csv",
        "ROBOFLOW_API_KEY_DIRECT = 'KOEk0qLzBFDc7zfyxtgs'",
    ]
    missing = [token for token in required_tokens if token not in combined]
    if missing:
        raise RuntimeError(f"Generated notebook is missing required tokens: {missing}")


def main():
    baseline = json.loads(SOURCE_NB.read_text(encoding="utf-8"))
    fourier = json.loads(FOURIER_SOURCE_NB.read_text(encoding="utf-8"))
    baseline_cells = baseline["cells"]
    fourier_helper_source = source_of(fourier["cells"][18])

    training_source = build_training_cell(
        source_of(baseline_cells[19]),
        fourier_helper_source,
    )
    dataset_download_source = build_dataset_download(source_of(baseline_cells[5]))

    new_cells = [
        markdown(TITLE),
        markdown(OUTPUTS),
        markdown("## Run Controls"),
        code(RUN_CONTROLS),
        clean_cell(baseline_cells[4]),
        clean_cell(baseline_cells[5], dataset_download_source),
        clean_cell(baseline_cells[6]),
        clean_cell(baseline_cells[7]),
        clean_cell(baseline_cells[8]),
        clean_cell(baseline_cells[9]),
        clean_cell(baseline_cells[10]),
        clean_cell(baseline_cells[11]),
        clean_cell(baseline_cells[12]),
        clean_cell(baseline_cells[13]),
        clean_cell(baseline_cells[14]),
        clean_cell(baseline_cells[15]),
        clean_cell(baseline_cells[16]),
        markdown(
            "## Training and Evaluation Helpers\n\n"
            "W1 is computed offline on copied datasets. The original downloaded "
            "dataset and regenerated split remain unchanged."
        ),
        code(training_source),
        markdown("## Candidate Transfer Runs"),
        code(RUN_CELL),
        markdown("## Compact Paper Row"),
        code(PAPER_ROW),
        markdown("## Final Download Checklist"),
        code(FINAL_DOWNLOAD),
    ]

    output = {
        "cells": new_cells,
        "metadata": baseline.get("metadata", {}),
        "nbformat": baseline.get("nbformat", 4),
        "nbformat_minor": baseline.get("nbformat_minor", 5),
    }
    validate_notebook(output)
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(
        json.dumps(output, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(OUT_NB)


if __name__ == "__main__":
    main()
