import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE_NB = ROOT / "shrimp-leakage-aware-segmentation" / "ONLY_fourier" / "training_log" / "015_modeBCD_fouirer-hev-aug-attention.ipynb"
OUT_DIR = ROOT / "shrimp-leakage-aware-segmentation" / "ONLY_fourier" / "016_fourier_strongaug_hook_ablation"
OUT_NB = OUT_DIR / "016_fourier_strongaug_hook_ablation_seed42.ipynb"


def source_of(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def clean_cell(cell, source=None):
    new_cell = copy.deepcopy(cell)
    if source is not None:
        new_cell["source"] = source
    new_cell["metadata"] = {k: v for k, v in new_cell.get("metadata", {}).items() if k not in {"execution", "trusted"}}
    new_cell["outputs"] = []
    if new_cell.get("cell_type") == "code":
        new_cell["execution_count"] = None
    return new_cell


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def replace_block(text, start_pattern, end_pattern, replacement, flags=re.DOTALL):
    pattern = start_pattern + r".*?" + end_pattern
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected exactly one replacement for {start_pattern!r} ... {end_pattern!r}, got {count}.")
    return new_text


title_md = """# ONLY Fourier Path 16 - Fourier x Teammate Strong Aug x Default Hook Ablation Seed 42

This notebook runs the missing rows for the 8-mode ablation:

- Fourier preprocessing: off/on
- teammate strong YOLO augmentation: off/on
- default Ultralytics Albumentations hook: off/on

Attention is intentionally excluded. All modes use the baseline `yolo11n-seg.pt` architecture.

Fixed contract:

- stratified grouped-specimen split
- seed 42
- base model: `yolo11n-seg.pt`
- epochs = 100
- patience = 30
- best.pt evaluation
- Fourier-on setting: high-pass boost, `sigma=50`, `alpha=0.10`

All eight modes are defined in one list for traceability, but only the two missing hook-on + strong-aug rows are enabled by default:

- M7: Fourier off, teammate strong aug on, default hook on
- M8: Fourier on, teammate strong aug on, default hook on
"""


outputs_md = """## Outputs To Download

After the run, download these artifacts from Kaggle/Colab:

- `only_fourier_strongaug_hook_ablation_seed42_summary.csv`
- `only_fourier_strongaug_hook_ablation_seed42_paper_row.csv`
- `only_fourier_strongaug_hook_ablation_seed42_partial.csv` if interrupted
- `train_args_fourier_strongaug_hook_ablation_seed42.json`
- `split_manifests/`
- selected `best.pt` checkpoints under `runs/segment/`
- `only_fourier_strongaug_hook_ablation_seed42_outputs.zip`

Rank rows first by labeled test mask mAP50. Use full-test metrics and healthy diagnostics as secondary interpretation columns.
"""


run_controls = """# Run controls for ONLY_fourier path 16.
from pathlib import Path

EXECUTION_PLAN = 'only_fourier_16_fourier_strongaug_hook_ablation_seed42'
SMOKE_RUN = False
SEEDS = [42]
DISABLE_ULTRALYTICS_ALBUMENTATIONS = None  # Controlled per mode via default_hook_enabled.
PINNED_ULTRALYTICS_VERSION = '8.4.62'

YOLO_MODELS = [
    'yolo11n-seg.pt',
]

SPLIT_POLICIES = [
    {
        'key': 'stratified_grouped_specimen',
        'name': 'Stratified grouped-specimen split',
        'description': 'Disease-stratified split where all images from one shrimp stay together.',
    },
]

if Path('/kaggle/working').exists():
    WORK_DIR = Path('/kaggle/working')
elif Path('/content').exists():
    WORK_DIR = Path('/content')
else:
    WORK_DIR = Path.cwd()

DATASET_DIR = WORK_DIR / 'shrimpDisHandSegV2-1'
EXPERIMENT_ROOT = WORK_DIR / 'shrimp_only_fourier_strongaug_hook_ablation_seed42'
RUNS_DIR = WORK_DIR / 'runs' / 'segment'
REPORT_DIR = EXPERIMENT_ROOT / 'reports'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print('EXECUTION_PLAN:', EXECUTION_PLAN)
print('WORK_DIR:', WORK_DIR)
print('DATASET_DIR:', DATASET_DIR)
print('SMOKE_RUN:', SMOKE_RUN)
print('SEEDS:', SEEDS)
print('DISABLE_ULTRALYTICS_ALBUMENTATIONS:', DISABLE_ULTRALYTICS_ALBUMENTATIONS)
print('PINNED_ULTRALYTICS_VERSION:', PINNED_ULTRALYTICS_VERSION)
print('YOLO_MODELS:', YOLO_MODELS)
print('SPLIT_POLICIES:', [p['key'] for p in SPLIT_POLICIES])
"""


yolo_import = """import ultralytics
from ultralytics import YOLO

print('Ultralytics version:', ultralytics.__version__)
print('Expected version:', PINNED_ULTRALYTICS_VERSION)
assert ultralytics.__version__ == PINNED_ULTRALYTICS_VERSION, 'Ultralytics version mismatch.'
"""


mode_block = """FOURIER_STRONGAUG_HOOK_ABLATION_MODES = [
    {'mode': 'M1', 'enabled': False, 'fourier_enabled': False, 'fourier_transform': 'none', 'fourier_sigma': None, 'fourier_alpha': None, 'strong_aug_enabled': False, 'clean_light_aug_enabled': False, 'default_hook_enabled': False, 'attention_enabled': False, 'model_type': 'baseline', 'attention_policy': 'none', 'key': 'M1_no_fourier_no_aug_hook_off', 'name': 'M1 | no Fourier | no explicit aug | hook off'},
    {'mode': 'M2', 'enabled': False, 'fourier_enabled': True, 'fourier_transform': 'highpass_boost', 'fourier_sigma': 50, 'fourier_alpha': 0.10, 'strong_aug_enabled': False, 'clean_light_aug_enabled': False, 'default_hook_enabled': False, 'attention_enabled': False, 'model_type': 'baseline', 'attention_policy': 'none', 'key': 'M2_fourier_G4_no_aug_hook_off', 'name': 'M2 | Fourier G4 | no explicit aug | hook off'},
    {'mode': 'M3', 'enabled': False, 'fourier_enabled': False, 'fourier_transform': 'none', 'fourier_sigma': None, 'fourier_alpha': None, 'strong_aug_enabled': False, 'clean_light_aug_enabled': False, 'default_hook_enabled': True, 'attention_enabled': False, 'model_type': 'baseline', 'attention_policy': 'none', 'key': 'M3_no_fourier_no_aug_hook_on', 'name': 'M3 | no Fourier | no explicit aug | hook on'},
    {'mode': 'M4', 'enabled': False, 'fourier_enabled': True, 'fourier_transform': 'highpass_boost', 'fourier_sigma': 50, 'fourier_alpha': 0.10, 'strong_aug_enabled': False, 'clean_light_aug_enabled': False, 'default_hook_enabled': True, 'attention_enabled': False, 'model_type': 'baseline', 'attention_policy': 'none', 'key': 'M4_fourier_G4_no_aug_hook_on', 'name': 'M4 | Fourier G4 | no explicit aug | hook on'},
    {'mode': 'M5', 'enabled': False, 'fourier_enabled': False, 'fourier_transform': 'none', 'fourier_sigma': None, 'fourier_alpha': None, 'strong_aug_enabled': True, 'clean_light_aug_enabled': False, 'default_hook_enabled': False, 'attention_enabled': False, 'model_type': 'baseline', 'attention_policy': 'none', 'key': 'M5_no_fourier_strong_aug_hook_off', 'name': 'M5 | no Fourier | teammate strong aug | hook off'},
    {'mode': 'M6', 'enabled': False, 'fourier_enabled': True, 'fourier_transform': 'highpass_boost', 'fourier_sigma': 50, 'fourier_alpha': 0.10, 'strong_aug_enabled': True, 'clean_light_aug_enabled': False, 'default_hook_enabled': False, 'attention_enabled': False, 'model_type': 'baseline', 'attention_policy': 'none', 'key': 'M6_fourier_G4_strong_aug_hook_off', 'name': 'M6 | Fourier G4 | teammate strong aug | hook off'},
    {'mode': 'M7', 'enabled': True, 'fourier_enabled': False, 'fourier_transform': 'none', 'fourier_sigma': None, 'fourier_alpha': None, 'strong_aug_enabled': True, 'clean_light_aug_enabled': False, 'default_hook_enabled': True, 'attention_enabled': False, 'model_type': 'baseline', 'attention_policy': 'none', 'key': 'M7_no_fourier_strong_aug_hook_on', 'name': 'M7 | no Fourier | teammate strong aug | hook on'},
    {'mode': 'M8', 'enabled': True, 'fourier_enabled': True, 'fourier_transform': 'highpass_boost', 'fourier_sigma': 50, 'fourier_alpha': 0.10, 'strong_aug_enabled': True, 'clean_light_aug_enabled': False, 'default_hook_enabled': True, 'attention_enabled': False, 'model_type': 'baseline', 'attention_policy': 'none', 'key': 'M8_fourier_G4_strong_aug_hook_on', 'name': 'M8 | Fourier G4 | teammate strong aug | hook on'},
]
EXPERIMENTS = [mode for mode in FOURIER_STRONGAUG_HOOK_ABLATION_MODES if mode.get('enabled', True)]"""


train_policy_funcs = """def train_args_for_mode(exp):
    if exp.get('strong_aug_enabled'):
        return STRONG_AUG_TRAIN_ARGS
    return NO_AUG_TRAIN_ARGS


def augmentation_policy_name(exp):
    if exp.get('strong_aug_enabled'):
        return 'teammate_strong_yolo_aug'
    return 'none_all_yolo_aug_zero'


"""


model_funcs = """def model_config_for_exp(exp):
    return YOLO_MODEL


def make_model_for_exp(exp):
    return YOLO(YOLO_MODEL)


"""


paper_row = """paper_cols = [
    'split_policy',
    'mode',
    'model',
    'base_model',
    'model_type',
    'attention_enabled',
    'attention_policy',
    'seed',
    'training_seed',
    'params_million',
    'ultralytics_version',
    'fourier_enabled',
    'strong_aug_enabled',
    'clean_light_aug_enabled',
    'default_hook_enabled',
    'preprocessing_policy',
    'fourier_policy',
    'fourier_transform',
    'fourier_sigma',
    'fourier_alpha',
    'baseline_augmentation_policy',
    'hidden_albumentations_disabled',
    'train_epochs_requested',
    'train_patience',
    'train_images',
    'valid_images',
    'test_images',
    'actual_train_image_files',
    'actual_valid_image_files',
    'actual_test_image_files',
    'train_coinfection_images',
    'valid_coinfection_images',
    'test_coinfection_images',
    'train_bg_mask_images',
    'valid_bg_mask_images',
    'test_bg_mask_images',
    'train_wssv_mask_images',
    'valid_wssv_mask_images',
    'test_wssv_mask_images',
    'train_specimens',
    'valid_specimens',
    'test_specimens',
    'train_test_group_overlap',
    'valid_test_group_overlap',
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
    'best_pt',
    'run_path',
]

available_paper_cols = [c for c in paper_cols if c in summary_df.columns]
paper_table = summary_df[available_paper_cols].copy()
paper_table_csv = REPORT_DIR / 'only_fourier_strongaug_hook_ablation_seed42_paper_row.csv'
paper_table.to_csv(paper_table_csv, index=False)

print(f'Saved compact Fourier x strong aug x hook ablation paper row: {paper_table_csv}')
display(paper_table)

fair_rows = paper_table[paper_table['split_policy'] == 'stratified_grouped_specimen']
if not fair_rows.empty:
    print('Fourier x strong aug x hook ablation rows ranked by labeled test mask mAP50:')
    display(fair_rows.sort_values('labeled_test_mask_map50', ascending=False).head(8))
"""


final_download = """print('Download these Kaggle/Colab outputs after training:')
print(f'1. Full ablation summary CSV: {REPORT_DIR / "only_fourier_strongaug_hook_ablation_seed42_summary.csv"}')
print(f'2. Compact paper row CSV: {REPORT_DIR / "only_fourier_strongaug_hook_ablation_seed42_paper_row.csv"}')
print(f'3. Partial CSV if interrupted: {REPORT_DIR / "only_fourier_strongaug_hook_ablation_seed42_partial.csv"}')
print(f'4. Split manifests: {REPORT_DIR / "split_manifests"}')
print(f'5. Selected run folder and checkpoint from {RUNS_DIR}')
print('')
print('Best available Fourier x strong aug x hook ablation row:')
if 'fair_rows' in globals() and not fair_rows.empty:
    display(fair_rows.sort_values('labeled_test_mask_map50', ascending=False)[['mode', 'model', 'fourier_enabled', 'strong_aug_enabled', 'default_hook_enabled', 'labeled_test_mask_map50', 'full_test_mask_map50', 'healthy_test_mask_fp_rate', 'best_pt', 'run_path']].head(1))
else:
    print('No fair grouped-specimen ablation row available yet.')

print(f'6. Ablation train args: {REPORT_DIR / "train_args_fourier_strongaug_hook_ablation_seed42.json"}')

import shutil
archive_path = shutil.make_archive(str(WORK_DIR / 'only_fourier_strongaug_hook_ablation_seed42_outputs'), 'zip', root_dir=str(EXPERIMENT_ROOT))
print(f'7. Zip archive: {archive_path}')
try:
    from google.colab import files
    files.download(archive_path)
except Exception:
    print('Automatic browser download is only available in Colab. On Kaggle, download the archive from the working directory.')
"""


def make_training_helpers(src):
    text = src.replace("TRAIN_PATIENCE = 1 if SMOKE_RUN else 40", "TRAIN_PATIENCE = 1 if SMOKE_RUN else 30")
    text = replace_block(
        text,
        r"FOURIER_ATTENTION_INTERACTION_MODES = \[",
        r"EXPERIMENTS = \[mode for mode in FOURIER_ATTENTION_INTERACTION_MODES if mode\.get\('enabled', True\)\]",
        mode_block,
    )
    text = re.sub(r"\n\nLIGHT_AUG_TRAIN_ARGS = \{.*?\n\}\n(?=\nSTRONG_AUG_TRAIN_ARGS = \{)", "\n", text, count=1, flags=re.DOTALL)
    text = replace_block(text, r"def train_args_for_mode\(exp\):", r"def fourier_device\(\):", train_policy_funcs + "def fourier_device():")
    text = replace_block(text, r"def model_config_for_exp\(exp\):", r"def params_million_for_yolo\(yolo_model\):", model_funcs + "def params_million_for_yolo(yolo_model):")
    text = text.replace(
        "train_args_json = REPORT_DIR / 'train_args_fourier_G4_simam_ca_interaction_seed42.json'\n"
        "train_args_payload = {'default_fourier_sigma': FOURIER_SIGMA, 'default_fourier_alpha': FOURIER_ALPHA, 'no_aug_train_args': NO_AUG_TRAIN_ARGS, 'light_aug_train_args': LIGHT_AUG_TRAIN_ARGS, 'strong_aug_train_args': STRONG_AUG_TRAIN_ARGS, 'fourier_attention_interaction_modes': FOURIER_ATTENTION_INTERACTION_MODES, 'simam_ca_yaml_path': str(SIMAM_CA_YAML_PATH), 'enabled_modes': EXPERIMENTS, 'train_epochs_requested': TRAIN_EPOCHS, 'train_patience': TRAIN_PATIENCE}\n"
        "train_args_json.write_text(json.dumps(train_args_payload, indent=2), encoding='utf-8')\n"
        "print('Saved Fourier G4 x SimAM+CA interaction train args:', train_args_json)",
        "train_args_json = REPORT_DIR / 'train_args_fourier_strongaug_hook_ablation_seed42.json'\n"
        "train_args_payload = {'default_fourier_sigma': FOURIER_SIGMA, 'default_fourier_alpha': FOURIER_ALPHA, 'no_aug_train_args': NO_AUG_TRAIN_ARGS, 'strong_aug_train_args': STRONG_AUG_TRAIN_ARGS, 'fourier_strongaug_hook_ablation_modes': FOURIER_STRONGAUG_HOOK_ABLATION_MODES, 'enabled_modes': EXPERIMENTS, 'train_epochs_requested': TRAIN_EPOCHS, 'train_patience': TRAIN_PATIENCE}\n"
        "train_args_json.write_text(json.dumps(train_args_payload, indent=2), encoding='utf-8')\n"
        "print('Saved Fourier x strong aug x hook ablation train args:', train_args_json)",
    )
    forbidden = ["SIMAM", "CoordAtt", "SIMAM_CA_YAML_PATH", "LIGHT_AUG_TRAIN_ARGS", "FOURIER_ATTENTION_INTERACTION_MODES"]
    remaining = [token for token in forbidden if token in text]
    if remaining:
        raise RuntimeError(f"Training helper still contains forbidden tokens: {remaining}")
    return text


def make_run_cell(src):
    lines = src.splitlines()
    new_lines = lines[:9] + lines[22:]
    text = "\n".join(new_lines)
    text = text.replace(
        "EXPERIMENTS = [mode for mode in FOURIER_ATTENTION_INTERACTION_MODES if mode.get('enabled', True)]\nprint('Enabled Fourier G4 x SimAM+CA interaction modes:', [mode['mode'] for mode in EXPERIMENTS])",
        "print('Enabled Fourier x strong aug x hook ablation modes:', [mode['mode'] for mode in EXPERIMENTS])",
    )
    text = text.replace("only_fourier_G4_simam_ca_interaction_seed42_partial.csv", "only_fourier_strongaug_hook_ablation_seed42_partial.csv")
    text = text.replace("only_fourier_G4_simam_ca_interaction_seed42_summary.csv", "only_fourier_strongaug_hook_ablation_seed42_summary.csv")
    text = text.replace("Saved Fourier G4 x SimAM+CA interaction summary CSV", "Saved Fourier x strong aug x hook ablation summary CSV")
    forbidden = ["TRAIN_PATIENCE =", "TRAIN_EPOCHS =", "TRAIN_BATCH =", "TRAIN_IMGSZ =", "PREDICT_CONF_FOR_COUNT =", "FOURIER_ATTENTION_INTERACTION_MODES", "SimAM", "SIMAM", "CoordAtt"]
    remaining = [token for token in forbidden if token in text]
    if remaining:
        raise RuntimeError(f"Run cell still contains forbidden/redundant tokens: {remaining}")
    return text


def main():
    nb = json.loads(SOURCE_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    new_cells = [
        markdown(title_md),
        markdown(outputs_md),
        markdown("## Run Controls"),
        code(run_controls),
        clean_cell(cells[4]),
        clean_cell(cells[5]),
        code(yolo_import),
        clean_cell(cells[7]),
        clean_cell(cells[8]),
        clean_cell(cells[9]),
        clean_cell(cells[10]),
        clean_cell(cells[11]),
        clean_cell(cells[12]),
        clean_cell(cells[13]),
        clean_cell(cells[14]),
        clean_cell(cells[15]),
        clean_cell(cells[16]),
        markdown("## Training and Evaluation Helpers"),
        code(make_training_helpers(source_of(cells[18]))),
        markdown("## Fourier x Teammate Strong Aug x Default Hook Ablation Runs"),
        code(make_run_cell(source_of(cells[20]))),
        markdown("## Compact Paper Row and Selected Checkpoint"),
        code(paper_row),
        markdown("## Final Download Checklist"),
        code(final_download),
    ]

    out_nb = {
        "cells": new_cells,
        "metadata": nb.get("metadata", {}),
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 5),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(out_nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(OUT_NB)


if __name__ == "__main__":
    main()
