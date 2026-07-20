"""Generate Colab notebooks for N=1 augmentation family vs W1 Fourier interaction."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from generate_017_learnable_fddem_no_aug import as_source, replace_assignment_block


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "022_fourier_hybrid_sweep_no_aug_hook_off_seed42.ipynb"

FAMILIES = [
    {
        "idx": "023",
        "slug": "hsv",
        "title": "HSV",
        "allowed_keys": ["hsv_h", "hsv_s", "hsv_v"],
        "settings": [
            ("mild", {"hsv_h": 0.005, "hsv_s": 0.20, "hsv_v": 0.10}),
            ("medium", {"hsv_h": 0.010, "hsv_s": 0.35, "hsv_v": 0.20}),
            ("strong", {"hsv_h": 0.020, "hsv_s": 0.50, "hsv_v": 0.30}),
        ],
    },
    {
        "idx": "024",
        "slug": "translate",
        "title": "Translate",
        "allowed_keys": ["translate"],
        "settings": [
            ("t003", {"translate": 0.03}),
            ("t005", {"translate": 0.05}),
            ("t008", {"translate": 0.08}),
        ],
    },
    {
        "idx": "025",
        "slug": "scale",
        "title": "Scale",
        "allowed_keys": ["scale"],
        "settings": [
            ("s010", {"scale": 0.10}),
            ("s020", {"scale": 0.20}),
            ("s035", {"scale": 0.35}),
        ],
    },
    {
        "idx": "026",
        "slug": "fliplr",
        "title": "Horizontal Flip",
        "allowed_keys": ["fliplr"],
        "settings": [
            ("p025", {"fliplr": 0.25}),
            ("p050", {"fliplr": 0.50}),
            ("p075", {"fliplr": 0.75}),
        ],
    },
    {
        "idx": "027",
        "slug": "degrees",
        "title": "Rotation",
        "allowed_keys": ["degrees"],
        "settings": [
            ("d003", {"degrees": 3.0}),
            ("d005", {"degrees": 5.0}),
            ("d008", {"degrees": 8.0}),
        ],
    },
]


def py_literal(value):
    return repr(value)


def safe_policy_from_args(args: dict[str, float]) -> str:
    return "_".join(f"{k}{str(v).replace('.', 'p')}" for k, v in args.items())


def make_modes(family: dict) -> str:
    modes = [
        {
            "mode": "N0",
            "enabled": True,
            "fourier_enabled": False,
            "image_fourier_enabled": False,
            "feature_fourier_enabled": False,
            "default_hook_enabled": False,
            "key": f"N0_{family['slug']}_baseline_no_aug_no_fourier_hook_off",
            "name": f"N0 | {family['title']} study | no augmentation | no Fourier | hook off",
            "model_yaml_key": None,
            "model_yaml": None,
            "model_variant": "baseline",
            "fourier_module_policy": "none",
            "image_fourier_policy": "none",
            "augmentation_family": "none",
            "augmentation_setting": "none",
            "augmentation_policy": "none",
            "aug_args": {},
        },
        {
            "mode": "N1",
            "enabled": True,
            "fourier_enabled": True,
            "image_fourier_enabled": True,
            "feature_fourier_enabled": False,
            "default_hook_enabled": False,
            "key": f"N1_{family['slug']}_w1_only_no_yolo_aug_hook_off",
            "name": f"N1 | {family['title']} study | W1 only | no YOLO augmentation | hook off",
            "model_yaml_key": None,
            "model_yaml": None,
            "model_variant": "baseline",
            "fourier_module_policy": "none",
            "image_fourier_policy": "highpass_s50_a0p10",
            "image_fourier_sigmas": [50],
            "image_fourier_alphas": [0.10],
            "augmentation_family": "none",
            "augmentation_setting": "w1_only",
            "augmentation_policy": "none",
            "aug_args": {},
        },
    ]

    mode_num = 2
    for setting_name, args in family["settings"]:
        policy = f"{family['slug']}_{setting_name}_{safe_policy_from_args(args)}"
        for with_w1 in [False, True]:
            suffix = "with_w1" if with_w1 else "aug_only"
            modes.append(
                {
                    "mode": f"N{mode_num}",
                    "enabled": True,
                    "fourier_enabled": bool(with_w1),
                    "image_fourier_enabled": bool(with_w1),
                    "feature_fourier_enabled": False,
                    "default_hook_enabled": False,
                    "key": f"N{mode_num}_{family['slug']}_{setting_name}_{suffix}_hook_off",
                    "name": f"N{mode_num} | {family['title']} {setting_name} | {suffix.replace('_', ' ')} | hook off",
                    "model_yaml_key": None,
                    "model_yaml": None,
                    "model_variant": "baseline",
                    "fourier_module_policy": "none",
                    "image_fourier_policy": "highpass_s50_a0p10" if with_w1 else "none",
                    "image_fourier_sigmas": [50] if with_w1 else [],
                    "image_fourier_alphas": [0.10] if with_w1 else [],
                    "augmentation_family": family["slug"],
                    "augmentation_setting": setting_name,
                    "augmentation_policy": policy,
                    "aug_args": args,
                }
            )
            mode_num += 1

    lines = ["N1_AUG_W1_MODES = ["]
    for mode in modes:
        lines.append("    {")
        for key, value in mode.items():
            lines.append(f"        {key!r}: {py_literal(value)},")
        lines.append("    },")
    lines.append("]\n")
    lines.append("EXPERIMENTS = [mode for mode in N1_AUG_W1_MODES if mode.get('enabled', True)]")
    return "\n".join(lines)


def replace_function_block(src: str, start_name: str, end_name: str, replacement: str) -> str:
    start = src.index(f"def {start_name}")
    end = src.index(f"def {end_name}", start)
    return src[:start] + replacement.rstrip() + "\n\n\n" + src[end:]


def generate_family(family: dict) -> Path:
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None

    idx = family["idx"]
    slug = family["slug"]
    output_stem = f"only_fourier_n1_{slug}_w1_interaction_colab_seed42"
    plan_name = f"only_fourier_{idx}_n1_{slug}_w1_interaction_colab_seed42"
    output_notebook = ROOT / f"{idx}_n1_{slug}_w1_interaction_colab_seed42.ipynb"

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        src = src.replace("only_fourier_22_hybrid_sweep_no_aug_hook_off_seed42", plan_name)
        src = src.replace("only_fourier_hybrid_sweep_no_aug_hook_off_seed42", output_stem)
        src = src.replace("shrimp_only_fourier_hybrid_sweep_no_aug_hook_off_seed42", f"shrimp_{output_stem}")
        src = src.replace("Fourier hybrid sweep", f"N=1 {family['title']} + W1 interaction")
        cell["source"] = as_source(src)

    nb["cells"][0]["source"] = as_source(
        f"""# ONLY Fourier Path {idx} - N=1 {family['title']} + W1 Interaction, Colab

This Colab-ready notebook tests one YOLO-native augmentation family at a time, paired with and without W1 Fourier preprocessing.

Strict contract:

- Roboflow dataset download
- stratified grouped-specimen split, seed 42
- `yolo11n-seg.pt`
- hidden/default Ultralytics Albumentations hook disabled for every row
- only the `{family['title']}` augmentation family may be nonzero in this notebook
- W1, when enabled, is deterministic image Fourier high-pass `sigma=50`, `alpha=0.10`

Rows:

- baseline no augmentation, no Fourier
- W1 only
- three `{family['title']}` settings without W1
- the same three `{family['title']}` settings with W1
"""
    )
    nb["cells"][1]["source"] = as_source(
        f"""## Outputs To Download

After the run, download or preserve:

- `{output_stem}_summary.csv`
- `{output_stem}_paper_row.csv`
- `{output_stem}_partial.csv` if interrupted
- `train_args_{output_stem}.json`
- `split_manifests/`
- selected `weights/best.pt`
- selected run `results.csv`
- `{output_stem}_outputs.zip`

Compare each augmentation setting as paired rows: augmentation-only versus augmentation + W1.
"""
    )
    nb["cells"][3]["source"] = as_source(
        f"""# Run controls for ONLY_fourier path {idx}.
from pathlib import Path

EXECUTION_PLAN = '{plan_name}'
SMOKE_RUN = False
SEEDS = [42]
DISABLE_ULTRALYTICS_ALBUMENTATIONS = True
PINNED_ULTRALYTICS_VERSION = '8.4.62'
AUGMENTATION_FAMILY = '{slug}'
AUGMENTATION_FAMILY_NAME = '{family['title']}'

YOLO_MODELS = [
    'yolo11n-seg.pt',
]

SPLIT_POLICIES = [
    {{
        'key': 'stratified_grouped_specimen',
        'name': 'Stratified grouped-specimen split',
        'description': 'Disease-stratified split where all images from one shrimp stay together.',
    }},
]

if Path('/content').exists():
    WORK_DIR = Path('/content')
else:
    WORK_DIR = Path.cwd()

DATASET_DIR = WORK_DIR / 'shrimpDisHandSegV2-1'
EXPERIMENT_ROOT = WORK_DIR / 'shrimp_{output_stem}'
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
print('AUGMENTATION_FAMILY:', AUGMENTATION_FAMILY)
print('YOLO_MODELS:', YOLO_MODELS)
print('SPLIT_POLICIES:', [p['key'] for p in SPLIT_POLICIES])
"""
    )

    # Replace the feature-module patch cell with a plain Ultralytics import/version check.
    for cell in nb["cells"]:
        if cell.get("id") == "fourier-hybrid-patch-cell":
            cell["id"] = f"n1-{slug}-ultralytics-import-cell"
            cell["source"] = as_source(
                """import ultralytics
from ultralytics import YOLO

print('Ultralytics version:', ultralytics.__version__)
print('Expected version:', PINNED_ULTRALYTICS_VERSION)
assert ultralytics.__version__ == PINNED_ULTRALYTICS_VERSION, 'Ultralytics version mismatch.'
"""
            )
            break
    else:
        raise RuntimeError("Could not find source patch/import cell.")

    helper_idx = 18
    helper_src = "".join(nb["cells"][helper_idx]["source"])
    helper_src = helper_src.replace(
        "TRAIN_BATCH = 8 if SMOKE_RUN else 16\n",
        "TRAIN_BATCH = 8 if SMOKE_RUN else 16\nTRAIN_WORKERS = 0 if SMOKE_RUN else 2\n",
    )
    helper_src = helper_src.replace(
        "yolo.train(data=str(yaml_path), task='segment', imgsz=TRAIN_IMGSZ, epochs=TRAIN_EPOCHS, batch=TRAIN_BATCH, patience=TRAIN_PATIENCE, seed=training_seed, project=str(RUNS_DIR), name=run_name, exist_ok=True, pretrained=True, plots=not SMOKE_RUN, verbose=True, **train_args_for_mode(exp))",
        "yolo.train(data=str(yaml_path), task='segment', imgsz=TRAIN_IMGSZ, epochs=TRAIN_EPOCHS, batch=TRAIN_BATCH, workers=TRAIN_WORKERS, patience=TRAIN_PATIENCE, seed=training_seed, project=str(RUNS_DIR), name=run_name, exist_ok=True, pretrained=True, plots=not SMOKE_RUN, verbose=True, **train_args_for_mode(exp))",
    )
    helper_src = helper_src.replace(
        "    del yolo, best_model\n    gc.collect()\n    return row",
        "    del yolo, best_model\n    cleanup_runtime_memory(exp.get('mode', 'run_experiment'))\n    return row",
    )
    helper_src = replace_assignment_block(helper_src, "FOURIER_HYBRID_MODES", make_modes(family))
    helper_src = helper_src.replace(
        "\n\nEXPERIMENTS = [mode for mode in FOURIER_HYBRID_MODES if mode.get('enabled', True)]",
        "",
    )
    helper_src = replace_function_block(
        helper_src,
        "assert_strict_no_augmentation",
        "augmentation_policy_name",
        f"""ALLOWED_AUG_KEYS = {family['allowed_keys']!r}


def assert_single_family_augmentation(train_args, exp):
    if exp.get('default_hook_enabled'):
        raise AssertionError(f"{{exp['mode']}} tried to enable the hidden/default Albumentations hook.")
    if train_args.get('auto_augment', None) is not None:
        raise AssertionError(f"{{exp['mode']}} auto_augment must stay None.")
    if bool(train_args.get('multi_scale', False)):
        raise AssertionError(f"{{exp['mode']}} multi_scale must stay False.")
    nonzero = {{key: train_args.get(key) for key in STRICT_NO_AUG_KEYS if float(train_args.get(key, 0.0) or 0.0) != 0.0}}
    illegal = {{key: value for key, value in nonzero.items() if key not in ALLOWED_AUG_KEYS}}
    if illegal:
        raise AssertionError(f"{{exp['mode']}} has augmentation outside {family['title']}: {{illegal}}")
    expected = exp.get('aug_args', {{}})
    for key in expected:
        if key not in ALLOWED_AUG_KEYS:
            raise AssertionError(f"{{exp['mode']}} aug_args contains disallowed key {{key}}.")
    return True


def train_args_for_mode(exp):
    train_args = dict(NO_AUG_TRAIN_ARGS)
    train_args.update(exp.get('aug_args') or {{}})
    assert_single_family_augmentation(train_args, exp)
    return train_args
""",
    )
    helper_src = helper_src.replace(
        "def augmentation_policy_name(exp):\n    return exp.get('augmentation_policy', 'none')",
        """def cleanup_runtime_memory(label=''):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'Runtime memory cleanup complete: {label}')


def augmentation_policy_name(exp):
    return exp.get('augmentation_policy', 'none')""",
    )
    helper_src = helper_src.replace(
        "def augmentation_policy_name(exp):\n    return 'none_all_yolo_aug_zero_hidden_hook_off'",
        "def augmentation_policy_name(exp):\n    return exp.get('augmentation_policy', 'none')",
    )
    helper_src = helper_src.replace(
        "def augmentation_policy_name(exp):\n    return exp.get('augmentation_policy', 'none')",
        """def cleanup_runtime_memory(label=''):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'Runtime memory cleanup complete: {label}')


def augmentation_policy_name(exp):
    return exp.get('augmentation_policy', 'none')""",
    )
    helper_src = helper_src.replace(
        "'model_variant': exp.get('model_variant', 'baseline'), 'model_yaml_key': exp.get('model_yaml_key'), 'model_yaml': model_yaml_for_mode(exp),\n        'fourier_module_policy': exp.get('fourier_module_policy', 'none'), 'params_million': current_params_million,",
        "'model_variant': exp.get('model_variant', 'baseline'), 'model_yaml_key': exp.get('model_yaml_key'), 'model_yaml': model_yaml_for_mode(exp),\n        'augmentation_family': exp.get('augmentation_family', 'none'), 'augmentation_setting': exp.get('augmentation_setting', 'none'),\n        'augmentation_args_json': json.dumps(exp.get('aug_args', {}), sort_keys=True),\n        'fourier_module_policy': exp.get('fourier_module_policy', 'none'), 'params_million': current_params_million,",
    )
    helper_src = helper_src.replace(
        "'fourier_hybrid_modes': FOURIER_HYBRID_MODES, 'enabled_mode_keys': [m['key'] for m in EXPERIMENTS], 'hybrid_yaml_paths': {k: str(v) for k, v in HYBRID_YAML_PATHS.items()}",
        f"'n1_aug_w1_modes': N1_AUG_W1_MODES, 'augmentation_family': AUGMENTATION_FAMILY, 'allowed_aug_keys': ALLOWED_AUG_KEYS, 'enabled_mode_keys': [m['key'] for m in EXPERIMENTS]",
    )
    helper_src = helper_src.replace(
        "print('Saved strict no-aug N=1",
        "print('Saved N=1",
    )
    nb["cells"][helper_idx]["source"] = as_source(helper_src)

    run_idx = 20
    run_src = "".join(nb["cells"][run_idx]["source"])
    run_src = run_src.replace(
        "print('Enabled strict no-aug N=1",
        "print('Enabled N=1",
    )
    run_src = run_src.replace(
        "                display(partial_df)\n                print(f'Saved partial CSV: {partial_csv}')",
        "                display(partial_df)\n                print(f'Saved partial CSV: {partial_csv}')\n                del partial_df\n                cleanup_runtime_memory(exp.get('mode', 'loop'))",
    )
    nb["cells"][run_idx]["source"] = as_source(run_src)

    paper_idx = 22
    paper_src = "".join(nb["cells"][paper_idx]["source"])
    if "'augmentation_family'," not in paper_src:
        paper_src = paper_src.replace(
            "'feature_fourier_enabled',\n",
            "'feature_fourier_enabled',\n    'augmentation_family',\n    'augmentation_setting',\n    'augmentation_args_json',\n",
        )
    paper_src = paper_src.replace(
        "Strict no-aug N=1",
        "N=1",
    )
    nb["cells"][paper_idx]["source"] = as_source(paper_src)

    final_idx = 24
    nb["cells"][final_idx]["source"] = as_source(
        f"""print('Download these Colab outputs after training:')
print(f'1. Full N=1 {family['title']} summary CSV: {{REPORT_DIR / "{output_stem}_summary.csv"}}')
print(f'2. Compact paper row CSV: {{REPORT_DIR / "{output_stem}_paper_row.csv"}}')
print(f'3. Partial CSV if interrupted: {{REPORT_DIR / "{output_stem}_partial.csv"}}')
print(f'4. Split manifests: {{REPORT_DIR / "split_manifests"}}')
print(f'5. Selected run folder and checkpoint from {{RUNS_DIR}}')
print('')
print('Selected N=1 {family['title']} checkpoint by validation healthy-aware score:')
if 'fair_rows' in globals() and not fair_rows.empty:
    display(fair_rows.sort_values('healthy_aware_labeled_val_mask_map50', ascending=False)[['mode', 'augmentation_policy', 'image_fourier_policy', 'healthy_aware_labeled_val_mask_map50', 'best_pt', 'run_path']].head(1))
else:
    print('No fair grouped-specimen row available yet.')

print(f'6. Train args: {{REPORT_DIR / "train_args_{output_stem}.json"}}')

import shutil
archive_path = shutil.make_archive(str(WORK_DIR / '{output_stem}_outputs'), 'zip', root_dir=str(EXPERIMENT_ROOT))
print(f'7. Zip archive: {{archive_path}}')
try:
    from google.colab import files
    files.download(archive_path)
except Exception:
    print('Automatic browser download is only available in Colab. Download the archive from the file browser.')
"""
    )

    # Ensure all code cells parse.
    for idx_cell, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))

    output_notebook.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return output_notebook


def main() -> None:
    for family in FAMILIES:
        out = generate_family(family)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
