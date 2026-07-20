"""Generate the strict no-augmentation FDDEM cutoff-sweep notebook."""

from __future__ import annotations

import json
from pathlib import Path

from generate_017_learnable_fddem_no_aug import as_source, replace_assignment_block


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "019_fddem_band_tuning_gamma001_no_aug_hook_off_seed42.ipynb"
OUTPUT_NOTEBOOK = ROOT / "020_fddem_cutoff_sweep_gamma001_no_aug_hook_off_seed42.ipynb"

PLAN_NAME = "only_fourier_20_fddem_cutoff_sweep_gamma001_no_aug_hook_off_seed42"
OUTPUT_STEM = "only_fourier_fddem_cutoff_sweep_gamma001_no_aug_hook_off_seed42"


FDDEM_CUTOFF_SWEEP_PATCH_CELL = r"""
# Patch Ultralytics with an auditable FDDEM cutoff-sweep module before importing YOLO.
import ast
import importlib.util
import py_compile
import re
import shutil
from pathlib import Path

spec = importlib.util.find_spec('ultralytics')
if spec is None or spec.origin is None:
    raise RuntimeError('Cannot locate installed ultralytics package for FDDEM cutoff sweep patching.')

ULTRA_PKG_DIR = Path(spec.origin).parent
CONV_PY = ULTRA_PKG_DIR / 'nn' / 'modules' / 'conv.py'
MODULES_INIT_PY = ULTRA_PKG_DIR / 'nn' / 'modules' / '__init__.py'
TASKS_PY = ULTRA_PKG_DIR / 'nn' / 'tasks.py'
MODEL_YAML_DIR = EXPERIMENT_ROOT / 'model_yamls'
MODEL_YAML_DIR.mkdir(parents=True, exist_ok=True)

print('Ultralytics package dir:', ULTRA_PKG_DIR)


def backup_once(path):
    backup = path.with_suffix(path.suffix + '.bak_fddem_cutoff_sweep_no_aug')
    if not backup.exists():
        shutil.copy2(path, backup)
        print('Backup created:', backup)


for patch_path in [CONV_PY, MODULES_INIT_PY, TASKS_PY]:
    if not patch_path.exists():
        raise FileNotFoundError(patch_path)
    backup_once(patch_path)


FDDEM_CUTOFF_SWEEP_CODE = '''

# Custom FDDEM cutoff-sweep module for shrimp ONLY_fourier experiments.
class FDDEMCutoff(nn.Module):
    # Conservative single-band high-pass FDDEM with explicit cutoff and gamma.
    # The cutoff is normalized radial frequency in rFFT coordinates.

    def __init__(self, c1, c2=None, cutoff=0.12, init_gamma=0.01):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.cutoff = float(cutoff)
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.spatial = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.complex_weight = nn.Parameter(torch.zeros(1, c2, 1, 1, 2))
        with torch.no_grad():
            self.complex_weight[..., 0].fill_(1.0)
        self.mix = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def _highpass_mask(self, height, width, device, dtype):
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32).abs().view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32).abs().view(1, width // 2 + 1)
        radius = torch.sqrt(fy * fy + fx * fx)
        mask = torch.sigmoid((radius - self.cutoff) * 40.0)
        return mask.to(dtype=dtype).view(1, 1, height, width // 2 + 1)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        dtype = x.dtype
        freq = torch.fft.rfft2(x.float(), norm='ortho')
        mask = self._highpass_mask(height, width, x.device, freq.real.dtype)
        gain = torch.view_as_complex(self.complex_weight.float())[0]
        filtered = freq * mask * gain
        branch = torch.fft.irfft2(filtered, s=(height, width), norm='ortho').to(dtype=dtype)
        detail = self.mix(torch.cat([self.spatial(x), branch], dim=1))
        return x + self.gamma.to(dtype=dtype) * detail
'''

conv_text = CONV_PY.read_text(encoding='utf-8')
if 'class FDDEMCutoff(nn.Module)' not in conv_text:
    CONV_PY.write_text(conv_text.rstrip() + FDDEM_CUTOFF_SWEEP_CODE + '\n', encoding='utf-8')
    py_compile.compile(str(CONV_PY), doraise=True)
    print('Patched conv.py with FDDEMCutoff.')
else:
    print('conv.py already contains FDDEMCutoff; leaving existing patch in place.')


def add_names_to_parenthesized_import(text, anchor, names):
    start = text.find(anchor)
    if start == -1:
        return text, False
    close = text.find(')', start)
    if close == -1:
        return text, False
    block = text[start:close]
    missing = [name for name in names if name not in block]
    if not missing:
        return text, True
    insert = ''.join(f'    {name},\n' for name in missing)
    return text[:close] + insert + text[close:], True


def add_names_to_base_modules(text, names):
    pattern = r'(base_modules\s*=\s*frozenset\(\s*\{)(.*?)(\}\s*\))'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise RuntimeError('Could not locate base_modules frozenset in tasks.py')
    body = match.group(2)
    missing = [name for name in names if name not in body]
    if not missing:
        return text
    body = body.rstrip() + ''.join(f'\n                {name},' for name in missing) + '\n            '
    return text[:match.start(2)] + body + text[match.end(2):]


CUSTOM_MODULES = ['FDDEMCutoff']

init_text = MODULES_INIT_PY.read_text(encoding='utf-8')
init_text, ok = add_names_to_parenthesized_import(init_text, 'from .conv import (', CUSTOM_MODULES)
if not ok:
    for name in CUSTOM_MODULES:
        line = f'from .conv import {name}\n'
        if line.strip() not in init_text:
            init_text += '\n' + line
if '__all__' in init_text:
    idx = init_text.find('__all__ = (')
    if idx != -1:
        close = init_text.find(')', idx)
        block = init_text[idx:close]
        insert = ''.join(f'    "{name}",\n' for name in CUSTOM_MODULES if f'"{name}"' not in block and f"'{name}'" not in block)
        init_text = init_text[:close] + insert + init_text[close:]
MODULES_INIT_PY.write_text(init_text, encoding='utf-8')
print('Patched modules __init__.py for FDDEMCutoff export.')

tasks_text = TASKS_PY.read_text(encoding='utf-8')
tasks_text, ok = add_names_to_parenthesized_import(tasks_text, 'from ultralytics.nn.modules import (', CUSTOM_MODULES)
if not ok:
    raise RuntimeError('Could not locate ultralytics.nn.modules import block in tasks.py')
tasks_text = add_names_to_base_modules(tasks_text, CUSTOM_MODULES)
TASKS_PY.write_text(tasks_text, encoding='utf-8')
ast.parse(TASKS_PY.read_text(encoding='utf-8'))
print('Patched tasks.py for FDDEMCutoff YAML parsing.')

BASE_YAML_TEMPLATE = '''nc: 2
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
{cutoff_layers}
'''

CUTOFF_SWEEP_CONFIGS = {
    'u1_highpass_c010_g001': {'cutoff': 0.10, 'policy': 'highpass_cutoff0p10_gamma0p01'},
    'u2_highpass_c012_g001': {'cutoff': 0.12, 'policy': 'highpass_cutoff0p12_gamma0p01'},
    'u3_highpass_c014_g001': {'cutoff': 0.14, 'policy': 'highpass_cutoff0p14_gamma0p01'},
    'u4_highpass_c016_g001': {'cutoff': 0.16, 'policy': 'highpass_cutoff0p16_gamma0p01'},
}

CUTOFF_SWEEP_YAML_PATHS = {}
for key, cfg in CUTOFF_SWEEP_CONFIGS.items():
    cutoff = cfg['cutoff']
    layers = f'''  # FDDEMCutoff P3/P4/P5, high-pass cutoff {cutoff}, gamma 0.01.
  - [16, 1, FDDEMCutoff, [256, {cutoff}, 0.01]]
  - [19, 1, FDDEMCutoff, [512, {cutoff}, 0.01]]
  - [22, 1, FDDEMCutoff, [1024, {cutoff}, 0.01]]
  - [[23, 24, 25], 1, Segment, [nc, 32, 256]]'''
    yaml_path = MODEL_YAML_DIR / f'yolo11n-seg-{key}.yaml'
    yaml_path.write_text(BASE_YAML_TEMPLATE.format(cutoff_layers=layers), encoding='utf-8')
    CUTOFF_SWEEP_YAML_PATHS[key] = yaml_path
    print('Created FDDEM cutoff-sweep model YAML:', key, yaml_path)

import ultralytics
from ultralytics import YOLO

print('Ultralytics version:', ultralytics.__version__)
print('Expected version:', PINNED_ULTRALYTICS_VERSION)
assert ultralytics.__version__ == PINNED_ULTRALYTICS_VERSION, 'Ultralytics version mismatch.'

for yaml_key, yaml_path in CUTOFF_SWEEP_YAML_PATHS.items():
    _smoke_model = YOLO(str(yaml_path))
    print('FDDEM cutoff-sweep YAML smoke build OK:', yaml_key, yaml_path.name)
    del _smoke_model
""".strip() + "\n"


def replace_all_sources(nb: dict, old: str, new: str) -> None:
    for cell in nb["cells"]:
        cell["source"] = as_source("".join(cell.get("source", [])).replace(old, new))


def main() -> None:
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None

    replace_all_sources(nb, "only_fourier_19_fddem_band_tuning_gamma001_no_aug_hook_off_seed42", PLAN_NAME)
    replace_all_sources(nb, "only_fourier_fddem_band_tuning_gamma001_no_aug_hook_off_seed42", OUTPUT_STEM)
    replace_all_sources(nb, "shrimp_only_fourier_fddem_band_tuning_gamma001_no_aug_hook_off_seed42", f"shrimp_{OUTPUT_STEM}")
    replace_all_sources(nb, "FDDEM band-tuning", "FDDEM cutoff sweep")
    replace_all_sources(nb, "FDDEM band tuning", "FDDEM cutoff sweep")
    replace_all_sources(nb, "band-tuning", "cutoff-sweep")
    replace_all_sources(nb, "band tuning", "cutoff sweep")

    nb["cells"][0]["source"] = as_source(
        """# ONLY Fourier Path 20 - FDDEM Cutoff Sweep, Gamma 0.01, No Augmentation

This notebook performs a narrow high-pass cutoff sweep around the Path 19 winner.

Strict contract:

- Kaggle-ready Roboflow download
- stratified grouped-specimen split, seed 42
- `yolo11n-seg.pt` family
- no explicit YOLO augmentation
- hidden/default Ultralytics Albumentations hook disabled for every row
- no random Fourier train-copy augmentation
- no fixed image preprocessing
- feature-domain FDDEM only

Path 19 showed the high-pass family was better than Gaussian bands, and cutoff `0.12` was the best available row. This notebook fixes `gamma=0.01` and sweeps:

- cutoff 0.10
- cutoff 0.12
- cutoff 0.14
- cutoff 0.16
"""
    )
    nb["cells"][1]["source"] = as_source(
        f"""## Outputs To Download

After the run, download or preserve:

- `{OUTPUT_STEM}_summary.csv`
- `{OUTPUT_STEM}_paper_row.csv`
- `{OUTPUT_STEM}_partial.csv` if interrupted
- `train_args_{OUTPUT_STEM}.json`
- `split_manifests/`
- generated FDDEM cutoff-sweep model YAMLs under `model_yamls/`
- selected `weights/best.pt`
- selected run `results.csv`
- `{OUTPUT_STEM}_outputs.zip`

Rank rows by healthy-aware labeled test score first, then labeled-only test mask mAP50. A candidate is not promoted if it gains mAP50 by creating excessive healthy false positives or disease misses.
"""
    )
    nb["cells"][3]["source"] = as_source(
        f"""# Run controls for ONLY_fourier path 20.
from pathlib import Path

EXECUTION_PLAN = '{PLAN_NAME}'
SMOKE_RUN = False
SEEDS = [42]
DISABLE_ULTRALYTICS_ALBUMENTATIONS = True
PINNED_ULTRALYTICS_VERSION = '8.4.62'

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

if Path('/kaggle/working').exists():
    WORK_DIR = Path('/kaggle/working')
elif Path('/content').exists():
    WORK_DIR = Path('/content')
else:
    WORK_DIR = Path.cwd()

DATASET_DIR = WORK_DIR / 'shrimpDisHandSegV2-1'
EXPERIMENT_ROOT = WORK_DIR / 'shrimp_{OUTPUT_STEM}'
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
    )

    for cell in nb["cells"]:
        if cell.get("id") == "fddem-band-tuning-patch-cell":
            cell["id"] = "fddem-cutoff-sweep-patch-cell"
            cell["source"] = as_source(FDDEM_CUTOFF_SWEEP_PATCH_CELL)
            break
    else:
        raise RuntimeError("Could not find the FDDEM band-tuning patch cell in source notebook.")

    helper_idx = 18
    helper_src = "".join(nb["cells"][helper_idx]["source"])
    helper_src = replace_assignment_block(
        helper_src,
        "FDDEM_BAND_TUNING_MODES",
        """FDDEM_CUTOFF_SWEEP_MODES = [
    {
        'mode': 'U0',
        'enabled': True,
        'fourier_enabled': False,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': False,
        'default_hook_enabled': False,
        'key': 'U0_no_fourier_no_aug_hook_off_baseline',
        'name': 'U0 | no Fourier | no aug | hook off | baseline YOLO11n-seg',
        'model_yaml_key': None,
        'model_yaml': None,
        'model_variant': 'baseline',
        'fourier_module_policy': 'none',
    },
    {
        'mode': 'U1',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'U1_fddem_highpass_c010_g001',
        'name': 'U1 | FDDEMCutoff high-pass cutoff 0.10 gamma 0.01',
        'model_yaml_key': 'u1_highpass_c010_g001',
        'model_yaml': str(CUTOFF_SWEEP_YAML_PATHS['u1_highpass_c010_g001']),
        'model_variant': 'fddem_cutoff_pyramid',
        'fourier_module_policy': 'highpass_cutoff0p10_gamma0p01',
    },
    {
        'mode': 'U2',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'U2_fddem_highpass_c012_g001',
        'name': 'U2 | FDDEMCutoff high-pass cutoff 0.12 gamma 0.01',
        'model_yaml_key': 'u2_highpass_c012_g001',
        'model_yaml': str(CUTOFF_SWEEP_YAML_PATHS['u2_highpass_c012_g001']),
        'model_variant': 'fddem_cutoff_pyramid',
        'fourier_module_policy': 'highpass_cutoff0p12_gamma0p01',
    },
    {
        'mode': 'U3',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'U3_fddem_highpass_c014_g001',
        'name': 'U3 | FDDEMCutoff high-pass cutoff 0.14 gamma 0.01',
        'model_yaml_key': 'u3_highpass_c014_g001',
        'model_yaml': str(CUTOFF_SWEEP_YAML_PATHS['u3_highpass_c014_g001']),
        'model_variant': 'fddem_cutoff_pyramid',
        'fourier_module_policy': 'highpass_cutoff0p14_gamma0p01',
    },
    {
        'mode': 'U4',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'U4_fddem_highpass_c016_g001',
        'name': 'U4 | FDDEMCutoff high-pass cutoff 0.16 gamma 0.01',
        'model_yaml_key': 'u4_highpass_c016_g001',
        'model_yaml': str(CUTOFF_SWEEP_YAML_PATHS['u4_highpass_c016_g001']),
        'model_variant': 'fddem_cutoff_pyramid',
        'fourier_module_policy': 'highpass_cutoff0p16_gamma0p01',
    },
]

EXPERIMENTS = [mode for mode in FDDEM_CUTOFF_SWEEP_MODES if mode.get('enabled', True)]""",
    )
    helper_src = helper_src.replace(
        "\n\nEXPERIMENTS = [mode for mode in FDDEM_BAND_TUNING_MODES if mode.get('enabled', True)]",
        "",
    )
    helper_src = helper_src.replace(
        "'fddem_band_tuning_modes': FDDEM_BAND_TUNING_MODES, 'enabled_mode_keys': [m['key'] for m in EXPERIMENTS], 'band_tuning_configs': BAND_TUNING_CONFIGS, 'band_tuning_yaml_paths': {k: str(v) for k, v in BAND_TUNING_YAML_PATHS.items()}",
        "'fddem_cutoff_sweep_modes': FDDEM_CUTOFF_SWEEP_MODES, 'enabled_mode_keys': [m['key'] for m in EXPERIMENTS], 'cutoff_sweep_configs': CUTOFF_SWEEP_CONFIGS, 'cutoff_sweep_yaml_paths': {k: str(v) for k, v in CUTOFF_SWEEP_YAML_PATHS.items()}",
    )
    nb["cells"][helper_idx]["source"] = as_source(helper_src)

    final_idx = 24
    final_src = f"""print('Download these Kaggle/Colab outputs after training:')\nprint(f'1. Full FDDEM cutoff-sweep summary CSV: {{REPORT_DIR / \"{OUTPUT_STEM}_summary.csv\"}}')\nprint(f'2. Compact paper row CSV: {{REPORT_DIR / \"{OUTPUT_STEM}_paper_row.csv\"}}')\nprint(f'3. Partial CSV if interrupted: {{REPORT_DIR / \"{OUTPUT_STEM}_partial.csv\"}}')\nprint(f'4. Split manifests: {{REPORT_DIR / \"split_manifests\"}}')\nprint(f'5. Generated model YAMLs: {{EXPERIMENT_ROOT / \"model_yamls\"}}')\nprint(f'6. Selected run folder and checkpoint from {{RUNS_DIR}}')\nprint('')\nprint('Selected strict no-aug FDDEM cutoff-sweep checkpoint:')\nif 'fair_rows' in globals() and not fair_rows.empty:\n    display(fair_rows.sort_values('healthy_aware_labeled_val_mask_map50', ascending=False)[['mode', 'model_variant', 'fourier_module_policy', 'healthy_aware_labeled_val_mask_map50', 'best_pt', 'run_path']].head(1))\nelse:\n    print('No fair grouped-specimen row available yet.')\n\nprint(f'7. Train args: {{REPORT_DIR / \"train_args_{OUTPUT_STEM}.json\"}}')\n\nimport shutil\narchive_path = shutil.make_archive(str(WORK_DIR / '{OUTPUT_STEM}_outputs'), 'zip', root_dir=str(EXPERIMENT_ROOT))\nprint(f'8. Zip archive: {{archive_path}}')\ntry:\n    from google.colab import files\n    files.download(archive_path)\nexcept Exception:\n    print('Automatic browser download is only available in Colab. On Kaggle, download the archive from the working directory.')\n"""
    nb["cells"][final_idx]["source"] = as_source(final_src)

    OUTPUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
