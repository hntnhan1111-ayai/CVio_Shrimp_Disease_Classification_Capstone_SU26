"""Generate the strict no-augmentation image+feature Fourier hybrid sweep notebook."""

from __future__ import annotations

import json
from pathlib import Path

from generate_017_learnable_fddem_no_aug import as_source, replace_assignment_block


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "021_fddem_confirmation_no_aug_hook_off_seed42.ipynb"
OUTPUT_NOTEBOOK = ROOT / "022_fourier_hybrid_sweep_no_aug_hook_off_seed42.ipynb"

PLAN_NAME = "only_fourier_22_hybrid_sweep_no_aug_hook_off_seed42"
OUTPUT_STEM = "only_fourier_hybrid_sweep_no_aug_hook_off_seed42"


HYBRID_PATCH_CELL = r"""
# Patch Ultralytics with feature-Fourier modules for image+feature hybrid rows.
import ast
import importlib.util
import py_compile
import re
import shutil
from pathlib import Path

spec = importlib.util.find_spec('ultralytics')
if spec is None or spec.origin is None:
    raise RuntimeError('Cannot locate installed ultralytics package for Fourier hybrid patching.')

ULTRA_PKG_DIR = Path(spec.origin).parent
CONV_PY = ULTRA_PKG_DIR / 'nn' / 'modules' / 'conv.py'
MODULES_INIT_PY = ULTRA_PKG_DIR / 'nn' / 'modules' / '__init__.py'
TASKS_PY = ULTRA_PKG_DIR / 'nn' / 'tasks.py'
MODEL_YAML_DIR = EXPERIMENT_ROOT / 'model_yamls'
MODEL_YAML_DIR.mkdir(parents=True, exist_ok=True)

print('Ultralytics package dir:', ULTRA_PKG_DIR)


def backup_once(path):
    backup = path.with_suffix(path.suffix + '.bak_fourier_hybrid_no_aug')
    if not backup.exists():
        shutil.copy2(path, backup)
        print('Backup created:', backup)


for patch_path in [CONV_PY, MODULES_INIT_PY, TASKS_PY]:
    if not patch_path.exists():
        raise FileNotFoundError(patch_path)
    backup_once(patch_path)


HYBRID_MODULE_CODE = '''

# FDDEM_HYBRID_MODULES_V1
class FDDEM(nn.Module):
    def __init__(self, c1, c2=None, num_bands=2, init_gamma=0.0):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.num_bands = int(num_bands)
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.spatial = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.complex_weight = nn.Parameter(torch.zeros(self.num_bands, c2, 1, 1, 2))
        with torch.no_grad():
            self.complex_weight[..., 0].fill_(1.0)
        self.mix = nn.Sequential(
            nn.Conv2d(c2 * (self.num_bands + 1), c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def _radial_masks(self, height, width, device, dtype):
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32).abs().view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32).abs().view(1, width // 2 + 1)
        radius = torch.sqrt(fy * fy + fx * fx)
        masks = []
        if self.num_bands >= 1:
            masks.append(torch.sigmoid((radius - 0.12) * 40.0))
        if self.num_bands >= 2:
            masks.append(torch.exp(-((radius - 0.22) ** 2) / (2.0 * 0.08 ** 2)))
        while len(masks) < self.num_bands:
            center = 0.08 + 0.08 * len(masks)
            masks.append(torch.exp(-((radius - center) ** 2) / (2.0 * 0.05 ** 2)))
        return torch.stack(masks, dim=0).to(dtype=dtype).view(self.num_bands, 1, height, width // 2 + 1)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        dtype = x.dtype
        freq = torch.fft.rfft2(x.float(), norm='ortho')
        masks = self._radial_masks(height, width, x.device, freq.real.dtype)
        gains = torch.view_as_complex(self.complex_weight.float())
        branches = []
        for band_idx in range(self.num_bands):
            filtered = freq * masks[band_idx] * gains[band_idx]
            branch = torch.fft.irfft2(filtered, s=(height, width), norm='ortho').to(dtype=dtype)
            branches.append(branch)
        detail = self.mix(torch.cat([self.spatial(x), *branches], dim=1))
        return x + self.gamma.to(dtype=dtype) * detail


class FDDEMCutoff(nn.Module):
    def __init__(self, c1, c2=None, cutoff=0.14, init_gamma=0.01):
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
if 'FDDEM_HYBRID_MODULES_V1' not in conv_text:
    CONV_PY.write_text(conv_text.rstrip() + HYBRID_MODULE_CODE + '\n', encoding='utf-8')
    py_compile.compile(str(CONV_PY), doraise=True)
    print('Patched conv.py with Fourier hybrid modules.')
else:
    print('conv.py already contains Fourier hybrid modules; leaving existing patch in place.')


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


CUSTOM_MODULES = ['FDDEM', 'FDDEMCutoff']

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
print('Patched modules __init__.py for Fourier hybrid exports.')

tasks_text = TASKS_PY.read_text(encoding='utf-8')
tasks_text, ok = add_names_to_parenthesized_import(tasks_text, 'from ultralytics.nn.modules import (', CUSTOM_MODULES)
if not ok:
    raise RuntimeError('Could not locate ultralytics.nn.modules import block in tasks.py')
tasks_text = add_names_to_base_modules(tasks_text, CUSTOM_MODULES)
TASKS_PY.write_text(tasks_text, encoding='utf-8')
ast.parse(TASKS_PY.read_text(encoding='utf-8'))
print('Patched tasks.py for Fourier hybrid YAML parsing.')

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
{hybrid_layers}
'''

HYBRID_YAMLS = {
    'w6_img_hp50a010_fddemcutoff_c014_g001': '''  # Image high-pass plus FDDEMCutoff cutoff 0.14, gamma 0.01.
  - [16, 1, FDDEMCutoff, [256, 0.14, 0.01]]
  - [19, 1, FDDEMCutoff, [512, 0.14, 0.01]]
  - [22, 1, FDDEMCutoff, [1024, 0.14, 0.01]]
  - [[23, 24, 25], 1, Segment, [nc, 32, 256]]''',
    'w7_img_hp50a010_fddem_twoband_identity': '''  # Image high-pass plus 017 F3 FDDEM two-band identity-gated reference.
  - [16, 1, FDDEM, [256, 2, 0.00]]
  - [19, 1, FDDEM, [512, 2, 0.00]]
  - [22, 1, FDDEM, [1024, 2, 0.00]]
  - [[23, 24, 25], 1, Segment, [nc, 32, 256]]''',
}

HYBRID_YAML_PATHS = {}
for key, layers in HYBRID_YAMLS.items():
    yaml_path = MODEL_YAML_DIR / f'yolo11n-seg-{key}.yaml'
    yaml_path.write_text(BASE_YAML_TEMPLATE.format(hybrid_layers=layers), encoding='utf-8')
    HYBRID_YAML_PATHS[key] = yaml_path
    print('Created Fourier hybrid model YAML:', key, yaml_path)

import ultralytics
from ultralytics import YOLO

print('Ultralytics version:', ultralytics.__version__)
print('Expected version:', PINNED_ULTRALYTICS_VERSION)
assert ultralytics.__version__ == PINNED_ULTRALYTICS_VERSION, 'Ultralytics version mismatch.'

for yaml_key, yaml_path in HYBRID_YAML_PATHS.items():
    _smoke_model = YOLO(str(yaml_path))
    print('Fourier hybrid YAML smoke build OK:', yaml_key, yaml_path.name)
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

    replace_all_sources(nb, "only_fourier_21_fddem_confirmation_no_aug_hook_off_seed42", PLAN_NAME)
    replace_all_sources(nb, "only_fourier_fddem_confirmation_no_aug_hook_off_seed42", OUTPUT_STEM)
    replace_all_sources(nb, "shrimp_only_fourier_fddem_confirmation_no_aug_hook_off_seed42", f"shrimp_{OUTPUT_STEM}")
    replace_all_sources(nb, "FDDEM confirmation", "Fourier hybrid sweep")

    nb["cells"][0]["source"] = as_source(
        """# ONLY Fourier Path 22 - Image High-Pass + Feature Fourier Hybrid Sweep, No Augmentation

This notebook tries to beat the current Fourier-only raw mAP50 leader:

- fixed image Fourier high-pass `sigma=50`, `alpha=0.10`
- prior labeled test mask mAP50: `0.345105`

Strict contract:

- Kaggle-ready Roboflow download
- stratified grouped-specimen split, seed 42
- `yolo11n-seg.pt` family
- no explicit YOLO augmentation
- hidden/default Ultralytics Albumentations hook disabled for every row
- Fourier-only deterministic preprocessing and/or Fourier feature modules
- no random train-copy augmentation

The key hypothesis is that the best raw-mAP image-space Fourier signal can be stacked with feature-domain Fourier refinement while still remaining a Fourier-only experiment.
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
- generated Fourier hybrid model YAMLs under `model_yamls/`
- selected `weights/best.pt`
- selected run `results.csv`
- `{OUTPUT_STEM}_outputs.zip`

Primary target: beat labeled-only test mask mAP50 `0.345105` from fixed image high-pass `sigma=50 alpha=0.10`.
"""
    )
    nb["cells"][3]["source"] = as_source(
        f"""# Run controls for ONLY_fourier path 22.
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
        if cell.get("id") == "fddem-confirmation-patch-cell":
            cell["id"] = "fourier-hybrid-patch-cell"
            cell["source"] = as_source(HYBRID_PATCH_CELL)
            break
    else:
        raise RuntimeError("Could not find the FDDEM confirmation patch cell in source notebook.")

    helper_idx = 18
    helper_src = "".join(nb["cells"][helper_idx]["source"])
    helper_src = replace_assignment_block(
        helper_src,
        "FDDEM_CONFIRMATION_MODES",
        """FOURIER_HYBRID_MODES = [
    {
        'mode': 'W0',
        'enabled': True,
        'fourier_enabled': False,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': False,
        'default_hook_enabled': False,
        'key': 'W0_no_fourier_no_aug_hook_off_baseline',
        'name': 'W0 | no Fourier | no aug | hook off | baseline YOLO11n-seg',
        'model_yaml_key': None,
        'model_yaml': None,
        'model_variant': 'baseline',
        'fourier_module_policy': 'none',
        'image_fourier_policy': 'none',
    },
    {
        'mode': 'W1',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': True,
        'feature_fourier_enabled': False,
        'default_hook_enabled': False,
        'key': 'W1_img_highpass_s50_a010',
        'name': 'W1 | image Fourier high-pass sigma 50 alpha 0.10',
        'model_yaml_key': None,
        'model_yaml': None,
        'model_variant': 'baseline',
        'fourier_module_policy': 'none',
        'image_fourier_policy': 'highpass_s50_a0p10',
        'image_fourier_sigmas': [50],
        'image_fourier_alphas': [0.10],
    },
    {
        'mode': 'W2',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': True,
        'feature_fourier_enabled': False,
        'default_hook_enabled': False,
        'key': 'W2_img_highpass_s40_a010',
        'name': 'W2 | image Fourier high-pass sigma 40 alpha 0.10',
        'model_yaml_key': None,
        'model_yaml': None,
        'model_variant': 'baseline',
        'fourier_module_policy': 'none',
        'image_fourier_policy': 'highpass_s40_a0p10',
        'image_fourier_sigmas': [40],
        'image_fourier_alphas': [0.10],
    },
    {
        'mode': 'W3',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': True,
        'feature_fourier_enabled': False,
        'default_hook_enabled': False,
        'key': 'W3_img_highpass_s50_a012',
        'name': 'W3 | image Fourier high-pass sigma 50 alpha 0.12',
        'model_yaml_key': None,
        'model_yaml': None,
        'model_variant': 'baseline',
        'fourier_module_policy': 'none',
        'image_fourier_policy': 'highpass_s50_a0p12',
        'image_fourier_sigmas': [50],
        'image_fourier_alphas': [0.12],
    },
    {
        'mode': 'W4',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': True,
        'feature_fourier_enabled': False,
        'default_hook_enabled': False,
        'key': 'W4_img_highpass_s60_a010',
        'name': 'W4 | image Fourier high-pass sigma 60 alpha 0.10',
        'model_yaml_key': None,
        'model_yaml': None,
        'model_variant': 'baseline',
        'fourier_module_policy': 'none',
        'image_fourier_policy': 'highpass_s60_a0p10',
        'image_fourier_sigmas': [60],
        'image_fourier_alphas': [0.10],
    },
    {
        'mode': 'W5',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': True,
        'feature_fourier_enabled': False,
        'default_hook_enabled': False,
        'key': 'W5_img_multiscale_s30a005_s80a005',
        'name': 'W5 | image Fourier multi-scale high-pass s30 a0.05 + s80 a0.05',
        'model_yaml_key': None,
        'model_yaml': None,
        'model_variant': 'baseline',
        'fourier_module_policy': 'none',
        'image_fourier_policy': 'multiscale_highpass_s30a0p05_s80a0p05',
        'image_fourier_sigmas': [30, 80],
        'image_fourier_alphas': [0.05, 0.05],
    },
    {
        'mode': 'W6',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': True,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'W6_img_s50a010_fddemcutoff_c014_g001',
        'name': 'W6 | image high-pass s50 a0.10 + FDDEMCutoff cutoff 0.14 gamma 0.01',
        'model_yaml_key': 'w6_img_hp50a010_fddemcutoff_c014_g001',
        'model_yaml': str(HYBRID_YAML_PATHS['w6_img_hp50a010_fddemcutoff_c014_g001']),
        'model_variant': 'hybrid_img_hp50_fddemcutoff014',
        'fourier_module_policy': 'fddemcutoff_cutoff0p14_gamma0p01',
        'image_fourier_policy': 'highpass_s50_a0p10',
        'image_fourier_sigmas': [50],
        'image_fourier_alphas': [0.10],
    },
    {
        'mode': 'W7',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': True,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'W7_img_s50a010_fddem_twoband_identity',
        'name': 'W7 | image high-pass s50 a0.10 + FDDEM two-band identity',
        'model_yaml_key': 'w7_img_hp50a010_fddem_twoband_identity',
        'model_yaml': str(HYBRID_YAML_PATHS['w7_img_hp50a010_fddem_twoband_identity']),
        'model_variant': 'hybrid_img_hp50_fddem_twoband_identity',
        'fourier_module_policy': 'fddem_twoband_gamma0_identity',
        'image_fourier_policy': 'highpass_s50_a0p10',
        'image_fourier_sigmas': [50],
        'image_fourier_alphas': [0.10],
    },
]

EXPERIMENTS = [mode for mode in FOURIER_HYBRID_MODES if mode.get('enabled', True)]""",
    )
    helper_src = helper_src.replace(
        "\n\nEXPERIMENTS = [mode for mode in FDDEM_CONFIRMATION_MODES if mode.get('enabled', True)]",
        "",
    )
    helper_src = helper_src.replace(
        "def apply_fourier_highpass_to_dataset(dataset_dir, sigma=FOURIER_SIGMA, alpha=FOURIER_ALPHA):\n    summary = {}\n    for split in ['train', 'valid', 'test']:\n        summary[split] = apply_fourier_highpass_to_image_dir(Path(dataset_dir) / split / 'images', sigma=sigma, alpha=alpha)\n    remove_yolo_label_caches(dataset_dir)\n    return summary",
        """def apply_fourier_highpass_to_dataset(dataset_dir, sigma=FOURIER_SIGMA, alpha=FOURIER_ALPHA):
    summary = {}
    for split in ['train', 'valid', 'test']:
        summary[split] = apply_fourier_highpass_to_image_dir(Path(dataset_dir) / split / 'images', sigma=sigma, alpha=alpha)
    remove_yolo_label_caches(dataset_dir)
    return summary


def apply_image_fourier_for_mode(dataset_dir, exp):
    if not exp.get('image_fourier_enabled'):
        return None
    sigmas = list(exp.get('image_fourier_sigmas') or [FOURIER_SIGMA])
    alphas = list(exp.get('image_fourier_alphas') or [FOURIER_ALPHA])
    if len(sigmas) != len(alphas):
        raise ValueError(f"{exp['mode']} image_fourier_sigmas and image_fourier_alphas length mismatch.")
    policy = exp.get('image_fourier_policy', 'highpass')
    print(f"Applying image Fourier policy for {exp['mode']}: {policy} | sigmas={sigmas} | alphas={alphas}")
    summary = {'policy': policy, 'sigmas': sigmas, 'alphas': alphas, 'passes': []}
    for pass_idx, (sigma, alpha) in enumerate(zip(sigmas, alphas), start=1):
        pass_summary = apply_fourier_highpass_to_dataset(dataset_dir, sigma=float(sigma), alpha=float(alpha))
        summary['passes'].append({'pass': pass_idx, 'sigma': float(sigma), 'alpha': float(alpha), 'summary': pass_summary})
    return summary""",
    )
    helper_src = helper_src.replace(
        "if exp.get('image_fourier_enabled'):\n        fourier_transform_summary = apply_fourier_highpass_to_dataset(dataset_dir, sigma=FOURIER_SIGMA, alpha=FOURIER_ALPHA)\n    else:\n        fourier_transform_summary = {'strategy': 'fourier_disabled', 'fourier_enabled': False, 'sigma': None, 'alpha': None}\n        print('Fixed image Fourier preprocessing disabled for this mode; dataset images remain original.')",
        "if exp.get('image_fourier_enabled'):\n        fourier_transform_summary = apply_image_fourier_for_mode(dataset_dir, exp)\n    else:\n        fourier_transform_summary = None\n        print('Fixed image Fourier preprocessing disabled for this mode; dataset images remain original.')",
    )
    helper_src = helper_src.replace(
        "'preprocessing_policy': 'none',\n        'fourier_policy': exp.get('fourier_module_policy', 'none') if exp.get('feature_fourier_enabled') else 'none',\n        'fourier_sigma': None, 'fourier_alpha': None,",
        "'preprocessing_policy': exp.get('image_fourier_policy', 'none') if exp.get('image_fourier_enabled') else 'none',\n        'image_fourier_policy': exp.get('image_fourier_policy', 'none'),\n        'image_fourier_sigmas': '|'.join(str(x) for x in exp.get('image_fourier_sigmas', [])),\n        'image_fourier_alphas': '|'.join(str(x) for x in exp.get('image_fourier_alphas', [])),\n        'fourier_policy': '+'.join([p for p in [exp.get('image_fourier_policy', 'none') if exp.get('image_fourier_enabled') else 'none', exp.get('fourier_module_policy', 'none') if exp.get('feature_fourier_enabled') else 'none'] if p != 'none']) or 'none',\n        'fourier_sigma': '|'.join(str(x) for x in exp.get('image_fourier_sigmas', [])) or None, 'fourier_alpha': '|'.join(str(x) for x in exp.get('image_fourier_alphas', [])) or None,",
    )
    helper_src = helper_src.replace(
        "'fddem_confirmation_modes': FDDEM_CONFIRMATION_MODES, 'enabled_mode_keys': [m['key'] for m in EXPERIMENTS], 'confirmation_yaml_paths': {k: str(v) for k, v in CONFIRMATION_YAML_PATHS.items()}",
        "'fourier_hybrid_modes': FOURIER_HYBRID_MODES, 'enabled_mode_keys': [m['key'] for m in EXPERIMENTS], 'hybrid_yaml_paths': {k: str(v) for k, v in HYBRID_YAML_PATHS.items()}",
    )
    nb["cells"][helper_idx]["source"] = as_source(helper_src)

    paper_idx = 22
    paper_src = "".join(nb["cells"][paper_idx]["source"])
    paper_src = paper_src.replace(
        "'feature_fourier_enabled',\n    'model_variant',",
        "'feature_fourier_enabled',\n    'image_fourier_policy',\n    'image_fourier_sigmas',\n    'image_fourier_alphas',\n    'model_variant',",
    )
    nb["cells"][paper_idx]["source"] = as_source(paper_src)

    final_idx = 24
    final_src = f"""print('Download these Kaggle/Colab outputs after training:')\nprint(f'1. Full Fourier hybrid summary CSV: {{REPORT_DIR / \"{OUTPUT_STEM}_summary.csv\"}}')\nprint(f'2. Compact paper row CSV: {{REPORT_DIR / \"{OUTPUT_STEM}_paper_row.csv\"}}')\nprint(f'3. Partial CSV if interrupted: {{REPORT_DIR / \"{OUTPUT_STEM}_partial.csv\"}}')\nprint(f'4. Split manifests: {{REPORT_DIR / \"split_manifests\"}}')\nprint(f'5. Generated model YAMLs: {{EXPERIMENT_ROOT / \"model_yamls\"}}')\nprint(f'6. Selected run folder and checkpoint from {{RUNS_DIR}}')\nprint('')\nprint('Selected strict no-aug Fourier hybrid checkpoint:')\nif 'fair_rows' in globals() and not fair_rows.empty:\n    display(fair_rows.sort_values('healthy_aware_labeled_val_mask_map50', ascending=False)[['mode', 'model_variant', 'image_fourier_policy', 'fourier_module_policy', 'healthy_aware_labeled_val_mask_map50', 'best_pt', 'run_path']].head(1))\nelse:\n    print('No fair grouped-specimen row available yet.')\n\nprint(f'7. Train args: {{REPORT_DIR / \"train_args_{OUTPUT_STEM}.json\"}}')\n\nimport shutil\narchive_path = shutil.make_archive(str(WORK_DIR / '{OUTPUT_STEM}_outputs'), 'zip', root_dir=str(EXPERIMENT_ROOT))\nprint(f'8. Zip archive: {{archive_path}}')\ntry:\n    from google.colab import files\n    files.download(archive_path)\nexcept Exception:\n    print('Automatic browser download is only available in Colab. On Kaggle, download the archive from the working directory.')\n"""
    nb["cells"][final_idx]["source"] = as_source(final_src)

    OUTPUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
