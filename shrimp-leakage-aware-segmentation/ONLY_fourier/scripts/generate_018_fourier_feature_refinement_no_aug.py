"""Generate the strict no-augmentation Fourier feature-refinement notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path

from generate_017_learnable_fddem_no_aug import as_source, replace_assignment_block, replace_block


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "training_log" / "12_only_fourier_12_factorial_A_H_highpass_s50_a0p10.ipynb"
OUTPUT_NOTEBOOK = ROOT / "018_fourier_feature_refinement_no_aug_hook_off_seed42.ipynb"

PLAN_NAME = "only_fourier_18_feature_refinement_no_aug_hook_off_seed42"
OUTPUT_STEM = "only_fourier_feature_refinement_no_aug_hook_off_seed42"


FOURIER_REFINEMENT_PATCH_CELL = r"""
# Patch Ultralytics with learnable Fourier feature-refinement modules before importing YOLO.
import ast
import importlib.util
import py_compile
import re
import shutil
from pathlib import Path

spec = importlib.util.find_spec('ultralytics')
if spec is None or spec.origin is None:
    raise RuntimeError('Cannot locate installed ultralytics package for Fourier refinement patching.')

ULTRA_PKG_DIR = Path(spec.origin).parent
CONV_PY = ULTRA_PKG_DIR / 'nn' / 'modules' / 'conv.py'
MODULES_INIT_PY = ULTRA_PKG_DIR / 'nn' / 'modules' / '__init__.py'
TASKS_PY = ULTRA_PKG_DIR / 'nn' / 'tasks.py'
MODEL_YAML_DIR = EXPERIMENT_ROOT / 'model_yamls'
MODEL_YAML_DIR.mkdir(parents=True, exist_ok=True)

print('Ultralytics package dir:', ULTRA_PKG_DIR)


def backup_once(path):
    backup = path.with_suffix(path.suffix + '.bak_fourier_refinement_no_aug')
    if not backup.exists():
        shutil.copy2(path, backup)
        print('Backup created:', backup)


for patch_path in [CONV_PY, MODULES_INIT_PY, TASKS_PY]:
    if not patch_path.exists():
        raise FileNotFoundError(patch_path)
    backup_once(patch_path)


FOURIER_REFINEMENT_CODE = '''

# Custom learnable Fourier feature modules for shrimp ONLY_fourier experiments.
class FDDEM(nn.Module):
    # Frequency-domain detail enhancement on feature maps.
    # YOLO-compatible version of the SEP-YOLO FDDEM idea:
    # rFFT over feature maps, learnable complex gains, radial frequency bands,
    # and a residual gate initialized near identity.

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


class ShrimpFEM(nn.Module):
    # MacFormer-style frequency enhancement block adapted to YOLO feature maps.
    # It fuses a spatial branch, a low-frequency branch, and a high-frequency
    # residual branch before the Segment head.

    def __init__(self, c1, c2=None, num_bands=2, init_gamma=0.01):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.spatial = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(c2 * 3, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def _lowpass_mask(self, height, width, device, dtype):
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32).view(1, width // 2 + 1)
        radius = torch.sqrt(fy * fy + fx * fx)
        mask = torch.exp(-(radius * radius) / (2.0 * 0.18 * 0.18))
        return mask.to(dtype=dtype).view(1, 1, height, width // 2 + 1)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        dtype = x.dtype
        freq = torch.fft.rfft2(x.float(), norm='ortho')
        low_mask = self._lowpass_mask(height, width, x.device, freq.real.dtype)
        low = torch.fft.irfft2(freq * low_mask, s=(height, width), norm='ortho').to(dtype=dtype)
        high = (x - low).to(dtype=dtype)
        detail = self.mix(torch.cat([self.spatial(x), low, high], dim=1))
        return x + self.gamma.to(dtype=dtype) * detail


class ShrimpFRF(nn.Module):
    # FRBNet-inspired frequency ratio filter for illumination/background robustness.
    # It uses the ratio between high-frequency residuals and low-frequency content
    # to learn a conservative residual correction.

    def __init__(self, c1, c2=None, num_bands=2, init_gamma=0.01):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.ratio_filter = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def _lowpass_mask(self, height, width, device, dtype):
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32).view(1, width // 2 + 1)
        radius = torch.sqrt(fy * fy + fx * fx)
        mask = torch.exp(-(radius * radius) / (2.0 * 0.16 * 0.16))
        return mask.to(dtype=dtype).view(1, 1, height, width // 2 + 1)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        dtype = x.dtype
        freq = torch.fft.rfft2(x.float(), norm='ortho')
        low_mask = self._lowpass_mask(height, width, x.device, freq.real.dtype)
        low = torch.fft.irfft2(freq * low_mask, s=(height, width), norm='ortho').to(dtype=dtype)
        high = (x - low).to(dtype=dtype)
        ratio = high / (low.abs() + 1e-3)
        ratio = ratio.clamp(min=-3.0, max=3.0)
        detail = self.ratio_filter(ratio).to(dtype=dtype)
        return x + self.gamma.to(dtype=dtype) * detail
'''

conv_text = CONV_PY.read_text(encoding='utf-8')
if 'class ShrimpFEM(nn.Module)' not in conv_text or 'class ShrimpFRF(nn.Module)' not in conv_text:
    if 'class FDDEM(nn.Module)' in conv_text:
        # Keep an earlier FDDEM patch if present, but still add the new FEM/FRF modules.
        extra_code = FOURIER_REFINEMENT_CODE.split('class ShrimpFEM(nn.Module):', 1)[1]
        conv_text = conv_text.rstrip() + '\n\nclass ShrimpFEM(nn.Module):' + extra_code + '\n'
    else:
        conv_text = conv_text.rstrip() + FOURIER_REFINEMENT_CODE + '\n'
    CONV_PY.write_text(conv_text, encoding='utf-8')
    py_compile.compile(str(CONV_PY), doraise=True)
    print('Patched conv.py with Fourier refinement modules.')
else:
    print('conv.py already contains Fourier refinement modules; leaving existing patch in place.')


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


CUSTOM_MODULES = ['FDDEM', 'ShrimpFEM', 'ShrimpFRF']

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
print('Patched modules __init__.py for Fourier refinement exports.')

tasks_text = TASKS_PY.read_text(encoding='utf-8')
tasks_text, ok = add_names_to_parenthesized_import(tasks_text, 'from ultralytics.nn.modules import (', CUSTOM_MODULES)
if not ok:
    raise RuntimeError('Could not locate ultralytics.nn.modules import block in tasks.py')
tasks_text = add_names_to_base_modules(tasks_text, CUSTOM_MODULES)
TASKS_PY.write_text(tasks_text, encoding='utf-8')
ast.parse(TASKS_PY.read_text(encoding='utf-8'))
print('Patched tasks.py for Fourier refinement YAML parsing.')

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
{refinement_layers}
'''

REFINEMENT_YAMLS = {
    'g1_fddem_p3p4p5_identity': '''  # Current best 017-style FDDEM on P3/P4/P5.
  - [16, 1, FDDEM, [256, 2, 0.00]]
  - [19, 1, FDDEM, [512, 2, 0.00]]
  - [22, 1, FDDEM, [1024, 2, 0.00]]
  - [[23, 24, 25], 1, Segment, [nc, 32, 256]]''',
    'g2_fddem_p3p4p5_conservative': '''  # Conservative one-band FDDEM with a small nonzero residual gate.
  - [16, 1, FDDEM, [256, 1, 0.01]]
  - [19, 1, FDDEM, [512, 1, 0.01]]
  - [22, 1, FDDEM, [1024, 1, 0.01]]
  - [[23, 24, 25], 1, Segment, [nc, 32, 256]]''',
    'g3_fddem_p4p5_only': '''  # Avoid P3 low-level noise amplification; refine P4/P5 only.
  - [19, 1, FDDEM, [512, 2, 0.00]]
  - [22, 1, FDDEM, [1024, 2, 0.00]]
  - [[16, 23, 24], 1, Segment, [nc, 32, 256]]''',
    'g4_fem_p3_proto': '''  # MacFormer-style FEM on the high-resolution mask/proto feature only.
  - [16, 1, ShrimpFEM, [256, 2, 0.01]]
  - [[23, 19, 22], 1, Segment, [nc, 32, 256]]''',
    'g5_frf_p3p4p5': '''  # FRBNet-inspired frequency-ratio filtering on P3/P4/P5.
  - [16, 1, ShrimpFRF, [256, 2, 0.01]]
  - [19, 1, ShrimpFRF, [512, 2, 0.01]]
  - [22, 1, ShrimpFRF, [1024, 2, 0.01]]
  - [[23, 24, 25], 1, Segment, [nc, 32, 256]]''',
}

REFINEMENT_YAML_PATHS = {}
for key, layers in REFINEMENT_YAMLS.items():
    yaml_path = MODEL_YAML_DIR / f'yolo11n-seg-{key}.yaml'
    yaml_path.write_text(BASE_YAML_TEMPLATE.format(refinement_layers=layers), encoding='utf-8')
    REFINEMENT_YAML_PATHS[key] = yaml_path
    print('Created Fourier refinement model YAML:', yaml_path)

import ultralytics
from ultralytics import YOLO

print('Ultralytics version:', ultralytics.__version__)
print('Expected version:', PINNED_ULTRALYTICS_VERSION)
assert ultralytics.__version__ == PINNED_ULTRALYTICS_VERSION, 'Ultralytics version mismatch.'

for yaml_key, yaml_path in REFINEMENT_YAML_PATHS.items():
    _smoke_model = YOLO(str(yaml_path))
    print('Fourier refinement YAML smoke build OK:', yaml_key, yaml_path.name)
    del _smoke_model
""".strip() + "\n"


def main() -> None:
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None

    nb["cells"][0]["source"] = as_source(
        """# ONLY Fourier Path 18 - Feature Refinement Candidates, No Augmentation

This notebook tests next-step Fourier feature-refinement candidates after Path 17 showed that Fourier helps under strict no augmentation, but fixed image high-pass can increase healthy false positives.

Strict contract:

- Kaggle-ready Roboflow download
- stratified grouped-specimen split, seed 42
- `yolo11n-seg.pt` family
- no explicit YOLO augmentation
- hidden/default Ultralytics Albumentations hook disabled for every row
- no random Fourier train-copy augmentation
- no fixed image preprocessing in this notebook
- feature-domain Fourier modules only

Candidate families:

- current FDDEM P3/P4/P5 reference from Path 17
- conservative FDDEM with one band and a small residual gate
- P4/P5-only FDDEM to avoid low-level P3 noise amplification
- MacFormer-style frequency enhancement on the high-resolution mask/proto feature
- FRBNet-inspired frequency-ratio filtering for illumination/background robustness
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
- generated Fourier-refinement model YAMLs under `model_yamls/`
- selected `weights/best.pt`
- selected run `results.csv`
- `{OUTPUT_STEM}_outputs.zip`

Rank rows by healthy-aware labeled test score first, then labeled-only test mask mAP50. A candidate is not promoted if it gains mAP50 by creating excessive healthy false positives.
"""
    )
    nb["cells"][3]["source"] = as_source(
        f"""# Run controls for ONLY_fourier path 18.
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

    install_src = "".join(nb["cells"][5]["source"])
    install_src = install_src.replace(
        "import ultralytics\nfrom ultralytics import YOLO\n\nprint('Ultralytics version:', ultralytics.__version__)\nprint('Expected version:', PINNED_ULTRALYTICS_VERSION)\nassert ultralytics.__version__ == PINNED_ULTRALYTICS_VERSION, 'Ultralytics version mismatch.'",
        "print('Ultralytics installed version:', installed_version('ultralytics'))\nprint('Expected version:', PINNED_ULTRALYTICS_VERSION)\nassert installed_version('ultralytics') == PINNED_ULTRALYTICS_VERSION, 'Ultralytics version mismatch.'",
    )
    nb["cells"][5]["source"] = as_source(install_src)

    patch_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "fourier-refinement-patch-cell",
        "metadata": {},
        "outputs": [],
        "source": as_source(FOURIER_REFINEMENT_PATCH_CELL),
    }
    nb["cells"].insert(6, patch_cell)

    helper_idx = 18
    helper_src = "".join(nb["cells"][helper_idx]["source"])

    helper_src = replace_assignment_block(
        helper_src,
        "ABLATION_MODES",
        """FOURIER_REFINEMENT_MODES = [
    {
        'mode': 'G0',
        'enabled': True,
        'fourier_enabled': False,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': False,
        'default_hook_enabled': False,
        'key': 'G0_no_fourier_no_aug_hook_off_baseline',
        'name': 'G0 | no Fourier | no aug | hook off | baseline YOLO11n-seg',
        'model_yaml_key': None,
        'model_yaml': None,
        'model_variant': 'baseline',
        'fourier_module_policy': 'none',
    },
    {
        'mode': 'G1',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'G1_fddem_p3p4p5_identity_no_aug_hook_off',
        'name': 'G1 | FDDEM P3/P4/P5 identity-gated | no aug | hook off',
        'model_yaml_key': 'g1_fddem_p3p4p5_identity',
        'model_yaml': str(REFINEMENT_YAML_PATHS['g1_fddem_p3p4p5_identity']),
        'model_variant': 'fddem_pyramid',
        'fourier_module_policy': 'fddem_p3p4p5_num_bands2_gamma0_identity',
    },
    {
        'mode': 'G2',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'G2_fddem_p3p4p5_conservative_no_aug_hook_off',
        'name': 'G2 | conservative FDDEM P3/P4/P5 one-band gamma0.01 | no aug | hook off',
        'model_yaml_key': 'g2_fddem_p3p4p5_conservative',
        'model_yaml': str(REFINEMENT_YAML_PATHS['g2_fddem_p3p4p5_conservative']),
        'model_variant': 'fddem_conservative',
        'fourier_module_policy': 'fddem_p3p4p5_one_band_gamma0p01',
    },
    {
        'mode': 'G3',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'G3_fddem_p4p5_only_no_aug_hook_off',
        'name': 'G3 | FDDEM P4/P5 only | no aug | hook off',
        'model_yaml_key': 'g3_fddem_p4p5_only',
        'model_yaml': str(REFINEMENT_YAML_PATHS['g3_fddem_p4p5_only']),
        'model_variant': 'fddem_p4p5',
        'fourier_module_policy': 'fddem_p4p5_num_bands2_gamma0_identity',
    },
    {
        'mode': 'G4',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'G4_fem_p3_proto_no_aug_hook_off',
        'name': 'G4 | MacFormer-style FEM on P3 mask/proto feature | no aug | hook off',
        'model_yaml_key': 'g4_fem_p3_proto',
        'model_yaml': str(REFINEMENT_YAML_PATHS['g4_fem_p3_proto']),
        'model_variant': 'fem_p3_proto',
        'fourier_module_policy': 'shrimp_fem_p3_proto_gamma0p01',
    },
    {
        'mode': 'G5',
        'enabled': True,
        'fourier_enabled': True,
        'image_fourier_enabled': False,
        'feature_fourier_enabled': True,
        'default_hook_enabled': False,
        'key': 'G5_frf_p3p4p5_no_aug_hook_off',
        'name': 'G5 | FRBNet-style frequency-ratio filter P3/P4/P5 | no aug | hook off',
        'model_yaml_key': 'g5_frf_p3p4p5',
        'model_yaml': str(REFINEMENT_YAML_PATHS['g5_frf_p3p4p5']),
        'model_variant': 'frf_pyramid',
        'fourier_module_policy': 'shrimp_frf_p3p4p5_gamma0p01',
    },
]

EXPERIMENTS = [mode for mode in FOURIER_REFINEMENT_MODES if mode.get('enabled', True)]""",
    )
    helper_src = helper_src.replace(
        "\n\nEXPERIMENTS = [mode for mode in ABLATION_MODES if mode.get('enabled', True)]",
        "",
    )

    helper_src = replace_block(
        helper_src,
        "LIGHT_AUG_TRAIN_ARGS = {",
        "\n\ndef fourier_device():",
        """STRICT_NO_AUG_KEYS = [
    'erasing', 'mosaic', 'mixup', 'cutmix', 'copy_paste', 'fliplr', 'flipud',
    'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear',
    'perspective', 'bgr', 'close_mosaic',
]


def assert_strict_no_augmentation(train_args, exp):
    if exp.get('default_hook_enabled'):
        raise AssertionError(f"{exp['mode']} tried to enable the hidden/default Albumentations hook.")
    if train_args.get('auto_augment', None) is not None:
        raise AssertionError(f"{exp['mode']} auto_augment must stay None.")
    if bool(train_args.get('multi_scale', False)):
        raise AssertionError(f"{exp['mode']} multi_scale must stay False.")
    nonzero = {key: train_args.get(key) for key in STRICT_NO_AUG_KEYS if float(train_args.get(key, 0.0) or 0.0) != 0.0}
    if nonzero:
        raise AssertionError(f"{exp['mode']} has nonzero augmentation knobs: {nonzero}")
    return True


def train_args_for_mode(exp):
    train_args = dict(NO_AUG_TRAIN_ARGS)
    assert_strict_no_augmentation(train_args, exp)
    return train_args


def augmentation_policy_name(exp):
    return 'none_all_yolo_aug_zero_hidden_hook_off'


def model_yaml_for_mode(exp):
    value = exp.get('model_yaml')
    return str(value) if value else None


def build_yolo_for_mode(exp):
    model_yaml = model_yaml_for_mode(exp)
    if model_yaml:
        model = YOLO(model_yaml)
        model.load(YOLO_MODEL)
        return model
    return YOLO(YOLO_MODEL)


""",
    )

    helper_src = helper_src.replace("if exp.get('fourier_enabled'):", "if exp.get('image_fourier_enabled'):")
    helper_src = helper_src.replace(
        "print('Fourier disabled for this mode; dataset images remain original.')",
        "print('Fixed image Fourier preprocessing disabled for this mode; dataset images remain original.')",
    )
    helper_src = helper_src.replace(
        "hook_disabled = not bool(exp.get('default_hook_enabled'))\n    configure_ultralytics_albumentations_hook(disable=hook_disabled)\n    yolo = YOLO(YOLO_MODEL)",
        "hook_disabled = True\n    if exp.get('default_hook_enabled'):\n        raise AssertionError('Default/hidden Albumentations hook must stay disabled in this notebook.')\n    configure_ultralytics_albumentations_hook(disable=True)\n    yolo = build_yolo_for_mode(exp)\n    current_params_million = round(sum(p.numel() for p in yolo.model.parameters()) / 1_000_000, 3)",
    )
    helper_src = helper_src.replace(
        "'training_seed': training_seed, 'fourier_enabled': bool(exp.get('fourier_enabled')),\n        'clean_light_aug_enabled': bool(exp.get('clean_light_aug_enabled')), 'default_hook_enabled': bool(exp.get('default_hook_enabled')),\n        'baseline_augmentation_policy': augmentation_policy_name(exp), 'hidden_albumentations_disabled': hook_disabled,\n        'preprocessing_policy': 'fourier_highpass' if exp.get('fourier_enabled') else 'none',\n        'fourier_policy': 'highpass_s50_a0p10_all_splits' if exp.get('fourier_enabled') else 'none',\n        'fourier_sigma': FOURIER_SIGMA if exp.get('fourier_enabled') else None, 'fourier_alpha': FOURIER_ALPHA if exp.get('fourier_enabled') else None,",
        "'training_seed': training_seed, 'fourier_enabled': bool(exp.get('fourier_enabled')),\n        'image_fourier_enabled': bool(exp.get('image_fourier_enabled')), 'feature_fourier_enabled': bool(exp.get('feature_fourier_enabled')),\n        'clean_light_aug_enabled': False, 'default_hook_enabled': False,\n        'model_variant': exp.get('model_variant', 'baseline'), 'model_yaml_key': exp.get('model_yaml_key'), 'model_yaml': model_yaml_for_mode(exp),\n        'fourier_module_policy': exp.get('fourier_module_policy', 'none'), 'params_million': current_params_million,\n        'baseline_augmentation_policy': augmentation_policy_name(exp), 'hidden_albumentations_disabled': hook_disabled,\n        'preprocessing_policy': 'none',\n        'fourier_policy': exp.get('fourier_module_policy', 'none') if exp.get('feature_fourier_enabled') else 'none',\n        'fourier_sigma': None, 'fourier_alpha': None,",
    )
    helper_src = helper_src.replace(
        "train_args_json = REPORT_DIR / 'train_args_factorial_A_H_highpass_s50_a0p10.json'\ntrain_args_payload = {'fourier_sigma': FOURIER_SIGMA, 'fourier_alpha': FOURIER_ALPHA, 'no_aug_train_args': NO_AUG_TRAIN_ARGS, 'light_aug_train_args': LIGHT_AUG_TRAIN_ARGS, 'ablation_modes': ABLATION_MODES, 'enabled_modes': EXPERIMENTS, 'train_epochs_requested': TRAIN_EPOCHS, 'train_patience': TRAIN_PATIENCE}\ntrain_args_json.write_text(json.dumps(train_args_payload, indent=2), encoding='utf-8')\nprint('Saved factorial A-H train args:', train_args_json)",
        f"train_args_json = REPORT_DIR / 'train_args_{OUTPUT_STEM}.json'\ntrain_args_payload = {{'strict_no_aug_train_args': NO_AUG_TRAIN_ARGS, 'fourier_refinement_modes': FOURIER_REFINEMENT_MODES, 'enabled_mode_keys': [m['key'] for m in EXPERIMENTS], 'refinement_yaml_paths': {{k: str(v) for k, v in REFINEMENT_YAML_PATHS.items()}}, 'train_epochs_requested': TRAIN_EPOCHS, 'train_patience': TRAIN_PATIENCE}}\ntrain_args_json.write_text(json.dumps(train_args_payload, indent=2), encoding='utf-8')\nprint('Saved strict no-aug Fourier refinement train args:', train_args_json)",
    )
    nb["cells"][helper_idx]["source"] = as_source(helper_src)

    run_idx = 20
    run_src = "".join(nb["cells"][run_idx]["source"])
    run_src = run_src.replace("EXPERIMENTS = [mode for mode in ABLATION_MODES if mode.get('enabled', True)]\n", "")
    run_src = run_src.replace("print('Enabled A-H modes:', [mode['mode'] for mode in EXPERIMENTS])", "print('Enabled strict no-aug Fourier refinement modes:', [mode['mode'] for mode in EXPERIMENTS])")
    run_src = run_src.replace("partial_csv = REPORT_DIR / 'only_fourier_factorial_A_H_highpass_s50_a0p10_partial.csv'", f"partial_csv = REPORT_DIR / '{OUTPUT_STEM}_partial.csv'")
    run_src = run_src.replace("summary_csv = REPORT_DIR / 'only_fourier_factorial_A_H_highpass_s50_a0p10_summary.csv'", f"summary_csv = REPORT_DIR / '{OUTPUT_STEM}_summary.csv'")
    run_src = run_src.replace("print(f'Saved baseline summary CSV: {summary_csv}')", "print(f'Saved strict no-aug Fourier refinement summary CSV: {summary_csv}')")
    run_src = run_src.replace("meta['params_million'] = params_million", "meta['base_params_million'] = params_million")
    nb["cells"][run_idx]["source"] = as_source(run_src)

    paper_idx = 22
    paper_src = "".join(nb["cells"][paper_idx]["source"])
    paper_src = paper_src.replace("'fourier_enabled',\n", "'fourier_enabled',\n    'image_fourier_enabled',\n    'feature_fourier_enabled',\n    'model_variant',\n    'model_yaml_key',\n    'model_yaml',\n    'fourier_module_policy',\n")
    paper_src = paper_src.replace("'default_hook_enabled',\n", "'default_hook_enabled',\n    'base_params_million',\n")
    paper_src = paper_src.replace("paper_table_csv = REPORT_DIR / 'only_fourier_factorial_A_H_highpass_s50_a0p10_paper_row.csv'", f"paper_table_csv = REPORT_DIR / '{OUTPUT_STEM}_paper_row.csv'")
    paper_src = paper_src.replace("print(f'Saved compact baseline paper row: {paper_table_csv}')", "print(f'Saved compact strict no-aug Fourier refinement paper row: {paper_table_csv}')")
    paper_src = paper_src.replace("print('A-H factorial high-pass s50 a0.10 grouped-specimen checkpoint by validation healthy-aware score:')", "print('Strict no-aug Fourier feature-refinement grouped-specimen checkpoint by validation healthy-aware score:')")
    nb["cells"][paper_idx]["source"] = as_source(paper_src)

    final_idx = 24
    final_src = f"""print('Download these Kaggle/Colab outputs after training:')\nprint(f'1. Full Fourier refinement summary CSV: {{REPORT_DIR / \"{OUTPUT_STEM}_summary.csv\"}}')\nprint(f'2. Compact paper row CSV: {{REPORT_DIR / \"{OUTPUT_STEM}_paper_row.csv\"}}')\nprint(f'3. Partial CSV if interrupted: {{REPORT_DIR / \"{OUTPUT_STEM}_partial.csv\"}}')\nprint(f'4. Split manifests: {{REPORT_DIR / \"split_manifests\"}}')\nprint(f'5. Generated model YAMLs: {{EXPERIMENT_ROOT / \"model_yamls\"}}')\nprint(f'6. Selected run folder and checkpoint from {{RUNS_DIR}}')\nprint('')\nprint('Selected strict no-aug Fourier refinement checkpoint:')\nif 'fair_rows' in globals() and not fair_rows.empty:\n    display(fair_rows.sort_values('healthy_aware_labeled_val_mask_map50', ascending=False)[['mode', 'model_variant', 'fourier_module_policy', 'healthy_aware_labeled_val_mask_map50', 'best_pt', 'run_path']].head(1))\nelse:\n    print('No fair grouped-specimen row available yet.')\n\nprint(f'7. Train args: {{REPORT_DIR / \"train_args_{OUTPUT_STEM}.json\"}}')\n\nimport shutil\narchive_path = shutil.make_archive(str(WORK_DIR / '{OUTPUT_STEM}_outputs'), 'zip', root_dir=str(EXPERIMENT_ROOT))\nprint(f'8. Zip archive: {{archive_path}}')\ntry:\n    from google.colab import files\n    files.download(archive_path)\nexcept Exception:\n    print('Automatic browser download is only available in Colab. On Kaggle, download the archive from the working directory.')\n"""
    nb["cells"][final_idx]["source"] = as_source(final_src)

    OUTPUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
