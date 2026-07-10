import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "yolov11n_attention" / "yolov11n_grouped_attention"
OUT_DIR = ROOT / "yolov11n_attention" / "att_backbone"

SINGLE_SRC = SOURCE_DIR / "aip491-01-yolo-seg-11n-single-modules (1).ipynb"
COMBINED_SRC = SOURCE_DIR / "aip491-01-yolo-seg-11n-combined-modules.ipynb"

SINGLE_OUT = OUT_DIR / "aip491-01-yolo-seg-11n-single-modules-att-backbone.ipynb"
COMBINED_OUT = OUT_DIR / "aip491-01-yolo-seg-11n-combined-modules-att-backbone.ipynb"


def read_notebook(path):
    return nbformat.read(path, as_version=4)


def write_notebook(path, notebook):
    notebook.setdefault("metadata", {})
    notebook["metadata"].setdefault(
        "kernelspec",
        {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
    )
    notebook["metadata"].setdefault(
        "language_info",
        {
            "name": "python",
            "version": "3.x",
        },
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def clear_outputs(notebook):
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


SETUP_SOURCE = """import importlib.util
import subprocess
import sys

ULTRALYTICS_VERSION = '8.4.62'
subprocess.check_call([
    sys.executable,
    '-m',
    'pip',
    'install',
    '-q',
    f'ultralytics=={ULTRALYTICS_VERSION}',
])

from ultralytics import YOLO
import os
from pathlib import Path

base_path = '/kaggle/working/shrimpDisHandSegV2-1'
data_yaml_path = os.path.join(base_path, 'data.yaml')

# Keep the model name and run name tied together so reports are not mislabeled.
# This baseline model is also used as pretrained initialization for attention models.
YOLO_MODELS = [
    # 'yolov8n-seg.pt',
    # 'yolov8m-seg.pt',
    'yolo11n-seg.pt',
    # 'yolo11m-seg.pt',
    # 'yolo26n-seg.pt',
    # 'yolo26m-seg.pt',
]

YOLO_MODEL = YOLO_MODELS[0]
MODEL_STEM = Path(YOLO_MODEL).stem
RUN_BASE_NAME = f'{MODEL_STEM}_shrimp_seg_clean_baseline'

# Load once here as a smoke check. Training cells instantiate fresh models per experiment.
model = YOLO(YOLO_MODEL)

print(f'Ultralytics version pinned to: {ULTRALYTICS_VERSION}')
print('Configured segmentation models:')
for configured_model in YOLO_MODELS:
    print(f'  - {configured_model}')
print(f'Run name prefix: {RUN_BASE_NAME}')
print(f'base_path: {base_path}')
print(f'data_yaml_path: {data_yaml_path}')
"""


FORCE_PATCHED_PACKAGE_SOURCE = """# =========================================================
# Force using the patched Ultralytics package
# =========================================================
import sys
from pathlib import Path

ULTRALYTICS_VERSION = globals().get('ULTRALYTICS_VERSION', '8.4.62')

# Clear previously imported ultralytics modules so patched files are reloaded.
for module_name in list(sys.modules.keys()):
    if module_name == 'ultralytics' or module_name.startswith('ultralytics.'):
        del sys.modules[module_name]

import ultralytics
from ultralytics import YOLO

ULTRALYTICS_PACKAGE_DIR = Path(ultralytics.__file__).resolve().parent

print('Using patched Ultralytics package from:')
print(ultralytics.__file__)
print('Requested Ultralytics version:', ULTRALYTICS_VERSION)
"""


YAML_CELL_SOURCE = """from pathlib import Path
import yaml

from ultralytics import YOLO
import ultralytics

ULTRALYTICS_PACKAGE_DIR = Path(ultralytics.__file__).resolve().parent
MODEL_CFG_DIR = ULTRALYTICS_PACKAGE_DIR / "cfg/models/11"
MODEL_CFG_DIR.mkdir(parents=True, exist_ok=True)

# New in this notebook: each attention recipe is added to backbone P3/P4/P5.
# The original head attention is kept so results remain comparable with the
# previous single-module and combined-module notebooks.
ADD_BACKBONE_ATTENTION = True
ADD_HEAD_ATTENTION = True

ATTENTION_RECIPES = {
    "simam": [("SimAM", lambda c: [])],
    "ca": [("CoordAtt", lambda c: [c, 32])],
    "eca": [("ECAAttention", lambda c: [3])],
    "cbam": [("CBAMAttention", lambda c: [c, 16, 7])],
    "ema": [("EMAAttention", lambda c: [c, 8])],
    "simam_ca": [("CoordAtt", lambda c: [c, 32]), ("SimAM", lambda c: [])],
    "eca_simam": [("ECAAttention", lambda c: [3]), ("SimAM", lambda c: [])],
    "ca_ema": [("CoordAtt", lambda c: [c, 32]), ("EMAAttention", lambda c: [c, 8])],
    "cbam_simam": [("CBAMAttention", lambda c: [c, 16, 7]), ("SimAM", lambda c: [])],
    "ema_simam_ca": [
        ("CoordAtt", lambda c: [c, 32]),
        ("EMAAttention", lambda c: [c, 8]),
        ("SimAM", lambda c: []),
    ],
}

YAML_NAMES = {
    "simam": "yolo11n-seg-simam-backbone-head.yaml",
    "ca": "yolo11n-seg-ca-backbone-head.yaml",
    "eca": "yolo11n-seg-eca-backbone-head.yaml",
    "cbam": "yolo11n-seg-cbam-backbone-head.yaml",
    "ema": "yolo11n-seg-ema-backbone-head.yaml",
    "simam_ca": "yolo11n-seg-simam-ca-backbone-head.yaml",
    "eca_simam": "yolo11n-seg-eca-simam-backbone-head.yaml",
    "ca_ema": "yolo11n-seg-ca-ema-backbone-head.yaml",
    "cbam_simam": "yolo11n-seg-cbam-simam-backbone-head.yaml",
    "ema_simam_ca": "yolo11n-seg-ema-simam-ca-backbone-head.yaml",
}


def add_attention(section, global_index_fn, from_idx, nominal_channels, recipe_key):
    current = from_idx
    for module_name, args_fn in ATTENTION_RECIPES[recipe_key]:
        section.append([current, 1, module_name, args_fn(nominal_channels)])
        current = global_index_fn()
    return current


def build_backbone_head_cfg(recipe_key):
    backbone = []
    head = []

    def add_backbone(f, n, m, args):
        backbone.append([f, n, m, args])
        return len(backbone) - 1

    def add_head(f, n, m, args):
        head.append([f, n, m, args])
        return len(backbone) + len(head) - 1

    def current_backbone_index():
        return len(backbone) - 1

    def current_head_index():
        return len(backbone) + len(head) - 1

    add_backbone(-1, 1, "Conv", [64, 3, 2])
    add_backbone(-1, 1, "Conv", [128, 3, 2])
    add_backbone(-1, 2, "C3k2", [256, False, 0.25])
    add_backbone(-1, 1, "Conv", [256, 3, 2])
    p3 = add_backbone(-1, 2, "C3k2", [256, False, 0.25])
    if ADD_BACKBONE_ATTENTION:
        p3 = add_attention(backbone, current_backbone_index, p3, 256, recipe_key)

    add_backbone(-1, 1, "Conv", [512, 3, 2])
    p4 = add_backbone(-1, 2, "C3k2", [512, False, 0.25])
    if ADD_BACKBONE_ATTENTION:
        p4 = add_attention(backbone, current_backbone_index, p4, 512, recipe_key)

    add_backbone(-1, 1, "Conv", [1024, 3, 2])
    add_backbone(-1, 2, "C3k2", [1024, True])
    add_backbone(-1, 1, "SPPF", [1024, 5])
    p5 = add_backbone(-1, 2, "C2PSA", [1024])
    if ADD_BACKBONE_ATTENTION:
        p5 = add_attention(backbone, current_backbone_index, p5, 1024, recipe_key)

    add_head(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    add_head([-1, p4], 1, "Concat", [1])
    top_p4 = add_head(-1, 2, "C3k2", [512, False])

    add_head(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    add_head([-1, p3], 1, "Concat", [1])
    out_p3 = add_head(-1, 2, "C3k2", [256, False])

    add_head(-1, 1, "Conv", [256, 3, 2])
    add_head([-1, top_p4], 1, "Concat", [1])
    out_p4 = add_head(-1, 2, "C3k2", [512, False])

    add_head(-1, 1, "Conv", [512, 3, 2])
    add_head([-1, p5], 1, "Concat", [1])
    out_p5 = add_head(-1, 2, "C3k2", [1024, True])

    if ADD_HEAD_ATTENTION:
        out_p3 = add_attention(head, current_head_index, out_p3, 256, recipe_key)
        out_p4 = add_attention(head, current_head_index, out_p4, 512, recipe_key)
        out_p5 = add_attention(head, current_head_index, out_p5, 1024, recipe_key)

    add_head([out_p3, out_p4, out_p5], 1, "Segment", ["nc", 32, 256])

    return {
        "nc": 2,
        "scales": {
            "n": [0.50, 0.25, 1024],
            "s": [0.50, 0.50, 1024],
            "m": [0.50, 1.00, 512],
            "l": [1.00, 1.00, 512],
            "x": [1.00, 1.50, 512],
        },
        "backbone": backbone,
        "head": head,
    }


MODEL_YAML_PATHS = {}
for key, yaml_name in YAML_NAMES.items():
    yaml_path = MODEL_CFG_DIR / yaml_name
    cfg = build_backbone_head_cfg(key)
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    MODEL_YAML_PATHS[key] = str(yaml_path)
    print("Created:", key, "->", yaml_path)

print("\\nSmoke-building generated attention-backbone YAML files...")
for key, yaml_path in MODEL_YAML_PATHS.items():
    model = YOLO(yaml_path)
    print(f"Smoke build OK: {key} ({len(model.model.model)} layers)")
"""


SINGLE_EXPERIMENT_SOURCE = """import sys
from pathlib import Path

EXPERIMENT_ROOT_STR = "/kaggle/working/yolo11n_single_modules_att_backbone_group_run"
RUN_BASE_NAME = "single_modules_att_backbone_group_run"

EXPERIMENTS = [
    {
        "key": "baseline",
        "name": "Clean Baseline - Grouped Stratified Light YOLO Augmentation",
        "model_type": "baseline",
    },
    {
        "key": "simam_backbone",
        "name": "YOLO11n-seg + SimAM Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["simam"],
    },
    {
        "key": "ca_backbone",
        "name": "YOLO11n-seg + Coordinate Attention Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["ca"],
    },
    {
        "key": "eca_backbone",
        "name": "YOLO11n-seg + ECA Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["eca"],
    },
    {
        "key": "cbam_backbone",
        "name": "YOLO11n-seg + CBAM Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["cbam"],
    },
    {
        "key": "ema_backbone",
        "name": "YOLO11n-seg + EMA Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["ema"],
    },
]

REQUIRED_SINGLE_EXPERIMENT_KEYS = [
    "baseline",
    "simam_backbone",
    "ca_backbone",
    "eca_backbone",
    "cbam_backbone",
    "ema_backbone",
]
experiment_by_key = {e["key"]: e for e in EXPERIMENTS}
missing_keys = [key for key in REQUIRED_SINGLE_EXPERIMENT_KEYS if key not in experiment_by_key]
if missing_keys:
    raise RuntimeError(f"Missing required single-module experiments: {missing_keys}")
EXPERIMENTS = [experiment_by_key[key] for key in REQUIRED_SINGLE_EXPERIMENT_KEYS]

print("Single-module attention-backbone experiments:")
print(f"Total experiments to run: {len(EXPERIMENTS)}")
for e in EXPERIMENTS:
    print("-", e["key"], ":", e["name"])
"""


COMBINED_EXPERIMENT_SOURCE = """import sys
from pathlib import Path

EXPERIMENT_ROOT_STR = "/kaggle/working/yolo11n_combined_modules_att_backbone_group_run"
RUN_BASE_NAME = "combined_modules_att_backbone_group_run"

EXPERIMENTS = [
    {
        "key": "baseline",
        "name": "Clean Baseline - Grouped Stratified Light YOLO Augmentation",
        "model_type": "baseline",
    },
    {
        "key": "simam_ca_backbone",
        "name": "YOLO11n-seg + SimAM + CA Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["simam_ca"],
    },
    {
        "key": "eca_simam_backbone",
        "name": "YOLO11n-seg + ECA + SimAM Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["eca_simam"],
    },
    {
        "key": "ca_ema_backbone",
        "name": "YOLO11n-seg + CA + EMA Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["ca_ema"],
    },
    {
        "key": "cbam_simam_backbone",
        "name": "YOLO11n-seg + CBAM + SimAM Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["cbam_simam"],
    },
    {
        "key": "ema_simam_ca_backbone",
        "name": "YOLO11n-seg + EMA + SimAM + CA Backbone+Head",
        "model_type": "attention",
        "yaml": MODEL_YAML_PATHS["ema_simam_ca"],
    },
]

print("Combined-module attention-backbone experiments:")
for e in EXPERIMENTS:
    print("-", e["key"], ":", e["name"])
"""


def patch_attention_cell(source):
    old = """ULTRA_DIR = Path("/kaggle/working/ultralytics")

if not ULTRA_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/ultralytics/ultralytics.git", str(ULTRA_DIR)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(ULTRA_DIR)], check=True)

conv_py = ULTRA_DIR / "ultralytics/nn/modules/conv.py"
init_py = ULTRA_DIR / "ultralytics/nn/modules/__init__.py"
tasks_py = ULTRA_DIR / "ultralytics/nn/tasks.py"
"""
    new = """ULTRALYTICS_VERSION = globals().get('ULTRALYTICS_VERSION', '8.4.62')
subprocess.run([
    sys.executable,
    '-m',
    'pip',
    'install',
    '-q',
    f'ultralytics=={ULTRALYTICS_VERSION}',
], check=True)

for module_name in list(sys.modules.keys()):
    if module_name == 'ultralytics' or module_name.startswith('ultralytics.'):
        del sys.modules[module_name]

import ultralytics

ULTRALYTICS_PACKAGE_DIR = Path(ultralytics.__file__).resolve().parent
conv_py = ULTRALYTICS_PACKAGE_DIR / "nn/modules/conv.py"
init_py = ULTRALYTICS_PACKAGE_DIR / "nn/modules/__init__.py"
tasks_py = ULTRALYTICS_PACKAGE_DIR / "nn/tasks.py"
print("Patching Ultralytics", ULTRALYTICS_VERSION, "at:", ULTRALYTICS_PACKAGE_DIR)
"""
    if old not in source:
        raise RuntimeError("Could not find the Ultralytics clone/install block to patch.")
    source = source.replace(old, new)

    # PyTorch warns on CUDA when AdaptiveMaxPool2d participates in backward while
    # deterministic algorithms are enabled. CBAM only needs global max pooling,
    # so use a deterministic reduction form instead.
    source = source.replace("        self.max_pool = nn.AdaptiveMaxPool2d(1)\n", "")
    source = source.replace(
        "    def forward(self, x):\n"
        "        ca = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))\n",
        "    @staticmethod\n"
        "    def _global_max_pool(x):\n"
        "        return x.flatten(2).max(dim=2, keepdim=True).values.unsqueeze(-1)\n"
        "\n"
        "    def forward(self, x):\n"
        "        ca = self.mlp(self.avg_pool(x)) + self.mlp(self._global_max_pool(x))\n",
    )
    return source


def update_train_cell(source):
    if "workers=0," not in source:
        raise RuntimeError("Could not find workers=0 in train cell.")
    source = source.replace("workers=0,", "workers=8,")
    marker = """# =========================================================
# Run all experiments in this notebook
# =========================================================
experiment_results = []
"""
    replacement = """# =========================================================
# Run all experiments in this notebook
# =========================================================
experiment_keys = [experiment["key"] for experiment in EXPERIMENTS]
print(f"Experiments scheduled ({len(experiment_keys)}): {experiment_keys}")
if len(experiment_keys) <= 1:
    raise RuntimeError(
        "Only one experiment is scheduled. Re-run the experiment selection cell "
        "and confirm all single/combined module keys are listed."
    )
experiment_results = []
"""
    if marker not in source:
        raise RuntimeError("Could not find experiment loop marker in train cell.")
    return source.replace(marker, replacement)


def build_notebook(src_path, out_path, title, experiment_source):
    notebook = read_notebook(src_path)
    notebook = copy.deepcopy(notebook)
    clear_outputs(notebook)

    set_source(
        notebook["cells"][0],
        f"""# {title}

Runs leakage-safe grouped-stratified YOLO11n segmentation experiments with attention added to backbone P3/P4/P5 while preserving the original head attention recipes. The clean baseline row follows `aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb`.
""",
    )
    set_source(notebook["cells"][6], SETUP_SOURCE)
    set_source(notebook["cells"][13], "## 6. Patch attention modules for Ultralytics 8.4.62\n")
    set_source(notebook["cells"][14], patch_attention_cell(source_text(notebook["cells"][14])))
    set_source(notebook["cells"][15], FORCE_PATCHED_PACKAGE_SOURCE)
    set_source(notebook["cells"][16], "## 7. Create attention-backbone YAML files and smoke-build\n")
    set_source(notebook["cells"][17], YAML_CELL_SOURCE)
    set_source(notebook["cells"][18], "## 8. Select attention-backbone experiment group\n")
    set_source(notebook["cells"][19], experiment_source)
    set_source(notebook["cells"][21], update_train_cell(source_text(notebook["cells"][21])))

    write_notebook(out_path, notebook)
    return out_path


def generate_notebooks():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = [
        build_notebook(
            SINGLE_SRC,
            SINGLE_OUT,
            "YOLO11n single attention modules with backbone attention",
            SINGLE_EXPERIMENT_SOURCE,
        ),
        build_notebook(
            COMBINED_SRC,
            COMBINED_OUT,
            "YOLO11n combined attention modules with backbone attention",
            COMBINED_EXPERIMENT_SOURCE,
        ),
    ]
    for path in created:
        print("Created notebook:", path)


def verify_local_smoke_build():
    sys.path.insert(0, str(ROOT / "yolov11n_attention" / "ultralytics"))
    from ultralytics import YOLO
    import yaml

    namespace = {}
    definition_source = YAML_CELL_SOURCE.split("MODEL_YAML_PATHS = {}", 1)[0]
    exec(definition_source, namespace)
    build_cfg = namespace["build_backbone_head_cfg"]
    yaml_names = namespace["YAML_NAMES"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for key, yaml_name in yaml_names.items():
            yaml_path = tmpdir / yaml_name
            yaml_path.write_text(yaml.safe_dump(build_cfg(key), sort_keys=False), encoding="utf-8")
            model = YOLO(str(yaml_path))
            print(f"Local smoke build OK: {key} ({len(model.model.model)} layers)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-local", action="store_true")
    args = parser.parse_args()

    generate_notebooks()
    if args.verify_local:
        verify_local_smoke_build()


if __name__ == "__main__":
    main()
