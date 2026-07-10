"""Notebook generation helpers for custom attention experiments."""

from __future__ import annotations

from pathlib import Path

from .yaml_utils import build_model_dict


BASELINE_NOTEBOOK = Path(
    "yolov11n_attention/yolov11n_grouped_attention/"
    "aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb"
)

SHARED_DIR = Path(__file__).resolve().parent


def make_intro_markdown(display_name: str, folder: str, idea: str, placement: str, expected: str, risk: str, metrics: str) -> str:
    return f"""# YOLO11n Custom Attention - {display_name}

This notebook is generated from the clean leakage-safe YOLO11n-seg baseline.
It preserves the Roboflow download, grouped shrimp-level split, class names,
training configuration, validation/test evaluation, and healthy-negative
false-positive checks. Only the model YAML and run/output naming are changed.

- Module folder: `{folder}`
- Idea: {idea}
- Placement: {placement}
- Expected benefit: {expected}
- Main risk: {risk}
- Metrics to monitor: {metrics}

Do not train this notebook until the model YAML passes the sanity check cell.
"""


def _attention_module_source() -> str:
    """Return self-contained attention module code for embedding in notebooks."""

    return (SHARED_DIR / "attention_modules.py").read_text(encoding="utf-8")


def make_register_cell(exp) -> str:
    """Create a Kaggle-standalone attention/register/model-yaml cell."""

    import yaml

    model_yaml_text = yaml.safe_dump(build_model_dict(exp), sort_keys=False)
    attention_source = _attention_module_source()
    return f"""{attention_source}

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

if importlib.util.find_spec('ultralytics') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'])

def register_custom_attention():
    import ultralytics.nn.tasks as tasks

    for module_cls in CUSTOM_ATTENTION_MODULES:
        setattr(tasks, module_cls.__name__, module_cls)

    tasks.CUSTOM_ATTENTION_MODULES = CUSTOM_ATTENTION_MODULES

    if getattr(tasks, "_shrimp_custom_attention_parse_patched", False):
        return

    source = inspect.getsource(tasks.parse_model)
    marker = "        elif m in frozenset(\\n            {{\\n                Detect,"
    insert = \"\"\"        elif m in CUSTOM_ATTENTION_MODULES:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
\"\"\"
    if marker not in source:
        raise RuntimeError(
            "Could not patch ultralytics.nn.tasks.parse_model: Detect branch marker not found. "
            f"Custom modules: {{', '.join(module_name_list())}}"
        )

    patched = source.replace(marker, insert + marker, 1)
    exec(compile(patched, "<shrimp_custom_attention_parse_model>", "exec"), tasks.__dict__)
    tasks._shrimp_custom_attention_parse_patched = True


MODEL_YAML = "/kaggle/working/custom_attention_models/{exp.folder}/model.yaml"
MODEL_YAML_TEXT = {model_yaml_text!r}
Path(MODEL_YAML).parent.mkdir(parents=True, exist_ok=True)
Path(MODEL_YAML).write_text(MODEL_YAML_TEXT, encoding="utf-8")

register_custom_attention()
print(f"Registered custom attention modules. MODEL_YAML={{MODEL_YAML}}")
"""


def make_sanity_cell() -> str:
    return """def _custom_modules_in_model(model):
    custom_names = {cls.__name__ for cls in CUSTOM_ATTENTION_MODULES}
    found = []
    for module in model.modules():
        name = module.__class__.__name__
        if name in custom_names and name not in found:
            found.append(name)
    return found


def sanity_check_yolo_seg_model(model_yaml, imgsz=640, device=None):
    from ultralytics import YOLO
    import torch

    register_custom_attention()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(model_yaml)
    yolo.model.to(device)
    yolo.model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        _ = yolo.model(dummy)
    found = _custom_modules_in_model(yolo.model)
    params = sum(p.numel() for p in yolo.model.parameters())
    print(f"Custom attention modules found: {', '.join(found) if found else 'none'}")
    print(f"Total parameters: {params:,}")
    print(f"PASS: {model_yaml}")
    return {"status": "PASS", "custom_modules": found, "params": params}

sanity_check_yolo_seg_model(
    model_yaml=MODEL_YAML,
    imgsz=640,
)
"""


def patch_baseline_notebook(nb, exp):
    """Return a patched copy of the baseline notebook for an experiment."""

    import nbformat

    cells = list(nb.cells)
    cells[0] = nbformat.v4.new_markdown_cell(
        make_intro_markdown(
            display_name=exp.display_name,
            folder=exp.folder,
            idea=exp.idea,
            placement=exp.placement,
            expected=exp.expected,
            risk=exp.risk,
            metrics=exp.metrics,
        )
    )

    register_cell = nbformat.v4.new_code_cell(make_register_cell(exp))
    sanity_cell = nbformat.v4.new_code_cell(make_sanity_cell())
    cells.insert(5, register_cell)
    cells.insert(6, sanity_cell)

    for cell in cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        source = source.replace("YOLO_MODEL = YOLO_MODELS[0]", "YOLO_MODEL = MODEL_YAML")
        source = source.replace("MODEL_STEM = Path(YOLO_MODEL).stem", f"MODEL_STEM = '{exp.folder}'")
        source = source.replace(
            "RUN_BASE_NAME = f'{MODEL_STEM}_shrimp_seg_clean_baseline'",
            "RUN_BASE_NAME = 'yolo11n-seg_shrimp_seg_clean_light_aug_baseline'",
        )
        source = source.replace(
            "RUN_BASE_NAME = f'{MODEL_STEM}_shrimp_seg_clean_light_aug_baseline'",
            "RUN_BASE_NAME = 'yolo11n-seg_shrimp_seg_clean_light_aug_baseline'",
        )
        source = source.replace(
            "EXPERIMENT_ROOT = Path('/kaggle/working/shrimp_yolo_seg_clean_light_aug_baseline')",
            f"EXPERIMENT_ROOT = Path('/kaggle/working/custom_attention_{exp.folder}')",
        )
        source = source.replace(
            "RUNS_DIR = Path('/kaggle/working/runs/segment')",
            "RUNS_DIR = Path('/kaggle/working/runs/custom_attention')",
        )
        source = source.replace(
            "'key': 'clean_light_aug_baseline'",
            f"'key': '{exp.folder}'",
        )
        source = source.replace(
            "'name': 'Clean Baseline - Grouped Stratified Light YOLO Augmentation'",
            f"'name': 'YOLO11n-seg + {exp.display_name}'",
        )
        source = source.replace("for model_name in YOLO_MODELS:", "for model_name in [MODEL_YAML]:")
        cell.source = source
        cell.outputs = []
        cell.execution_count = None

    nb.cells = cells
    return nb


def create_notebook_from_baseline(baseline_path: str | Path, output_path: str | Path, exp) -> None:
    """Create a patched experiment notebook using nbformat."""

    import nbformat

    baseline_path = Path(baseline_path)
    output_path = Path(output_path)
    nb = nbformat.read(baseline_path, as_version=4)
    nb = patch_baseline_notebook(nb, exp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_path)


__all__ = [
    "BASELINE_NOTEBOOK",
    "create_notebook_from_baseline",
    "make_intro_markdown",
    "make_register_cell",
    "make_sanity_cell",
    "patch_baseline_notebook",
]
