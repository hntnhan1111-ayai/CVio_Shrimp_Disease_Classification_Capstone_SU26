"""Notebook generation helpers for the research attention modules."""

from __future__ import annotations

from pathlib import Path
import re

from .yaml_utils import ResearchAttentionExperiment
from .yaml_utils import build_model_dict


BASELINE_NOTEBOOK = Path("yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb")


def make_intro_markdown(exp: ResearchAttentionExperiment) -> str:
    return f"""# {exp.display_name}

This notebook is derived from the clean leakage-safe YOLO11n-seg baseline and
uses the same self-contained style as `yolov11n_simam_NhomB`: custom attention
classes, Ultralytics parser patching, model YAML creation, and sanity checking
are all defined inside this notebook.
It keeps the Roboflow dataset path, shrimp-grouped no-leakage split, data YAML,
class names, seed, training configuration, validation/test evaluation,
healthy false-positive checks, and report/export flow from the baseline.

- Python class: `{exp.class_name}`
- Placement: P4-only before the Segment head
- Model YAML: created by this notebook at runtime
- Experiment name: `{exp.folder}`
- Idea: {exp.idea}
- Expected benefit: {exp.expected}
- Main risk: {exp.risk}
- Metrics to monitor: {exp.metrics}

Do not train this notebook if the sanity-check cell fails.
"""


def _attention_module_source() -> str:
    source = (Path(__file__).resolve().parent / "attention_modules.py").read_text(encoding="utf-8")
    source = source.replace("from __future__ import annotations\n\n", "")
    return source


def make_register_cell(exp: ResearchAttentionExperiment) -> str:
    attention_source = _attention_module_source()
    return f"""# -- Patch research attention modules into ultralytics namespace -----------------
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

VENDORED_ULTRALYTICS = Path.cwd() / 'ultralytics'
if (VENDORED_ULTRALYTICS / 'ultralytics' / '__init__.py').exists() and str(VENDORED_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(VENDORED_ULTRALYTICS))

loaded_ultralytics = sys.modules.get('ultralytics')
if loaded_ultralytics is not None and not hasattr(loaded_ultralytics, '__version__'):
    for module_name in list(sys.modules):
        if module_name == 'ultralytics' or module_name.startswith('ultralytics.'):
            del sys.modules[module_name]

if importlib.util.find_spec('ultralytics') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'])

{attention_source}

EXPERIMENT_NAME = "{exp.folder}"

def register_research_attention_inline():
    import ultralytics.nn.modules as nn_modules
    import ultralytics.nn.modules.conv as conv_module
    import ultralytics.nn.tasks as tasks_module

    for module_cls in RESEARCH_ATTENTION_MODULES:
        setattr(tasks_module, module_cls.__name__, module_cls)
        setattr(nn_modules, module_cls.__name__, module_cls)
        setattr(conv_module, module_cls.__name__, module_cls)

    nn_modules.__all__ = list(
        set(list(getattr(nn_modules, '__all__', [])) + module_name_list(RESEARCH_ATTENTION_MODULES))
    )
    tasks_module.RESEARCH_ATTENTION_MODULES = RESEARCH_ATTENTION_MODULES

    if getattr(tasks_module, '_shrimp_research_attention_inline_patched', False):
        return

    source = inspect.getsource(tasks_module.parse_model)
    insert = \"\"\"        elif m in RESEARCH_ATTENTION_MODULES:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
\"\"\"
    if 'elif m in RESEARCH_ATTENTION_MODULES:' not in source:
        markers = (
            \"        elif m in frozenset(\\n            {{\\n                Detect,\",
            \"        elif m is SemanticSegment:\",
            \"        elif m in frozenset({{TorchVision, Index}}):\",
        )
        for marker in markers:
            if marker in source:
                source = source.replace(marker, insert + marker, 1)
                break
        else:
            raise RuntimeError('Could not patch ultralytics.nn.tasks.parse_model for research attention modules.')

    exec(compile(source, '<research_attention_inline_parse_model>', 'exec'), tasks_module.__dict__)
    tasks_module._shrimp_research_attention_inline_patched = True


register_research_attention_inline()
print('Research attention modules registered inline:', ', '.join(module_name_list()))
print('Experiment name:', EXPERIMENT_NAME)
"""


def make_yaml_and_sanity_cell(exp: ResearchAttentionExperiment) -> str:
    import yaml

    yaml_text = yaml.safe_dump(build_model_dict(exp), sort_keys=False)
    return f'''# -- Create model YAML inside the notebook and run a build/shape sanity check ----
import math
from pathlib import Path

import torch
from ultralytics import YOLO

YAML_CONTENT = {yaml_text!r}

yaml_dir = Path.cwd() / 'custom_attention_research_modules' / EXPERIMENT_NAME / 'generated_yamls'
yaml_dir.mkdir(parents=True, exist_ok=True)
yaml_path = yaml_dir / '{exp.yaml_name}'
yaml_path.write_text(YAML_CONTENT, encoding='utf-8')

MODEL_YAML = str(yaml_path)
print('YAML written to:', yaml_path)


def _flatten_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _flatten_tensors(item)


def sanity_check_yolo_seg_model_inline(model_yaml, imgsz=640, device=None):
    register_research_attention_inline()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(str(model_yaml))
    yolo.model.to(device).eval()
    print('Model summary:')
    try:
        yolo.model.info(verbose=True)
    except TypeError:
        yolo.model.info()

    dummy = torch.zeros(1, 3, int(imgsz), int(imgsz), device=device)
    with torch.no_grad():
        output = yolo.model(dummy)

    tensors = list(_flatten_tensors(output))
    for idx, tensor in enumerate(tensors):
        if torch.isnan(tensor.detach()).any():
            raise RuntimeError(f'NaN detected in output tensor #{{idx}} with shape {{tuple(tensor.shape)}}')

    custom_names = {{cls.__name__ for cls in RESEARCH_ATTENTION_MODULES}}
    found = []
    for module in yolo.model.modules():
        name = module.__class__.__name__
        if name in custom_names and name not in found:
            found.append(name)
    if not found:
        raise RuntimeError('No research attention module found in the built model.')

    params = sum(p.numel() for p in yolo.model.parameters())
    print('Custom attention modules found:', ', '.join(found))
    print('Checked output tensors:', len(tensors))
    print(f'Total parameters: {{params:,}}')
    print('PASS:', model_yaml)
    return {{'status': 'PASS', 'custom_modules': found, 'params': params}}


sanity_check_yolo_seg_model_inline(MODEL_YAML, imgsz=640)
'''


def _patch_code_source(source: str, exp: ResearchAttentionExperiment) -> str:
    source = re.sub(r"YOLO_MODELS = \[\n.*?\n\]", "YOLO_MODELS = [MODEL_YAML]", source, flags=re.S)
    source = source.replace("YOLO_MODEL = YOLO_MODELS[0]", "YOLO_MODEL = MODEL_YAML")
    source = source.replace("MODEL_STEM = Path(YOLO_MODEL).stem", "MODEL_STEM = EXPERIMENT_NAME")
    source = source.replace("RUN_BASE_NAME = f'{MODEL_STEM}_shrimp_seg_clean_baseline'", "RUN_BASE_NAME = EXPERIMENT_NAME")
    source = source.replace(
        "RUN_BASE_NAME = f'{MODEL_STEM}_shrimp_seg_clean_light_aug_baseline'",
        "RUN_BASE_NAME = EXPERIMENT_NAME",
    )
    source = source.replace(
        "EXPERIMENT_ROOT = Path('/kaggle/working/shrimp_yolo_seg_clean_light_aug_baseline')",
        "WORKING_ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()\n"
        "EXPERIMENT_ROOT = WORKING_ROOT / 'custom_attention_research_modules' / EXPERIMENT_NAME / 'outputs'",
    )
    source = source.replace(
        "RUNS_DIR = Path('/kaggle/working/runs/segment')",
        "RUNS_DIR = WORKING_ROOT / 'runs' / 'custom_attention_research_modules'",
    )
    source = source.replace(
        "def write_data_yaml(dataset_dir, yaml_path, val_dir='valid', test_dir='test'):\n"
        "    with open(data_yaml_path, 'r') as f:",
        "def write_data_yaml(dataset_dir, yaml_path, val_dir='valid', test_dir='test'):\n"
        "    dataset_dir = Path(dataset_dir).resolve()\n"
        "    with open(data_yaml_path, 'r') as f:",
    )
    source = source.replace("'key': 'clean_light_aug_baseline'", "'key': EXPERIMENT_NAME")
    source = source.replace(
        "'name': 'Clean Baseline - Grouped Stratified Light YOLO Augmentation'",
        f"'name': 'YOLO11n-seg + {exp.display_name}'",
    )
    source = source.replace(
        "run_name = f'{RUN_BASE_NAME}_{exp[\"key\"]}' + ('_smoke' if SMOKE_RUN else '')",
        "run_name = EXPERIMENT_NAME + ('_smoke' if SMOKE_RUN else '')",
    )
    source = source.replace(
        "partial_df.to_csv(REPORT_DIR / 'seg_clean_light_aug_baseline_partial.csv', index=False)",
        "partial_df.to_csv(REPORT_DIR / f'{EXPERIMENT_NAME}_results_partial.csv', index=False)",
    )
    source = source.replace(
        "summary_csv = REPORT_DIR / 'seg_clean_light_aug_baseline_summary.csv'",
        "summary_csv = REPORT_DIR / f'{EXPERIMENT_NAME}_results_summary.csv'",
    )
    source = source.replace(
        "results_df = pd.read_csv(REPORT_DIR / 'seg_clean_light_aug_baseline_summary.csv')",
        "results_df = pd.read_csv(REPORT_DIR / f'{EXPERIMENT_NAME}_results_summary.csv')",
    )
    return source


def patch_baseline_notebook(nb, exp: ResearchAttentionExperiment):
    import nbformat

    cells = list(nb.cells)
    cells[0] = nbformat.v4.new_markdown_cell(make_intro_markdown(exp))
    cells.insert(5, nbformat.v4.new_code_cell(make_register_cell(exp)))
    cells.insert(6, nbformat.v4.new_code_cell(make_yaml_and_sanity_cell(exp)))

    for cell in cells:
        if cell.cell_type == "code":
            cell.source = _patch_code_source(cell.source, exp)
            cell.outputs = []
            cell.execution_count = None

    nb.cells = cells
    return nb


def create_notebook_from_baseline(
    baseline_path: str | Path,
    output_path: str | Path,
    exp: ResearchAttentionExperiment,
) -> None:
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
    "make_yaml_and_sanity_cell",
    "patch_baseline_notebook",
]
