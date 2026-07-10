import argparse
import copy
import sys
import tempfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "yolov11n_attention" / "att_position_experiments"
TEMPLATE_DIR = ROOT / "yolov11n_attention" / "att_backbone"
SINGLE_TEMPLATE = TEMPLATE_DIR / "aip491-01-yolo-seg-11n-single-modules-att-backbone.ipynb"
COMBINED_TEMPLATE = TEMPLATE_DIR / "aip491-01-yolo-seg-11n-combined-modules-att-backbone.ipynb"


DIRECTIONS = [
    {
        "key": "neck_concat_attention",
        "folder": "01_neck_concat_attention",
        "title": "Neck concat attention",
        "description": "Add attention right after every neck Concat fusion and before the following C3k2.",
        "placement": "neck_concat_all",
    },
    {
        "key": "p3_head_only_attention",
        "folder": "02_p3_head_only_attention",
        "title": "P3 head output attention only",
        "description": "Add attention only on the P3/8 head output before Segment.",
        "placement": "head_p3_only",
    },
    {
        "key": "p4_head_only_attention",
        "folder": "03_p4_head_only_attention",
        "title": "P4 head output attention only",
        "description": "Add attention only on the P4/16 head output before Segment.",
        "placement": "head_p4_only",
    },
    {
        "key": "c2psa_after_attention",
        "folder": "04_c2psa_after_attention",
        "title": "C2PSA semantic bottleneck attention",
        "description": "Add attention after the final C2PSA backbone block only.",
        "placement": "after_c2psa",
    },
    {
        "key": "backbone_p3_only_attention",
        "folder": "05_backbone_p3_only_attention",
        "title": "Backbone P3 attention only",
        "description": "Add attention after the backbone P3/8 C3k2 feature only.",
        "placement": "backbone_p3_only",
    },
    {
        "key": "neck_p3_p4_attention",
        "folder": "06_neck_p3_p4_attention",
        "title": "Top-down neck P3/P4 attention",
        "description": "Add attention after the top-down P4 and P3 Concat fusions; skip P5.",
        "placement": "neck_p3_p4_topdown",
    },
]


def read_notebook(path):
    return nbformat.read(path, as_version=4)


def write_notebook(path, notebook):
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def clear_outputs(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def make_yaml_cell(direction):
    key = direction["key"]
    placement = direction["placement"]
    description = direction["description"]
    return f'''from pathlib import Path
import yaml

from ultralytics import YOLO
import ultralytics

ULTRALYTICS_PACKAGE_DIR = Path(ultralytics.__file__).resolve().parent
MODEL_CFG_DIR = ULTRALYTICS_PACKAGE_DIR / "cfg/models/11"
MODEL_CFG_DIR.mkdir(parents=True, exist_ok=True)

DIRECTION_KEY = "{key}"
PLACEMENT = "{placement}"
PLACEMENT_DESCRIPTION = "{description}"

ATTENTION_RECIPES = {{
    "simam": [("SimAM", lambda c: [])],
    "ca": [("CoordAtt", lambda c: [c, 32])],
    "eca": [("ECAAttention", lambda c: [3])],
    "cbam": [("CBAMAttention", lambda c: [c, 16, 7])],
    "ema": [("EMAAttention", lambda c: [c, 8])],
    # Keep the original best combined order from earlier notebooks: CA -> SimAM.
    "simam_ca": [("CoordAtt", lambda c: [c, 32]), ("SimAM", lambda c: [])],
    "eca_simam": [("ECAAttention", lambda c: [3]), ("SimAM", lambda c: [])],
    "ca_ema": [("CoordAtt", lambda c: [c, 32]), ("EMAAttention", lambda c: [c, 8])],
    "cbam_simam": [("CBAMAttention", lambda c: [c, 16, 7]), ("SimAM", lambda c: [])],
    "ema_simam_ca": [
        ("CoordAtt", lambda c: [c, 32]),
        ("EMAAttention", lambda c: [c, 8]),
        ("SimAM", lambda c: []),
    ],
}}

YAML_NAMES = {{
    recipe_key: f"yolo11n-seg-{{recipe_key}}-{key}.yaml"
    for recipe_key in ATTENTION_RECIPES
}}


def add_attention(section, global_index_fn, from_idx, nominal_channels, recipe_key):
    current = from_idx
    for module_name, args_fn in ATTENTION_RECIPES[recipe_key]:
        section.append([current, 1, module_name, args_fn(nominal_channels)])
        current = global_index_fn()
    return current


def maybe_attention(section, global_index_fn, current, nominal_channels, recipe_key, location):
    if location == PLACEMENT:
        return add_attention(section, global_index_fn, current, nominal_channels, recipe_key)
    return current


def build_cfg(recipe_key):
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
    p3 = maybe_attention(backbone, current_backbone_index, p3, 256, recipe_key, "backbone_p3_only")

    add_backbone(-1, 1, "Conv", [512, 3, 2])
    p4 = add_backbone(-1, 2, "C3k2", [512, False, 0.25])

    add_backbone(-1, 1, "Conv", [1024, 3, 2])
    add_backbone(-1, 2, "C3k2", [1024, True])
    add_backbone(-1, 1, "SPPF", [1024, 5])
    p5 = add_backbone(-1, 2, "C2PSA", [1024])
    p5 = maybe_attention(backbone, current_backbone_index, p5, 1024, recipe_key, "after_c2psa")

    add_head(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    concat_p4_topdown = add_head([-1, p4], 1, "Concat", [1])
    if PLACEMENT in {{"neck_concat_all", "neck_p3_p4_topdown"}}:
        concat_p4_topdown = add_attention(head, current_head_index, concat_p4_topdown, 512, recipe_key)
    top_p4 = add_head(-1, 2, "C3k2", [512, False])

    add_head(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    concat_p3_topdown = add_head([-1, p3], 1, "Concat", [1])
    if PLACEMENT in {{"neck_concat_all", "neck_p3_p4_topdown"}}:
        concat_p3_topdown = add_attention(head, current_head_index, concat_p3_topdown, 256, recipe_key)
    out_p3 = add_head(-1, 2, "C3k2", [256, False])
    out_p3 = maybe_attention(head, current_head_index, out_p3, 256, recipe_key, "head_p3_only")

    add_head(-1, 1, "Conv", [256, 3, 2])
    concat_p4_bottomup = add_head([-1, top_p4], 1, "Concat", [1])
    if PLACEMENT == "neck_concat_all":
        concat_p4_bottomup = add_attention(head, current_head_index, concat_p4_bottomup, 512, recipe_key)
    out_p4 = add_head(-1, 2, "C3k2", [512, False])
    out_p4 = maybe_attention(head, current_head_index, out_p4, 512, recipe_key, "head_p4_only")

    add_head(-1, 1, "Conv", [512, 3, 2])
    concat_p5_bottomup = add_head([-1, p5], 1, "Concat", [1])
    if PLACEMENT == "neck_concat_all":
        concat_p5_bottomup = add_attention(head, current_head_index, concat_p5_bottomup, 1024, recipe_key)
    out_p5 = add_head(-1, 2, "C3k2", [1024, True])

    add_head([out_p3, out_p4, out_p5], 1, "Segment", ["nc", 32, 256])

    return {{
        "nc": 2,
        "scales": {{
            "n": [0.50, 0.25, 1024],
            "s": [0.50, 0.50, 1024],
            "m": [0.50, 1.00, 512],
            "l": [1.00, 1.00, 512],
            "x": [1.00, 1.50, 512],
        }},
        "backbone": backbone,
        "head": head,
    }}


MODEL_YAML_PATHS = {{}}
for recipe_key, yaml_name in YAML_NAMES.items():
    yaml_path = MODEL_CFG_DIR / yaml_name
    yaml_path.write_text(yaml.safe_dump(build_cfg(recipe_key), sort_keys=False))
    MODEL_YAML_PATHS[recipe_key] = str(yaml_path)
    print("Created:", recipe_key, "->", yaml_path)

print(f"\\nPlacement: {{DIRECTION_KEY}} - {{PLACEMENT_DESCRIPTION}}")
print("Smoke-building generated YAML files...")
for recipe_key, yaml_path in MODEL_YAML_PATHS.items():
    model = YOLO(yaml_path)
    print(f"Smoke build OK: {{recipe_key}} ({{len(model.model.model)}} layers)")
'''


def make_experiment_cell(direction, mode):
    key = direction["key"]
    title = direction["title"]
    if mode == "single":
        entries = [
            ("baseline", "Clean Baseline - Grouped Stratified Light YOLO Augmentation", "baseline", None),
            (f"simam_{key}", f"YOLO11n-seg + SimAM @ {title}", "attention", "simam"),
            (f"ca_{key}", f"YOLO11n-seg + CoordAtt @ {title}", "attention", "ca"),
            (f"eca_{key}", f"YOLO11n-seg + ECA @ {title}", "attention", "eca"),
            (f"cbam_{key}", f"YOLO11n-seg + CBAM @ {title}", "attention", "cbam"),
            (f"ema_{key}", f"YOLO11n-seg + EMA @ {title}", "attention", "ema"),
        ]
        heading = f"Single-module experiments - {title}"
    else:
        entries = [
            ("baseline", "Clean Baseline - Grouped Stratified Light YOLO Augmentation", "baseline", None),
            (f"simam_ca_{key}", f"YOLO11n-seg + SimAM+CA @ {title}", "attention", "simam_ca"),
            (f"eca_simam_{key}", f"YOLO11n-seg + ECA+SimAM @ {title}", "attention", "eca_simam"),
            (f"ca_ema_{key}", f"YOLO11n-seg + CA+EMA @ {title}", "attention", "ca_ema"),
            (f"cbam_simam_{key}", f"YOLO11n-seg + CBAM+SimAM @ {title}", "attention", "cbam_simam"),
            (f"ema_simam_ca_{key}", f"YOLO11n-seg + EMA+SimAM+CA @ {title}", "attention", "ema_simam_ca"),
        ]
        heading = f"Combined-module experiments - {title}"

    run_root = f"/kaggle/working/yolo11n_{mode}_{key}_group_run"
    run_base = f"{mode}_{key}_group_run"
    lines = [
        "from pathlib import Path",
        "",
        f'EXPERIMENT_ROOT_STR = "{run_root}"',
        f'RUN_BASE_NAME = "{run_base}"',
        "",
        "EXPERIMENTS = [",
    ]
    for exp_key, name, model_type, recipe_key in entries:
        lines.extend(
            [
                "    {",
                f'        "key": "{exp_key}",',
                f'        "name": "{name}",',
                f'        "model_type": "{model_type}",',
            ]
        )
        if recipe_key is not None:
            lines.append(f'        "yaml": MODEL_YAML_PATHS["{recipe_key}"],')
        lines.append("    },")
    required = [entry[0] for entry in entries]
    lines.extend(
        [
            "]",
            "",
            f"REQUIRED_EXPERIMENT_KEYS = {required!r}",
            'experiment_by_key = {e["key"]: e for e in EXPERIMENTS}',
            "missing_keys = [key for key in REQUIRED_EXPERIMENT_KEYS if key not in experiment_by_key]",
            "if missing_keys:",
            '    raise RuntimeError(f"Missing required experiments: {missing_keys}")',
            "EXPERIMENTS = [experiment_by_key[key] for key in REQUIRED_EXPERIMENT_KEYS]",
            "",
            f'print("{heading}:")',
            'print(f"Total experiments to run: {len(EXPERIMENTS)}")',
            "for e in EXPERIMENTS:",
            '    print("-", e["key"], ":", e["name"])',
            "",
        ]
    )
    return "\n".join(lines)


def build_notebook(template_path, out_path, direction, mode):
    notebook = copy.deepcopy(read_notebook(template_path))
    clear_outputs(notebook)
    title_mode = "single attention modules" if mode == "single" else "combined attention modules"
    set_source(
        notebook.cells[0],
        f"""# YOLO11n {title_mode}: {direction['title']}

{direction['description']}

This notebook keeps the same leakage-safe grouped split, clean light augmentation, training loop, and main metrics as the baseline/single/combined notebooks. Only the attention placement YAML and experiment list are changed.
""",
    )
    set_source(notebook.cells[16], "## 7. Create placement-specific attention YAML files and smoke-build\n")
    set_source(notebook.cells[17], make_yaml_cell(direction))
    set_source(notebook.cells[18], f"## 8. Select {direction['title']} experiment group\n")
    set_source(notebook.cells[19], make_experiment_cell(direction, mode))
    write_notebook(out_path, notebook)


def generate_notebooks():
    created = []
    for direction in DIRECTIONS:
        out_dir = OUT_ROOT / direction["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        single_out = out_dir / f"aip491-01-yolo-seg-11n-single-{direction['key']}.ipynb"
        combined_out = out_dir / f"aip491-01-yolo-seg-11n-combined-{direction['key']}.ipynb"
        build_notebook(SINGLE_TEMPLATE, single_out, direction, "single")
        build_notebook(COMBINED_TEMPLATE, combined_out, direction, "combined")
        created.extend([single_out, combined_out])
    return created


def verify_local_smoke_build():
    sys.path.insert(0, str(ROOT / "yolov11n_attention" / "ultralytics"))
    from ultralytics import YOLO
    import yaml

    for direction in DIRECTIONS:
        namespace = {}
        source = make_yaml_cell(direction).split("MODEL_YAML_PATHS = {}", 1)[0]
        exec(source, namespace)
        build_cfg = namespace["build_cfg"]
        yaml_names = namespace["YAML_NAMES"]
        print(f"\nVerifying {direction['key']}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for recipe_key, yaml_name in yaml_names.items():
                yaml_path = tmpdir / yaml_name
                yaml_path.write_text(yaml.safe_dump(build_cfg(recipe_key), sort_keys=False), encoding="utf-8")
                model = YOLO(str(yaml_path))
                print(f"  OK {recipe_key}: {len(model.model.model)} layers")


def validate_generated_notebooks(paths):
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        print(f"Validated notebook: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-local", action="store_true")
    args = parser.parse_args()

    created = generate_notebooks()
    validate_generated_notebooks(created)
    if args.verify_local:
        verify_local_smoke_build()


if __name__ == "__main__":
    main()
