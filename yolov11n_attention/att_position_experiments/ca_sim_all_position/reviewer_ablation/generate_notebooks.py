"""Generate reviewer-requested CA/SimAM ablation notebooks from the strong template."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "yolov11n_ca_simam_02_head_p3_strong.ipynb"

CONFIGS = [
    ("01_component_order/01_baseline", "baseline", [], []),
    ("01_component_order/02_ca_only_p3_p4_p5", "ca_only", ["p3", "p4", "p5"], ["CoordAtt"]),
    ("01_component_order/03_simam_only_p3_p4_p5", "simam_only", ["p3", "p4", "p5"], ["SimAM"]),
    ("01_component_order/04_ca_to_simam_p3_p4_p5", "ca_to_simam", ["p3", "p4", "p5"], ["CoordAtt", "SimAM"]),
    ("01_component_order/05_simam_to_ca_p3_p4_p5", "simam_to_ca", ["p3", "p4", "p5"], ["SimAM", "CoordAtt"]),
    ("02_single_level_placement/01_ca_to_simam_p3_only", "ca_to_simam_p3", ["p3"], ["CoordAtt", "SimAM"]),
    ("02_single_level_placement/02_ca_to_simam_p4_only", "ca_to_simam_p4", ["p4"], ["CoordAtt", "SimAM"]),
    ("02_single_level_placement/03_ca_to_simam_p5_only", "ca_to_simam_p5", ["p5"], ["CoordAtt", "SimAM"]),
    ("03_pairwise_placement_optional/01_ca_to_simam_p3_p4", "ca_to_simam_p3_p4", ["p3", "p4"], ["CoordAtt", "SimAM"]),
    ("03_pairwise_placement_optional/02_ca_to_simam_p3_p5", "ca_to_simam_p3_p5", ["p3", "p5"], ["CoordAtt", "SimAM"]),
    ("03_pairwise_placement_optional/03_ca_to_simam_p4_p5", "ca_to_simam_p4_p5", ["p4", "p5"], ["CoordAtt", "SimAM"]),
]


def architecture_cell(key, placements, modules):
    recipe = [(m, "lambda c: [c, 32]" if m == "CoordAtt" else "lambda c: []") for m in modules]
    recipe_text = ", ".join(f'("{m}", {fn})' for m, fn in recipe)
    return f'''from pathlib import Path
import yaml

from ultralytics import YOLO
import ultralytics

ULTRALYTICS_PACKAGE_DIR = Path(ultralytics.__file__).resolve().parent
MODEL_CFG_DIR = ULTRALYTICS_PACKAGE_DIR / "cfg/models/11"
MODEL_CFG_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_KEY = "{key}"
PLACEMENTS = {placements!r}
ATTENTION_RECIPE = [{recipe_text}]
YAML_NAME = f"yolo11n-seg-{{EXPERIMENT_KEY}}.yaml"

def add_attention(section, global_index_fn, from_idx, nominal_channels):
    current = from_idx
    for module_name, args_fn in ATTENTION_RECIPE:
        section.append([current, 1, module_name, args_fn(nominal_channels)])
        current = global_index_fn()
    return current

def build_cfg():
    backbone, head = [], []

    def add_backbone(f, n, m, args):
        backbone.append([f, n, m, args])
        return len(backbone) - 1

    def add_head(f, n, m, args):
        head.append([f, n, m, args])
        return len(backbone) + len(head) - 1

    def current_head_index():
        return len(backbone) + len(head) - 1

    add_backbone(-1, 1, "Conv", [64, 3, 2])
    add_backbone(-1, 1, "Conv", [128, 3, 2])
    add_backbone(-1, 2, "C3k2", [256, False, 0.25])
    add_backbone(-1, 1, "Conv", [256, 3, 2])
    p3_backbone = add_backbone(-1, 2, "C3k2", [256, False, 0.25])
    add_backbone(-1, 1, "Conv", [512, 3, 2])
    p4_backbone = add_backbone(-1, 2, "C3k2", [512, False, 0.25])
    add_backbone(-1, 1, "Conv", [1024, 3, 2])
    add_backbone(-1, 2, "C3k2", [1024, True])
    add_backbone(-1, 1, "SPPF", [1024, 5])
    p5_backbone = add_backbone(-1, 2, "C2PSA", [1024])

    add_head(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    add_head([-1, p4_backbone], 1, "Concat", [1])
    top_p4 = add_head(-1, 2, "C3k2", [512, False])
    add_head(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    add_head([-1, p3_backbone], 1, "Concat", [1])
    out_p3 = add_head(-1, 2, "C3k2", [256, False])
    if "p3" in PLACEMENTS:
        out_p3 = add_attention(head, current_head_index, out_p3, 256)

    add_head(-1, 1, "Conv", [256, 3, 2])
    add_head([-1, top_p4], 1, "Concat", [1])
    out_p4 = add_head(-1, 2, "C3k2", [512, False])
    if "p4" in PLACEMENTS:
        out_p4 = add_attention(head, current_head_index, out_p4, 512)

    add_head(-1, 1, "Conv", [512, 3, 2])
    add_head([-1, p5_backbone], 1, "Concat", [1])
    out_p5 = add_head(-1, 2, "C3k2", [1024, True])
    if "p5" in PLACEMENTS:
        out_p5 = add_attention(head, current_head_index, out_p5, 1024)

    add_head([out_p3, out_p4, out_p5], 1, "Segment", ["nc", 32, 256])
    return {{
        "nc": 2,
        "scales": {{
            "n": [0.50, 0.25, 1024], "s": [0.50, 0.50, 1024],
            "m": [0.50, 1.00, 512], "l": [1.00, 1.00, 512],
            "x": [1.00, 1.50, 512],
        }},
        "backbone": backbone,
        "head": head,
    }}

MODEL_YAML_PATH = MODEL_CFG_DIR / YAML_NAME
MODEL_YAML_PATH.write_text(yaml.safe_dump(build_cfg(), sort_keys=False))
print("Created:", MODEL_YAML_PATH)
print("Placements:", PLACEMENTS or "none (baseline)")
print("Recipe:", [m for m, _ in ATTENTION_RECIPE] or "none (baseline)")

if ATTENTION_RECIPE:
    model = YOLO(str(MODEL_YAML_PATH))
    import torch
    model.model.eval()
    with torch.no_grad():
        smoke_output = model.model(torch.zeros(1, 3, 256, 256))
    print(f"Build/forward OK: {{len(model.model.model)}} layers; output={{type(smoke_output).__name__}}")
else:
    print("Baseline uses the official yolo11n-seg.pt architecture.")
'''


def experiment_cell(key, placements, modules):
    label = "Baseline (no custom attention)" if not modules else f"{' -> '.join(modules)} @ {'+'.join(p.upper() for p in placements)} head outputs"
    model_type = "baseline" if not modules else "attention"
    yaml_value = 'None' if not modules else 'str(MODEL_YAML_PATH)'
    return f'''import sys
sys.path.insert(0, "/content/ultralytics")

EXPERIMENT_ROOT_STR = "/content/reviewer_ablation/{key}_run"
RUN_BASE_NAME = "reviewer_ablation"

EXPERIMENTS = [
    {{
        "key": "{key}",
        "name": "YOLO11n-seg {label}",
        "model_type": "{model_type}",
        "yaml": {yaml_value},
    }},
]

print("Reviewer ablation configuration")
for e in EXPERIMENTS:
    print("-", e["key"], ":", e["name"])
'''


def main():
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for relative_dir, key, placements, modules in CONFIGS:
        nb = json.loads(json.dumps(template))
        nb.setdefault("metadata", {})["experiment"] = {
            "study": "reviewer_ca_simam_ablation",
            "key": key,
            "placements": placements,
            "attention_order": modules,
        }
        nb["metadata"].pop("placement", None)
        nb["metadata"].pop("ca_sim_position_experiment", None)
        title = f"# Reviewer Ablation: {key}\n\nControlled strong-policy experiment for CA/SimAM component, order, and placement analysis.\n"
        nb["cells"][0]["source"] = title.splitlines(keepends=True)
        nb["cells"][18]["source"] = architecture_cell(key, placements, modules).splitlines(keepends=True)
        nb["cells"][19]["source"] = [f"## 8. Select reviewer ablation experiment: `{key}`\n"]
        nb["cells"][20]["source"] = experiment_cell(key, placements, modules).splitlines(keepends=True)
        for cell in nb["cells"]:
            cell["outputs"] = [] if cell.get("cell_type") == "code" else cell.get("outputs", [])
            cell.pop("execution_count", None)
        target_dir = ROOT / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"yolov11n_reviewer_ablation_{key}.ipynb"
        target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
