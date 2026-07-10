"""Create Kaggle-standalone notebooks for CoLPSCRescueLiteGate."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
import yaml
from nbformat.validator import validate

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from create_report_improvement_notebooks import ATTENTION_CELL_TEMPLATE, BASE_MODEL  # noqa: E402


MODULE_FOLDER = "colpsc_rescue_lite"
CLASS_NAME = "CoLPSCRescueLiteGate"
YAML_ARGS = [3, 5, 0.05, 0.25, 0.20, 0.20, 1.0, 0.35, 0.05, 0.10, 0.0]


CLASS_SOURCE = r'''

class CoLPSCRescueLiteGate(nn.Module):
    """Controlled CoTE-LPSC hybrid with rescue/refinement branches.

    CoTE-SR acts as the lesion-preserving branch. LPSC is used only as weak
    suppression, gated by (1 - lesion), so it is discouraged from suppressing
    regions already supported by the lesion branch. ECA/SimAM rescue and
    BoundaryLite refinement remain low-weight terms inside a low-gamma residual
    gate. Input and output shapes are [B, C, H, W].
    """

    def __init__(
        self,
        c1: int,
        eca_k: int = 3,
        triplet_k: int = 5,
        alpha_init: float = 0.05,
        eta_init: float = 0.25,
        beta_init: float = 0.20,
        delta_init: float = 0.20,
        w_l_init: float = 1.0,
        w_r_init: float = 0.35,
        w_b_init: float = 0.05,
        w_s_init: float = 0.10,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        channels = _resolve_channels(c1)
        self.triplet = TripletLiteGate(channels, k=triplet_k)
        self.eca_lesion = ECAGate(channels, k_size=eca_k)
        self.simam = SimAMGate(channels)
        self.nam = NAMGate(channels)
        self.local_prior = LocalContrastPriorLite(channels)
        self.eca_rescue = ECAGate(channels, k_size=eca_k)
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.lambda_t = nn.Parameter(torch.tensor(1.0))
        self.lambda_c = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))
        self.w_l = nn.Parameter(torch.tensor(float(w_l_init)))
        self.w_r = nn.Parameter(torch.tensor(float(w_r_init)))
        self.w_b = nn.Parameter(torch.tensor(float(w_b_init)))
        self.w_s = nn.Parameter(torch.tensor(float(w_s_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        t = self.triplet(x)
        c = self.eca_lesion(x).expand_as(x)
        s = self.simam(x)
        d = torch.abs(t - c)
        lesion = torch.sigmoid(self.lambda_t * t + self.lambda_c * c + self.eta * s - self.alpha * d)

        n = self.nam(x)
        p = self.local_prior(x)
        suppress_score = torch.sigmoid(self.beta * n + self.delta * p)
        weak_suppression = suppress_score * (1.0 - lesion)

        rescue = self.eca_rescue(x).expand_as(x) * s

        boundary_raw = self.dw3(x) + self.dw5(x)
        boundary = torch.sigmoid(boundary_raw - self.smooth(boundary_raw))
        boundary_refine = boundary * lesion

        z = (
            self.w_l * (lesion - 0.5)
            + self.w_r * (rescue - 0.5)
            + self.w_b * (boundary_refine - 0.25)
            - self.w_s * weak_suppression
        )
        gate = 2.0 * torch.sigmoid(z) - 1.0
        return identity + self.gamma * identity * gate
'''


VARIANTS = [
    {
        "template": ROOT / "cote_gate" / "cote_gate.ipynb",
        "notebook": "colpsc_rescue_lite.ipynb",
        "experiment": "colpsc_rescue_lite",
        "title": "CoLPSC-RescueLite P4 - Controlled CoTE-LPSC Rescue Gate",
        "train_name": "YOLO11n-seg + CoLPSC-RescueLite - Controlled CoTE-LPSC Rescue Gate",
        "augmentation_note": "Clean/light YOLO augmentation from the original baseline; no strong augmentation.",
    },
    {
        "template": ROOT / "cote_gate" / "cote_gate_strong_augmentation.ipynb",
        "notebook": "colpsc_rescue_lite_p4_strong.ipynb",
        "experiment": "colpsc_rescue_lite_p4_strong",
        "title": "CoLPSC-RescueLite P4 Strong - Controlled CoTE-LPSC Rescue Gate",
        "train_name": "YOLO11n-seg + CoLPSC-RescueLite - Controlled CoTE-LPSC Rescue Gate + Strong Augmentation",
        "augmentation_note": "P4-only placement with strong augmentation matched to the current SimAM_CA strong notebook.",
    },
]


def build_attention_cell(folder: str, experiment: str) -> str:
    source = ATTENTION_CELL_TEMPLATE
    source = source.replace("\n\nRESEARCH_ATTENTION_MODULES:", CLASS_SOURCE + "\n\nRESEARCH_ATTENTION_MODULES:", 1)
    source = source.replace(
        "    LPSCUncertaintyAttenuatedGate,\n)",
        "    LPSCUncertaintyAttenuatedGate,\n    CoLPSCRescueLiteGate,\n)",
        1,
    )
    source = source.replace("__MODULE_FOLDER_NAME__", folder)
    source = source.replace("__EXPERIMENT_NAME__", experiment)
    return source


def build_yaml_text() -> str:
    model = {
        "nc": BASE_MODEL["nc"],
        "depth_multiple": BASE_MODEL["depth_multiple"],
        "width_multiple": BASE_MODEL["width_multiple"],
        "backbone": BASE_MODEL["backbone"],
        "head": [list(item) for item in BASE_MODEL["head"]],
    }
    model["head"][-2] = [19, 1, CLASS_NAME, YAML_ARGS]
    return yaml.safe_dump(model, sort_keys=False)


def build_yaml_cell() -> str:
    yaml_text = build_yaml_text()
    return f'''# -- Create model YAML inside the notebook and run a build/shape sanity check ----
from pathlib import Path

import torch
from ultralytics import YOLO

YAML_CONTENT = {yaml_text!r}

yaml_dir = Path.cwd() / 'custom_attention_research_modules' / MODULE_FOLDER_NAME / 'generated_yamls'
yaml_dir.mkdir(parents=True, exist_ok=True)
yaml_path = yaml_dir / f'{{EXPERIMENT_NAME}}.yaml'
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


def build_header(variant: dict[str, object]) -> str:
    return f'''# {variant["title"]}

This Kaggle-standalone notebook implements `CoLPSCRescueLiteGate`.

- Module folder: `{MODULE_FOLDER}`
- Experiment name: `{variant["experiment"]}`
- Python class: `{CLASS_NAME}`
- Placement: P4-only before the YOLO11n-seg Segment head
- Attention/YAML setup: defined inline in this notebook; no external custom module or YAML file is required on Kaggle
- Augmentation: {variant["augmentation_note"]}
- Dataset protocol: grouped/no-leakage split, same `nc`, class names, validation/test logic, and healthy-negative evaluation

Design:

```text
lesion = sigmoid(CoTE-SR evidence)
weak_suppression = sigmoid(LPSC weak evidence) * (1 - lesion)
rescue = ECA(x) * SimAM(x)
boundary_refine = BoundaryLite(x) * lesion
gate = 2 * sigmoid(w_l*lesion + w_r*rescue + w_b*boundary_refine - w_s*weak_suppression) - 1
out = x + gamma * x * gate
```

Track: full/labeled-only mask mAP50 and mAP50-95, healthy FP-rate, FP masks/image, disease miss rate, count MAE, healthy-aware score, and segmentation loss gap.
'''


def replace_train_name(source: str, train_name: str) -> str:
    return re.sub(r"'name': 'YOLO11n-seg \+ .*?',", f"'name': '{train_name}',", source, count=1)


def write_variant(variant: dict[str, object]) -> Path:
    nb = nbformat.read(variant["template"], as_version=4)
    nb.nbformat_minor = 5
    nb.cells[0].source = build_header(variant)
    nb.cells[5].source = build_attention_cell(MODULE_FOLDER, str(variant["experiment"]))
    nb.cells[6].source = build_yaml_cell()
    nb.cells[17].source = replace_train_name(nb.cells[17].source, str(variant["train_name"]))

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    validate(nb)
    out_dir = ROOT / MODULE_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(variant["notebook"])
    nbformat.write(nb, out_path)
    return out_path


def main() -> None:
    for variant in VARIANTS:
        path = write_variant(variant)
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
