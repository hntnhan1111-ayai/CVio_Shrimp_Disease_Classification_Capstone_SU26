# Custom Attention Experiments for YOLO11n-Seg
This directory contains lightweight custom attention experiments for shrimp disease instance segmentation (BG/WSSV) with YOLO11n-seg. The implementation follows `attention_custom_module_proposals.md` and keeps the clean baseline protocol unchanged.
## Source and Policy
- Source proposal: `attention_custom_module_proposals.md`.
- Baseline notebook: `yolov11n_attention/yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb`.
- The baseline notebook is not modified.
- The grouped shrimp-level no-leakage split is preserved in every generated notebook.
- Do not train a module before its `model.yaml` passes sanity check.
## Implemented Priority Modules
| Priority | Folder | Class | Placement | Main risk |
|---:|---|---|---|---|
| 1 | `sge_eca_head_gate` | `SGEECAHeadGate` | P3/P4 before Segment | Group size must divide channels safely; effect may be close to ECA-only. |
| 2 | `ces_lite` | `CESLite` | P3/P4 before Segment | Coordinate and neuron gates may overlap spatially; dynamic shape/export must be checked. |
| 3 | `low_fp_residual_spatial_gate` | `LowFPResidualSpatialGate` | P3/P4 before Segment | A strong spatial gate can suppress small lesions and lower recall. |
| 4 | `p3p4_semantic_attention_gate` | `P3P4SemanticAttentionGate` | P3/P4 before Segment, single-input safe variant | True semantic-guided multi-input gate is not used yet; self-context may be weaker. |
| 5 | `boundary_aware_lite_attention` | `BoundaryAwareLiteAttention` | P3/P4 before Segment | May amplify healthy texture/noise and increase false positives. |
| 6 | `context_suppression_gate_lite` | `ContextSuppressionGateLite` | P3/P4 before Segment, single-input context variant | Context can suppress true disease if disease/healthy images are visually similar. |
| 7 | `ca_spatial_low_fp_gate` | `CASpatialLowFPGate` | P3/P4 before Segment | Two spatial/position gates can be redundant and reduce small-lesion recall. |
| 8 | `prototype_aware_mask_gate_lite` | `PrototypeAwareMaskGateLite` | P3 only before Segment | P3 texture is noisy; false positives may increase. |

## Compatibility / Legacy Notes
Older experiment folders such as `triplet_attention_segment_head`, `nam_attention`, `ca_lite_spatial_gate`, and result notebooks under `res/` are preserved. The new priority suite lives in `_shared/` plus the folders listed above.

## Kaggle Standalone Notebook Policy
The eight priority notebooks are generated to be Kaggle-standalone for custom attention code. Each notebook embeds:

- the attention module class definitions;
- `register_custom_attention()`;
- the experiment `model.yaml` text, written at runtime to `/kaggle/working/custom_attention_models/<module>/model.yaml`;
- a local sanity-check function.

Therefore, these notebooks do not need:

```python
from custom_attention._shared.register_attention import register_custom_attention
```

The `_shared/` package and `run_all_sanity_checks.py` are still kept for local repository development and batch sanity checks. On Kaggle, open the module notebook directly and run the embedded register/sanity cells before training.
## Run One Sanity Check
```bash
python custom_attention/_shared/sanity_check.py --model custom_attention/sge_eca_head_gate/model.yaml --imgsz 640 --device cpu
```
## Run All Sanity Checks
```bash
python custom_attention/run_all_sanity_checks.py
```
The script continues after failures and prints a PASS/FAIL table. A nonzero exit means at least one module failed.
## Open a Training Notebook
Start with the first priority module after sanity checks pass:

```text
custom_attention/sge_eca_head_gate/sge_eca_head_gate.ipynb
```
Every generated notebook keeps the Roboflow download, grouped split, class names, training configuration, validation/test evaluation, healthy false-positive metrics, and export/report logic from the clean baseline.
## Important Warnings
- Do not train modules that fail sanity check.
- Do not change the dataset split or anti-leakage protocol for attention ablations.
- Do not compare attention runs against baseline unless seed, split, augmentation, image size, and evaluation logic are aligned.
- All custom modules are proposals that require experimental verification.
