# Context-Suppression Gate Lite

## Module

- Folder: `context_suppression_gate_lite`
- Python class: `ContextSuppressionGateLite`
- Priority: 6
- Model YAML: `model.yaml`
- Notebook: `context_suppression_gate_lite.ipynb`

## Idea

Global pooled context gate as a safe single-input approximation of P5-guided suppression.

## Inherited Attention Concepts

This module is derived from the proposal in `attention_custom_module_proposals.md` and uses lightweight residual attention ideas suitable for YOLO11n-seg. The implementation preserves input/output shape `[B, C, H, W]`, does not change channels, and uses learnable `gamma` initialized to zero.

## YOLO11n-Seg Placement

P3/P4 before Segment, single-input context variant

Layer mapping used by the YAML:

- P3 neck output: layer `16`
- P4 neck output: layer `19`
- P5 neck output: layer `22`
- Segment input after attention: `[23, 24, 22]`

## Expected Benefit

Improve precision and reduce false positives with low compute.

## Risks

Context can suppress true disease if disease/healthy images are visually similar.

## Sanity Check

Run from the repository root:

```bash
python custom_attention/_shared/sanity_check.py --model custom_attention/context_suppression_gate_lite/model.yaml --imgsz 640 --device cpu
```

Or run the full priority suite:

```bash
python custom_attention/run_all_sanity_checks.py
```

## Training Notebook

Open `context_suppression_gate_lite.ipynb` and run cells only after the sanity check cell passes. The notebook is generated from the leakage-safe baseline and only changes the model YAML, experiment key/name, and output root.

For Kaggle, this notebook is standalone for custom attention code: it embeds the attention classes, registration function, and YAML text. Do not add `from custom_attention._shared...` imports unless you also upload the full repository package and put it on `sys.path`.

## Metrics to Track

healthy FP, recall, mask mAP50, latency

## Notes

This module is a theoretical/custom proposal and must be validated experimentally. Do not claim it improves mAP, mask AP, or healthy false positives until it is trained and evaluated under the same leakage-safe protocol.
