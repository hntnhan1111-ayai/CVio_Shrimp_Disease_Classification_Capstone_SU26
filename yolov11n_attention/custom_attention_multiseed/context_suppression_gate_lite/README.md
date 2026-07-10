# Context-Suppression Gate Lite - Multiseed Verification

## Direction

Generalization candidate from the priority module group.

## Source Module

- Source folder: `custom_attention/context_suppression_gate_lite`
- Model YAML: `custom_attention/context_suppression_gate_lite/model.yaml`
- Source class: `ContextSuppressionGateLite` from `custom_attention/_shared/attention_modules.py`
- Placement: P3/P4 before `Segment`

## Multiseed Scope

This folder only reruns the existing Context-Suppression Gate Lite module across
the configured seeds. It does not introduce a new gate or alter the dataset.

## Notebook

Open:

```text
context_suppression_gate_lite_multiseed.ipynb
```

The notebook is standalone: it contains the copied Context-Suppression setup
and runs the three seeds directly in notebook cells. It does not call
`run_top4_multiseed.py` or other shared pipeline scripts.

## Expected Trade-Off

The single-seed validation mAP was not the highest, but test performance was
strong. This makes it a useful generalization check against Triplet and CESA.
