# SGE-ECA Head Gate - Multiseed Verification

## Direction

Stable/deployment candidate from the priority module group.

## Source Module

- Source folder: `custom_attention/sge_eca_head_gate`
- Model YAML: `custom_attention/sge_eca_head_gate/model.yaml`
- Source class: `SGEECAHeadGate` from `custom_attention/_shared/attention_modules.py`
- Placement: P3/P4 before `Segment`

## Multiseed Scope

This folder only reruns the existing SGE-ECA Head Gate module across the
configured seeds. It does not create a new attention module or modify the
baseline protocol.

## Notebook

Open:

```text
sge_eca_head_gate_multiseed.ipynb
```

The notebook is standalone: it contains the copied SGE-ECA setup and runs the
three seeds directly in notebook cells. It does not call `run_top4_multiseed.py`
or other shared pipeline scripts.

## Expected Trade-Off

SGE-ECA was not the top single-seed mAP result, but it had a strong balance of
low healthy false positives, low miss rate, and low loss gap. This makes it the
main deployment-oriented candidate.
