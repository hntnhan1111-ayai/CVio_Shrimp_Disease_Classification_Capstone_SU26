# LPSC-Gate

Full name: Lesion-Preserving NAM-SimAM Contrast Gate

Python class: `LPSCGate`

## Idea

LPSC-Gate starts from NAM-style low-FP gating, adds SimAM neuron saliency to
avoid suppressing weak lesions too aggressively, and uses a very light
depthwise local contrast prior for mask/boundary support.

## Placement

P4-only before the Segment head:

```text
Segment input = [P3 layer 16, LPSCGate(P4 layer 19) layer 23, P5 layer 22]
```

P3 is intentionally avoided in this first pass because the contrast prior may
be sensitive to healthy surface texture.

## Expected Benefit

Reduce healthy false positives while preserving more weak lesion and boundary
evidence than NAM alone.

## Risk

The contrast prior can respond to healthy highlights or shell/gill texture and
increase false positives. It may also trail CoTE-Gate on peak localization mAP.

## Sanity Check

```bash
python custom_attention_research_modules/_shared/sanity_check.py --model custom_attention_research_modules/lpsc_gate/lpsc_gate.yaml --imgsz 640
```

## Notebook

```text
custom_attention_research_modules/lpsc_gate/lpsc_gate.ipynb
```

## Output Directory

```text
runs/custom_attention_research_modules/lpsc_gate
custom_attention_research_modules/lpsc_gate/outputs/reports/
```

## Metrics To Watch

healthy FP, precision, disease miss rate, mask mAP50-95, boundary quality,
healthy-aware score, params/FLOPs/latency/export status.
