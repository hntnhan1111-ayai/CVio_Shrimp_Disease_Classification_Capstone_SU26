# SCSG-Gate

Full name: Scale-Aware Coordinate-SGE Gate

Python class: `SCSGGate`

## Idea

SCSG-Gate combines Coordinate Attention for position-aware localization with
SGE for group semantic denoising. It is intended as the lower-risk stable
candidate rather than the most aggressive high-mAP candidate.

## Placement

P4-only before the Segment head:

```text
Segment input = [P3 layer 16, SCSGGate(P4 layer 19) layer 23, P5 layer 22]
```

## Expected Benefit

Improve stability, healthy-aware score, and localization over SGE-ECA style
gating while staying mobile-friendly and lightweight.

## Risk

It may be too conservative to beat stronger Triplet-like modules on peak mAP.
Coordinate positional cues can also still amplify elongated healthy texture.

## Sanity Check

```bash
python custom_attention_research_modules/_shared/sanity_check.py --model custom_attention_research_modules/scsg_gate/scsg_gate.yaml --imgsz 640
```

## Notebook

```text
custom_attention_research_modules/scsg_gate/scsg_gate.ipynb
```

## Output Directory

```text
runs/custom_attention_research_modules/scsg_gate
custom_attention_research_modules/scsg_gate/outputs/reports/
```

## Metrics To Watch

mask mAP50, mask mAP50-95, healthy FP, miss rate, seg_loss_gap, stability
across seeds in later shortlist runs.
