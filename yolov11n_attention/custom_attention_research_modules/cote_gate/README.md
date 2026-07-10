# CoTE-Gate

Full name: Consensus-Regularized Triplet-ECA Gate

Python class: `CoTEGate`

## Idea

CoTE-Gate combines a safe Triplet-Lite spatial/axis gate with ECA channel
evidence, then applies soft suppression only where those two attention sources
disagree. It is designed to keep Triplet-like localization while reducing
healthy false positives and overfit.

The Triplet branch is a safe Triplet-Lite implementation using standard
depthwise/pointwise convolution plus descriptor branches with contiguous
permuted tensors.

## Placement

P4-only before the Segment head:

```text
Segment input = [P3 layer 16, CoTEGate(P4 layer 19) layer 23, P5 layer 22]
```

## Expected Benefit

Keep competitive mask mAP while reducing healthy FP, seg_loss_gap, and
activation from disease-like healthy texture.

## Risk

If disagreement suppression becomes too strong, weak or small lesions can be
missed. This is why `alpha` starts low and `gamma` starts at `0.0`.

## Sanity Check

```bash
python custom_attention_research_modules/_shared/sanity_check.py --model custom_attention_research_modules/cote_gate/cote_gate.yaml --imgsz 640
```

## Notebook

```text
custom_attention_research_modules/cote_gate/cote_gate.ipynb
```

## Output Directory

```text
runs/custom_attention_research_modules/cote_gate
custom_attention_research_modules/cote_gate/outputs/reports/
```

## Metrics To Watch

test mask mAP50, full test mAP50, healthy FP, disease miss rate,
healthy-aware score, seg_loss_gap, params/FLOPs/latency/export status.
