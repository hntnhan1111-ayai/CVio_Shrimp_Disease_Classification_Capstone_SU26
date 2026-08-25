# Council Review 3 - Classification-facing requirements

Meeting: Council No.03, 09 Aug 2026. Approved with revisions.

## Reviewer 1
1. Add ablation evidence demonstrating the effectiveness of ASL-LDAM + SimAM-DCFR.
2. Add real-world images captured under different environments and phone camera resolutions, not only dataset images.
3. Explain the “lightweight” claim using final architecture, model size, parameter count, key results, and what phone configurations can run it.

## Reviewer 2
1. Explain why proposed Macro-F1 changes from 91.29% on EXT-3 to 82.18% on the integrated dataset while CE reaches 86.51% integrated. Do not use the unsupported explanation “data are easier.”
2. Fourier/spectrum comment belongs to segmentation and should not be mixed into the classification causal story.

## Reviewer 3
1. Improve evaluation/parameter comparison presentation on the relevant slide/table.
2. Be careful with YOLO11/Yolo26 rationale: the classification Table 6.3 and segmentation baseline tables are different objects.

## Evidence-safe integrated interpretation
The proposed model is worse than CE on both source subsets after joint four-class/multi-source optimization. Therefore the integrated degradation is not confined to one source. Plausible mechanisms include source/domain interaction, changed class priors/margins, and WSSV vs WSSV_BG boundary conflict; these are hypotheses, not causal proof.
