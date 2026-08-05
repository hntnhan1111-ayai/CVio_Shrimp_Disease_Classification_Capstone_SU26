# Model card: CVio shrimp image classifiers

## Model family and release state

This repository studies YOLO26m image classifiers trained with either CE or a
project composition of ASL-LDAM and SimAM-DCFR. Four conceptual checkpoint roles
exist, but only the two EXT-3-Original binaries are released and verified.

| Role | Outputs | Release state |
|---|---:|---|
| SDI-4 CE | 4 | binary unresolved |
| SDI-4 proposed | 4 | binary unresolved |
| EXT-3-Original CE | 3 | exact hash verified and published through Git LFS |
| EXT-3-Original proposed | 3 | exact hash and architecture verified; published through Git LFS |

See [CHECKPOINTS.md](CHECKPOINTS.md) for hashes and evidence.

## Intended use

- Research on class-imbalanced shrimp-disease image classification.
- Reproduction of fixed seed-42 experimental procedures.
- Architecture and loss ablation within the documented dataset regime.
- Qualitative XAI review with explicit non-causal interpretation.

## Out-of-scope use

- Veterinary or clinical diagnosis.
- Autonomous treatment, culling, farm, or biosecurity decisions.
- Four-class inference with the three-class EXT-3 checkpoints.
- Performance claims outside the documented split and acquisition domain.
- Statistical claims across seeds or specimens.

## Training data and labels

SDI-4 contains Healthy, BG, WSSV, and WSSV_BG. Its fixed image-level split has
804 training, 172 validation, and 173 test images. EXT-3-Original contains only
Healthy, BG, and WSSV and has a 47-image retained test package. The combined
regime has a 220-image test package.

The data were already background-removed. The split is not proven
specimen-independent, and images are not redistributed here.

## Reported and verified performance

- SDI-4 reported official selected: Accuracy 0.913295, Macro-F1 0.910137;
  official-final binary unresolved.
- SDI-4 reported CE: Accuracy/Macro-F1 0.890200; binary unresolved.
- EXT-3 proposed package: Accuracy 0.914894, Macro-F1 0.912937, Kappa 0.869444.
- EXT-3 CE package: Accuracy 0.702128, Macro-F1 0.722990, Kappa 0.550239.
- Combined package: CE Macro-F1 0.865105 exceeds proposed 0.821801.

EXT-3 metrics were cross-checked against exact checkpoint hashes and retained
package outputs; the unavailable 47 source images prevented a second inference
run.

## Architecture and loss

The proposed architecture inserts a late SimAM-DCFR block before the original
`Classify` head. ASL-LDAM subtracts a class-count-derived margin from the
true-class logit, scales the logits, then applies the repository's single-label
softmax ASL. This implementation order is not proven optimal against all
alternatives.

## Risks and mitigations

| Risk | Mitigation in this release |
|---|---|
| Wrong dataset regime | Separate paths, class-order fields, and manifests |
| Filename/provenance mismatch | SHA-256 plus checkpoint metadata and architecture audit |
| Overstated official result | SDI-4 reported metrics explicitly separated from unresolved binaries |
| Domain failure | Non-clinical intended-use statement and field-validation gap |
| Misleading explanation | XAI panels labeled qualitative-only |
| Deployment confusion | Current TFLite files labeled historical/unresolved, not official proposed |

## Ethical and operational notes

Predictions can be wrong and may be affected by background removal, camera,
lighting, geography, husbandry context, and label quality. Domain experts must
review outputs, and laboratory confirmation remains necessary for disease
decisions.

## Reproducibility

Seed 42, image size 224, 30-epoch budget, batch 32, and AdamW are stored in the
audited checkpoints. See [REPRODUCE.md](REPRODUCE.md),
[`weights/manifest.json`](../weights/manifest.json), and
[`FINAL_AUDIT.md`](../artifacts/release_audit/FINAL_AUDIT.md).
