# Custom Attention Comparison vs Clean Baseline

Nguon doi chieu baseline: `yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb`.

Baseline clean/light YOLO11n-seg:

| Metric | Value |
|---|---:|
| Val mask mAP50 / mAP50-95 | 0.502 / 0.173 |
| Full test mask mAP50 / mAP50-95 | 0.441 / 0.161 |
| Diseased-only test mask mAP50 / mAP50-95 | 0.466 / 0.169 |
| Healthy FP rate | 36.6% |
| Healthy FP masks/image | 0.415 |
| Disease box miss rate | 12.5% |
| Healthy-aware score | 0.393 |

Notes:

- `Full` = full test set, including healthy images.
- `Disease` = labeled/diseased-only test subset.
- `Delta` values are versus the clean baseline above.
- Clean/light runs are directly comparable to the baseline. Strong-augmentation runs are useful signals, but augmentation is a confounder.
- Multiseed aggregate CSV was empty, so multiseed metrics below were read from the notebook output tables. Full-test multiseed mAP50 was not complete enough to claim a mean.

## Single-Run Custom Attention Modules

These are the completed notebooks under `custom_attention/res`.

| Module | Define attention | Full mAP50 | Disease mAP50 | Full/Disease mAP50-95 | Healthy FP | Miss | HA score | Delta vs baseline | Pros when applied | Cons / risk |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Baseline clean | No custom attention; original YOLO11n-seg Segment head. | 0.441 | 0.466 | 0.161 / 0.169 | 36.6% | 12.5% | 0.393 | reference | Stable clean protocol; best clean single-run target. | Still has moderate healthy false positives. |
| Triplet Attention Segment Head | Cross-dimension Triplet attention before Segment, modeling C-H, C-W, and H-W interactions. | 0.422 | 0.456 | 0.135 / 0.146 | 34.1% | 14.8% | 0.381 | Full -0.019, Disease -0.010, FP -2.4 pp, HA -0.012 | Closest single-run to baseline; reduces healthy FP slightly; good candidate for threshold/seed follow-up. | mAP50-95 drops; miss rate rises; permute-style attention can add export/latency risk. |
| CESA-Lite Segment Head | Legacy segment-head lightweight channel/spatial enhancement, using ECA-like channel and saliency/spatial cues. | 0.413 | 0.449 | 0.131 / 0.142 | 34.1% | 18.2% | 0.360 | Full -0.028, Disease -0.017, FP -2.4 pp, HA -0.033 | Good accuracy/FP balance among single runs; segment-head placement is relevant to mask features. | Higher disease miss; lower mask quality at mAP50-95. |
| Context-Suppression Gate Lite | Global pooled context gate on P3/P4, approximating P5-guided suppression with one input. | 0.392 | 0.439 | 0.118 / 0.133 | 36.6% | 18.2% | 0.349 | Full -0.049, Disease -0.027, FP +0.0 pp, HA -0.044 | Keeps FP rate equal to baseline; context idea is useful for healthy-negative control. | Suppresses true disease too often; no net gain in HA score. |
| SGE-ECA Head Gate | Spatial Group Enhance plus ECA channel selection, used as a very light P3/P4 pre-head gate. | 0.375 | 0.408 | 0.116 / 0.127 | 31.7% | 11.4% | 0.335 | Full -0.066, Disease -0.058, FP -4.9 pp, HA -0.058 | Lightweight; reduces healthy FP and miss slightly; deployable structure. | Accuracy loss is still large; SGE group choice may be sensitive. |
| CA-Lite Spatial Gate | Coordinate Attention lite combined with a residual spatial gate on P3/P4. | 0.375 | 0.399 | 0.117 / 0.126 | 39.0% | 19.3% | 0.305 | Full -0.066, Disease -0.067, FP +2.4 pp, HA -0.088 | Coordinate/spatial mechanism is interpretable for shrimp body geometry. | Worse FP and miss; likely over-gates small lesions. |
| CES-Lite | Coordinate Attention + ECA + SimAM-style neuron saliency in a lightweight residual gate. | 0.373 | 0.404 | 0.111 / 0.119 | 46.3% | 6.8% | 0.326 | Full -0.068, Disease -0.063, FP +9.8 pp, HA -0.067 | Very low miss rate; can preserve lesion recall. | Healthy FP increases strongly; HA score drops. |
| Boundary-Aware Lite Attention | Local contrast/boundary residual gate intended to emphasize lesion edges and mask detail. | 0.359 | 0.385 | 0.091 / 0.099 | 48.8% | 8.0% | 0.300 | Full -0.082, Disease -0.081, FP +12.2 pp, HA -0.093 | Lower miss suggests it sees lesion-like regions. | Boundary/texture cue amplifies healthy noise; worst FP among several candidates. |
| Low-FP CBAM Lite | Lightweight CBAM-style channel gate plus spatial gate, tuned for lower false positives. | 0.356 | 0.394 | 0.125 / 0.138 | 36.6% | 9.1% | 0.325 | Full -0.085, Disease -0.073, FP +0.0 pp, HA -0.068 | Miss improves; CBAM-lite is easy to explain. | Does not reduce FP; full/disease mAP lower. |
| Prototype-Aware Mask Gate Lite | P3 mask/prototype-aware gate using a mask hint plus channel calibration before Segment. | 0.349 | 0.390 | 0.094 / 0.105 | 29.3% | 14.8% | 0.316 | Full -0.092, Disease -0.076, FP -7.3 pp, HA -0.077 | Strong FP reduction; segmentation-specific placement is conceptually good. | P3/mask texture seems noisy; mAP50-95 drops hard. |
| Attention Gate Neck P3/P4 | U-Net-style attention gate over P3/P4 neck features before Segment. | 0.349 | 0.379 | 0.114 / 0.126 | 41.5% | 14.8% | 0.296 | Full -0.092, Disease -0.087, FP +4.9 pp, HA -0.097 | Segmentation-gate idea fits neck features. | Current implementation hurts both accuracy and FP. |
| P3P4 Semantic Attention Gate | Semantic guide/gate over P3/P4 neck features to filter noisy feature maps before Segment. | 0.341 | 0.382 | 0.087 / 0.097 | 53.7% | 17.0% | 0.276 | Full -0.100, Disease -0.084, FP +17.1 pp, HA -0.117 | Intended to filter noisy P3/P4 semantics. | Empirically not recommended; FP explodes and mask quality drops. |
| NAM Attention | Normalization-based attention using BN/normalization importance weights for feature reweighting. | 0.335 | 0.385 | 0.117 / 0.138 | 29.3% | 14.8% | 0.307 | Full -0.106, Disease -0.081, FP -7.3 pp, HA -0.086 | Good FP suppression; simple and lightweight. | Accuracy loss too large; BN/NAM signal may be unstable on small data. |
| Low-FP Residual Spatial Gate | Residual spatial suppression gate using shallow spatial descriptors to reduce healthy texture activation. | 0.327 | 0.361 | 0.107 / 0.118 | 41.5% | 9.1% | 0.279 | Full -0.114, Disease -0.105, FP +4.9 pp, HA -0.114 | Low miss rate. | Fails its main low-FP objective in this run; not a priority. |
| CA-Spatial Low-FP Gate | Coordinate Attention combined with a spatial low-FP gate for position-aware suppression. | 0.321 | 0.355 | 0.089 / 0.098 | 31.7% | 12.5% | 0.278 | Full -0.120, Disease -0.112, FP -4.9 pp, HA -0.116 | Reduces FP without increasing miss. | Accuracy and boundary quality collapse; too much spatial suppression. |

Single-run takeaway:

- Best clean single-run candidate: `Triplet Attention Segment Head`, because it is closest to baseline and slightly lowers FP.
- Best lightweight low-FP signal: `SGE-ECA Head Gate`, but only if the accuracy drop can be recovered.
- Avoid as primary candidates in current form: `P3P4 Semantic Attention Gate`, `CA-Spatial Low-FP Gate`, `Low-FP Residual Spatial Gate`, and `SCSG-like over-suppressive gates`.

## Research Modules and P4 Strong Variants

These are the completed notebooks under `custom_attention_research_modules/results` plus extracted improvement results. Strong rows use the strong augmentation policy, so compare cautiously against the clean baseline.

| Module | Define attention | Aug | Full mAP50 | Disease mAP50 | Full/Disease mAP50-95 | Healthy FP | Miss | HA score | Delta vs baseline | Pros when applied | Cons / risk |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| CoTE | Consensus-Regularized Triplet-ECA Gate: Triplet-Lite spatial/axis gate + ECA channel evidence with soft disagreement suppression on P4. | clean | 0.372 | 0.416 | 0.099 / 0.112 | 43.9% | 10.2% | 0.329 | Full -0.069, Disease -0.050, FP +7.3 pp | Lower miss than baseline; consensus idea can rescue lesion evidence. | FP and mAP are worse under clean/light. |
| LPSC | Lesion-Preserving NAM-SimAM Contrast Gate: NAM low-FP gate + SimAM saliency + local contrast prior on P4. | clean | 0.355 | 0.401 | 0.102 / 0.114 | 31.7% | 11.4% | 0.320 | Full -0.086, Disease -0.065, FP -4.9 pp | Low-FP direction works; miss slightly better than baseline. | Disease mAP drop is too large. |
| SCSG | Scale-Aware Coordinate-SGE Gate: Coordinate Attention plus SGE group semantic denoising on P4. | clean | 0.269 | 0.301 | 0.076 / 0.084 | 41.5% | 12.5% | 0.219 | Full -0.171, Disease -0.165, FP +4.9 pp | None from current run except it trains. | Not recommended; severe accuracy loss. |
| CoTE strong | Same CoTE Triplet-ECA disagreement gate, trained with strong augmentation. | strong | 0.433 | 0.487 | 0.123 / 0.136 | 31.7% | 9.1% | 0.421 | Full -0.008, Disease +0.020, FP -4.9 pp, HA +0.027 | Best balanced research result; improves disease mAP and HA while reducing FP. | Strong augmentation confounds architecture conclusion; mAP50-95 still lower than clean baseline. |
| LPSC strong | Same LPSC NAM-SimAM contrast gate, trained with strong augmentation. | strong | 0.427 | 0.463 | 0.133 / 0.143 | 26.8% | 21.6% | 0.384 | Full -0.014, Disease -0.003, FP -9.8 pp | Best FP suppression; useful if false positives are the deployment priority. | Miss rate rises sharply; may be over-conservative. |
| CoLPSC | Hybrid CoTE + LPSC gate combining lesion evidence, weak suppression, channel rescue, and local refinement. | clean | 0.322 | 0.374 | 0.079 / 0.090 | 46.3% | 13.6% | 0.283 | Full -0.119, Disease -0.092, FP +9.8 pp | Hybrid concept is interesting. | Current clean result is worse on nearly every metric. |
| CoLPSC strong | Same CoLPSC hybrid gate, trained with strong augmentation. | strong | 0.367 | 0.409 | 0.107 / 0.119 | 29.3% | 19.3% | 0.327 | Full -0.074, Disease -0.057, FP -7.3 pp | Reduces FP under strong aug. | Large mAP and miss penalty. |
| CoTE-SR P4 strong | CoTE with SimAM rescue path to preserve weak lesion responses. | strong | 0.402 | 0.451 | 0.118 / 0.131 | 39.0% | 10.2% | 0.373 | Full -0.039, Disease -0.015, FP +2.4 pp | Best of the six newer P4 variants for disease mAP; miss is low. | Still below baseline and raises FP. |
| LPSC-UA P4 strong | LPSC uncertainty-attenuated gate that weakens NAM suppression when NAM and SimAM disagree. | strong | 0.374 | 0.417 | 0.123 / 0.138 | 29.3% | 21.6% | 0.331 | Full -0.067, Disease -0.049, FP -7.3 pp | Low FP; uncertainty attenuation is plausible. | Too much disease miss. |
| CoTE-SD P4 strong | CoTE soft-disagreement gate using softened/tanh disagreement suppression. | strong | 0.348 | 0.403 | 0.101 / 0.119 | 36.6% | 14.8% | 0.325 | Full -0.093, Disease -0.063, FP +0.0 pp | Does not worsen FP rate. | Accuracy lower; no clear advantage. |
| LPSC-SG P4 strong | LPSC soft gate with weaker NAM suppression and stronger SimAM-style rescue. | strong | 0.361 | 0.399 | 0.111 / 0.123 | 39.0% | 10.2% | 0.322 | Full -0.080, Disease -0.067, FP +2.4 pp | Miss rate improves. | FP and mAP worse. |
| LPSC-ER P4 strong | LPSC ECA-rescue variant adding ECA channel rescue to soft LPSC suppression. | strong | 0.354 | 0.397 | 0.112 / 0.125 | 26.8% | 20.5% | 0.310 | Full -0.087, Disease -0.069, FP -9.8 pp | Good FP suppression. | Disease miss too high; mAP too low. |
| CoTE-BL P4 strong | CoTE with a lightweight boundary/local contrast prior. | strong | 0.338 | 0.382 | 0.098 / 0.110 | 22.0% | 38.6% | 0.276 | Full -0.103, Disease -0.084, FP -14.6 pp | Lowest FP among these variants. | Miss rate is unacceptable; low FP comes from missing disease. |

Research takeaway:

- `CoTE strong` is the best research result overall, but it should be rerun under clean/light or baseline should be rerun with the same strong augmentation before making a strict architecture claim.
- `LPSC strong` and some LPSC variants prove the low-FP direction, but the penalty is higher miss rate.
- The six newer P4 variants did not beat `CoTE strong`; `CoTE-SR P4 strong` is the least bad among them for accuracy, while `CoTE-BL` is too conservative.
- `colpsc_rescue_lite` notebooks currently have no outputs/metrics, so they are not included as completed runs.

## Top-4 Multiseed Signal

These values are means over three seed rows found in the multiseed result notebooks. Full-test mAP50 was not fully aggregated, so this table focuses on diseased-only and healthy-aware metrics.

| Module | Define attention | Seeds | Disease mAP50 mean +/- std | Disease mAP50-95 mean | Healthy FP mean +/- std | Miss mean | Count MAE mean | HA score mean +/- std | Delta vs baseline | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Triplet Attention Segment Head | Cross-dimension Triplet attention before Segment, modeling C-H, C-W, and H-W interactions. | 3 | 0.449 +/- 0.021 | 0.145 | 28.5% +/- 2.8 pp | 19.3% | 0.426 | 0.370 +/- 0.015 | Disease -0.018, FP -8.1 pp, HA -0.023 | Best multiseed candidate; lower FP is consistent, but miss is higher. |
| SGE-ECA Head Gate | Spatial Group Enhance plus ECA channel selection on P3/P4 before Segment. | 3 | 0.400 +/- 0.013 | 0.127 | 35.8% +/- 2.8 pp | 14.4% | 0.555 | 0.314 +/- 0.019 | Disease -0.067, FP -0.8 pp, HA -0.079 | Stable but not accurate enough. |
| CESA-Lite Segment Head | Legacy segment-head lightweight channel/spatial enhancement with ECA-like and saliency/spatial cues. | 3 | 0.395 +/- 0.027 | 0.125 | 28.5% +/- 13.4 pp | 23.1% | 0.584 | 0.303 +/- 0.018 | Disease -0.071, FP -8.1 pp, HA -0.090 | FP reduction is unstable and miss is high. |
| Context-Suppression Gate Lite | Global pooled context gate on P3/P4 for lightweight healthy-texture suppression. | 3 | 0.385 +/- 0.048 | 0.131 | 42.3% +/- 2.8 pp | 11.7% | 0.586 | 0.296 +/- 0.058 | Disease -0.082, FP +5.7 pp, HA -0.098 | Not recommended as currently trained. |

## Final Recommendation

| Rank | Candidate | Define attention | Why |
|---:|---|---|---|
| 1 | CoTE strong | Triplet-Lite spatial/axis + ECA channel evidence with soft disagreement suppression. | Best balanced result found: disease mAP50 improves over clean baseline and FP decreases, but needs augmentation-controlled confirmation. |
| 2 | Triplet Attention Segment Head | Cross-dimension Triplet attention before Segment. | Best clean/multiseed custom candidate: closest to baseline and consistent FP reduction, though not yet better overall. |
| 3 | LPSC strong | NAM low-FP gate + SimAM saliency + local contrast prior. | Best low-FP option, useful if deployment cost of healthy false positives is higher than disease miss risk. |
| 4 | CESA-Lite Segment Head | Segment-head lightweight channel/spatial attention with ECA-like and saliency cues. | Good single-run clean balance, worth keeping as secondary ablation. |
| 5 | SGE-ECA Head Gate | SGE spatial group-wise enhancement plus ECA channel gate. | Lightweight and reduces FP/miss in single-run, but accuracy needs recovery. |

Not recommended as primary direction from current runs: `SCSG`, `CoLPSC clean/strong`, `P3P4 Semantic Attention Gate`, `CA-Spatial Low-FP Gate`, `Low-FP Residual Spatial Gate`, `Boundary-Aware Lite Attention`, and `CoTE-BL P4 strong`.

Next controlled experiment should rerun the strongest candidates under the exact same protocol:

1. Baseline clean/light, `Triplet`, `CESA-Lite`, `SGE-ECA`, `CoTE`, `LPSC`.
2. Same three seeds.
3. Same threshold sweep.
4. Report full test, diseased-only test, healthy FP, miss rate, count MAE, HA score, params/FLOPs/FPS.
