# Final Report Audit: Segmentation Data and Fourier Experiments

**Audited source:** `CVio_Codex_Final_Report_Handoff_2026-08-11/CVio_Final_Report_2026-08-12.pdf`  
**Audit date:** 2026-08-13  
**Scope:** report formatting, segmentation data and annotation, segmentation model selection, Fourier methodology/results, robustness, inference cost, mobile evidence, and the relevant review-council requests.

## 1. Executive Assessment

The current report is **substantially stronger and more truthful than the earlier version** in the segmentation/Fourier scope. It now contains the grouped-specimen protocol, the mixed-split limitation for the expanded dataset, a technically correct W1 formulation, spatial- and frequency-domain visualizations, a paired spectrum study, a bounded explanation of the W1/augmentation interaction, the final segmentation architecture and metrics, and a small real-world mobile acquisition case study.

It is **not ready for final submission without another formatting pass**. The most visible problems are duplicate page numbers across long page ranges, stale or inconsistent table-of-contents pagination, mixed body-text colors and font families, and several late insertions that disrupt section flow. The main remaining content gaps are the absence of ground-truth evaluation for the real-world phone images, missing camera-resolution metadata, no matched U-Net experiment, an incomplete Fourier-candidate table in Methodology, and a potentially misleading comparison of YOLO-only timing across methods.

Page references below distinguish **PDF physical page** from the **printed footer page**, because the report currently has conflicting page-number fields.

## 2. Formatting Defects and Inconsistencies

### 2.1 Must fix before submission

| Priority | Location | Defect | Required correction |
|---|---|---|---|
| Critical | PDF physical pp. 20-64 and 92-111 | Two page numbers are printed on the same page: a chapter-local counter and a report-level counter. For example, physical p. 92 contains both `1` and `75`. | Remove the chapter-local page-number fields and keep one continuous report-level sequence. Rebuild the TOC, List of Figures, and List of Tables afterward. |
| Critical | TOC and chapter openings | TOC page references do not reliably match the visible footer sequence. Chapter IV is listed at p. 55 while its opening is physically on p. 65 with footer 54; later offsets become larger around Chapter V/VI. | After fixing section breaks and page numbering, update all Word fields rather than manually editing page numbers. Spot-check Chapters IV, V, VI, references, and appendices. |
| High | Throughout, especially physical pp. 15-18, 72-85, 139-155 | Main prose alternates among black, navy, and dark blue. Project Objective #4 is visibly styled differently from adjacent objectives; Table 1.4 and surrounding text are predominantly navy. | Apply named Word styles globally. Body text should use one font, size, and black color. Reserve theme colors for headings only. Clear direct formatting before reapplying styles. |
| High | Physical pp. 93-94, 103-107, 111; pp. 133-134 | Body text changes from the dominant Tinos family to Liberation Serif or Times Roman. | Normalize all body paragraphs and table text to the report's chosen typeface. Do not fix this by selecting all text blindly; update the relevant paragraph/table styles. |

The PDF font inventory confirms the inconsistency rather than a visual impression alone. The dominant body style is approximately **Tinos 11.96 pt**, but large passages use Tinos near 10.9/9.6 pt, Liberation Serif near 11.96 pt, and Times Roman near 11.1 pt. Smaller table text can be intentional, but changing font families and colors across ordinary body paragraphs is not.

### 2.2 Structural and layout defects

| Priority | Location | Defect | Required correction |
|---|---|---|---|
| High | Physical pp. 65-66, printed pp. 54-55 | Table 4.1 begins with only one row at the bottom of p. 65 and continues on p. 66 without a continuation caption. | Keep the caption and at least two rows together, or move the whole table to p. 66. If split, repeat the header and add `Table 4.1 (continued)`. |
| High | Physical p. 72, printed p. 61 | The page begins with an orphaned sentence, then inserts `2.6.1 Multi-Source Retraining Protocol` before Section 3.1 on segmentation annotation. This breaks the logical flow. | Move Section 2.6.1 back into the classification/data subsection and start the segmentation-data section cleanly. |
| High | Physical pp. 140-141, printed pp. 123-124 | Figure 6.15 and Figure 6.15A show the same candidate-screen chart. | Keep one figure only. Place it directly before the interpretation that cites it and update all cross-references. |
| Medium | Physical p. 84, printed p. 73; Appendix F | The 18-panel W1 decomposition is too dense at main-text scale and is repeated as audit material in the appendix. | Keep the concise 8-panel spectrum figure in Methodology and move the full decomposition to the appendix. Reference it from the main text. |
| Medium | Throughout | Figure/table suffixes such as `4.4A`, `6.9B`, `6.15A/B/C`, and `6.20A` are used as late insertion identifiers. | Prefer sequential Word captions. If suffixes must remain, ensure every caption, TOC entry, and cross-reference uses the same identifier. |
| Medium | Physical p. 141, printed p. 124 | `Direct Spectrum Analysis of the W1-Augmentation Interaction` is styled as a subsection but has no subsection number. | Number it consistently, for example `12.3.1`, and update the TOC if that heading level is included. |
| Low | Throughout | Notation alternates between `->`, `=>`, and arrow glyphs; dimensions use both `x` and `×`; Fourier parameters alternate between text names and Greek symbols. | Standardize to `→` in prose/architecture labels, `⇒` for logical implication, `×` for dimensions, and `σ`, `α` after defining the symbols. |

## 3. Status of the Council's Priority Requests

| Council request | Status in current report | Evidence now present | Remaining action |
|---|---|---|---|
| Test real images from different environments and phone cameras | **Partially addressed** | Section 16 reports 21 usable photos acquired using three phone models; one dark image was excluded. Figure 6.20A shows representative inputs/predictions. | The images have no veterinary/laboratory or independent expert ground truth, so this is an operational case study, not an accuracy test. Add original pixel dimensions, camera/module resolution, capture mode, environment/lighting, and device per image. For quantitative claims, annotate a predefined field set and report the same metrics as the held-out test set. |
| Add final-model architecture, size, parameters, key results, and supported mobile configurations | **Addressed, with scope caveat** | Table 6.21 reports YOLO11n-seg + CA→SimAM at P3/P4/P5, 2,854,718 parameters, 9.7 GFLOPs, checkpoint/export sizes, and key segmentation metrics. Tables 6.22-6.23 report three Android devices and observed runtime. | State explicitly whether the measured mobile runtime belongs to the final segmentation model, which export format/backend was used, warm-up/repeats, and whether preprocessing/postprocessing are included. Do not generalize a classification mobile prototype to segmentation. |
| Show image-spectrum changes | **Addressed strongly** | Figures 4.10-4.12 and 6.15B-C show spatial outputs, FFT/log-power views, theoretical gain, decomposition, and matched augmentation-order effects. The paired study reports 24 images × 12 realizations. | Remove the duplicate candidate chart, keep one readable main-text spectrum figure, and move dense audit panels to the appendix. |
| Normalize fonts and sizes | **Not resolved** | None; inconsistencies remain visible and measurable. | Apply Word styles, remove direct formatting, fix duplicate pagination, then regenerate the PDF and visually inspect all pages. |
| Add an evaluation column to the table on slide 39/54 with minimal changes | **Outside this PDF audit** | The report provides metrics and rationale that can populate such a column. | In the slide deck, add one `Assessment/Decision` column using the Table 6.9 trade-offs; do not redesign the slide. Verify the final slide number after recompilation. |
| Explain Table 6.3 and why YOLO11 was selected | **Addressed in the current numbering** | The corresponding comparison is now Table 6.9. The following text explains that YOLO11n-seg led full-test mAP50, labeled-only mAP50, and HScore, while other candidates led mAP50-95 or Healthy FP. | In the response letter, say that the board's `Table 6.3` is `Table 6.9` in the revised report. Retain the trade-off explanation; do not claim YOLO11n-seg is best on every metric. |
| Compare segmentation with U-Net if time permits | **Not performed** | The report explicitly records this as a limitation and explains the semantic-vs-instance evaluation mismatch. RTMDet-Ins-S provides a cross-architecture instance-segmentation comparator, but it is not U-Net. | Do not add an unverified U-Net row. Only run U-Net on the same manifest and report semantic Dice/IoU/mIoU, then define any instance-conversion procedure before comparing instance metrics. Otherwise retain the limitation and describe the request as future work. |

## 4. Segmentation Data and Annotation Audit

### 4.1 Content that is adequately documented

- The original dataset is reported as 1,149 images from 416 specimen groups, including 746 disease-labeled images, 403 Healthy empty-label negatives, and 1,031 mask instances.
- BG and WSSV are correctly described as the two positive mask classes. WSSV_BG is not incorrectly introduced as a third mask class.
- The report correctly states that Ultralytics segmentation can represent multiple instances/classes in one image; the single-label overlap policy is presented as a project annotation decision rather than a model-format impossibility.
- The deterministic WSSV-priority rule for overlapping co-infection regions is explained.
- The principal split uses 905/115/129 images and 331/40/45 specimen groups with no convention-matched group overlap.
- The expanded-dataset experiment clearly distinguishes 1,149 convention-matched images from 303 unmatched images and acknowledges that the random-stratified unmatched subset cannot guarantee specimen-level isolation.
- The absence of veterinary review and inter-annotator agreement is disclosed rather than hidden.

### 4.2 Corrections needed

1. **Fix the grouped-split equation.** On physical p. 74 / printed p. 63, replace the juxtaposed expression with:

   `G_i = G_j ⇒ S_i = S_j`

   This states the intended rule: if two images share a specimen group, they must share the same split.

2. **Resolve the annotator inconsistency.** The Methodology says masks were created by the "project team," while limitations later refer to "one annotator." Use the verified fact consistently. If one person annotated and the team reviewed informally, state that explicitly; do not imply independent multi-annotator agreement.

3. **Correct the Methodology opening.** The claim that instance segmentation provides evidence for "evaluation leakage" conflates model output with split auditing. Suggested wording:

   > Instance segmentation provided spatial evidence for localized disease signs, while evaluation leakage was controlled through specimen-grouped partitioning and explicit split-overlap checks.

4. **Strengthen visual annotation evidence.** Figure 4.5 shows BG and WSSV examples but does not visibly demonstrate the two policies most likely to be questioned. Add one compact four-panel figure: BG-only mask, WSSV-only mask, co-infection with the WSSV-priority overlap, and Healthy image with an empty label. Use an enlarged crop for the overlap case.

5. **Keep the expanded dataset bounded.** Continue describing it as a dataset-composition transfer assessment, not proof of cross-farm, cross-device, or field generalization.

## 5. Fourier Methodology and Results Audit

### 5.1 Claims supported by recorded evidence

- W1 is correctly formulated as a mild residual high-frequency boost, not a binary high-pass image: `x_W1 = clip[x + α(x - x_low), 0, 255]`, with `σ=50` and `α=0.10`.
- Under the matched seed-42, no-augmentation, hidden-hook-off setting, W1 increased labeled mask mAP50 from **23.54% to 34.51%** (`+10.97 pp`) but increased Healthy FP from **39.02% to 63.41%**. The report therefore does not present the gain as cost-free.
- Under the matched strong-augmentation, hook-off setting, W1 changed labeled mask mAP50 from **59.51% to 57.71%** (`-1.80 pp`) while Healthy FP changed from **34.15% to 31.71%**.
- The paired spectrum experiment supports the redundancy hypothesis without overstating causality: W1's mean global high-band increment fell from `0.000670` without strong augmentation to `0.000151` after matched strong augmentation, retaining **22.5%**. Laplacian variance retained **21.1%**, mean gradient **56.1%**, and lesion-region gradient approximately **79.0%**.
- The report correctly says this is seed-42, mechanism-consistent evidence rather than multi-seed causal proof.

### 5.2 Corrections and additions needed

1. **Complete the Fourier candidate table.** Table 4.9 lists high-pass enhancement, high-frequency damping, band-pass filtering, and low-frequency flattening, but the Results candidate table also evaluates homomorphic filtering and Gaussian low-pass. Add both Methodology rows and their recorded configurations so the Methods and Results populations match.

2. **Do not attribute all model-only timing differences to Fourier.** Table 6.18 reports W1 preprocessing separately, which is correct. However, W1's YOLO prediction time also differs markedly from the other YOLO models despite the shared architecture. Only the measured Fourier preprocessing component is directly interpretable as W1 overhead. Add this note below the table:

   > The Fourier preprocessing time is the directly attributable W1 overhead. Differences in the reported YOLO-only prediction time across rows may also reflect predictor initialization, warm-up, data-loader state, preprocessing, synchronization, or run-order effects. They should not be interpreted as an architectural inference penalty caused by W1 without a randomized, repeated timing benchmark using the same loaded model runner.

3. **Clarify which model is final.** The report's final segmentation model is YOLO11n-seg + CA→SimAM, not W1. Keep W1 as a controlled preprocessing finding and explicitly state that strong augmentation made W1 unnecessary for the selected final pipeline.

4. **Add the noisy-test evidence only if robustness is discussed as a Fourier result.** The current robustness table highlights selected gains for the attention model. It does not provide the complete Fourier-vs-non-Fourier corruption panel already recorded in the project evidence. A concise appendix table can report, for every corruption, full mAP50, labeled mAP50, labeled mAP50-95, Healthy FP, and HScore for W1 and matched comparators. If space is limited, omit the claim rather than showing only favorable Fourier cases.

5. **Keep the mechanism language conditional.** Strong augmentation introduces its own color, interpolation, boundary, and geometric-frequency changes. The measured spectra show overlap/reduction of W1's incremental effect, but they do not prove that any single augmentation operation caused the mAP decrease.

6. **Move FDDEM out of the central validated narrative.** The current report correctly labels it exploratory and avoids unaudited numerical claims. It can remain in an appendix or future-work paragraph, but it should not appear alongside validated W1 results as if it were an equally established contribution.

## 6. Final Model and Mobile Evidence

The current report adequately supplies a compact final-model summary, but the deployment claim must be split into three evidence levels:

1. **Model complexity:** 2,854,718 parameters, 9.7 GFLOPs, approximately 6.1 MB checkpoint, and 5.65 MB FP16 TFLite. This supports the term *compact*.
2. **Observed device execution:** identify the exact final segmentation export, runtime/backend, input size, thread/delegate settings, preprocessing, postprocessing, warm-up count, timed repetitions, and per-device latency distribution. This supports *runs on the tested devices*.
3. **Field accuracy:** requires labeled real-world images. The present 21-photo test supports qualitative operational feasibility only, because no independent disease-region ground truth was available.

Add camera metadata to the real-world table: phone model, camera module or nominal resolution, original file dimensions, capture environment, lighting condition, distance, orientation, and whether the image was resized or compressed before inference. Do not reconstruct unavailable metadata from marketing specifications; report it as unavailable.

## 7. Recommended Revision Order

1. Fix section breaks, duplicate page numbers, and TOC/List fields.
2. Normalize body font, size, and color using Word styles; specifically inspect Project Objective #4 and Table 1.4.
3. Repair the Methodology flow around physical pp. 65-74 and the Table 4.1 split.
4. Apply the segmentation-data corrections: implication formula, annotator wording, leakage sentence, and annotation-policy figure.
5. Complete Table 4.9 and remove duplicate Figure 6.15/6.15A.
6. Add the timing-comparability caveat and explicitly distinguish the final attention model from W1.
7. Expand the real-world metadata table and preserve the no-ground-truth limitation.
8. Decide whether the full Fourier noisy-test table belongs in the appendix; avoid selective robustness claims.
9. Keep U-Net as an explicit uncompleted conditional request unless a protocol-matched experiment is actually run.
10. Re-export the PDF, update all fields, and conduct a final page-by-page visual check.

## 8. Submission Readiness Verdict

**Scientific/content readiness in the audited scope:** conditionally ready after the listed corrections. The core segmentation and Fourier findings are now mostly evidence-aligned and appropriately bounded.

**Formatting readiness:** not ready. Duplicate pagination, stale navigation fields, and mixed typography are submission-level defects.

**Council-request completeness:** spectrum analysis and YOLO11 selection are addressed; final-model/mobile information is substantially addressed; real-world evaluation is only qualitative; U-Net remains unperformed; the slide-table request must be verified in the presentation file separately.
