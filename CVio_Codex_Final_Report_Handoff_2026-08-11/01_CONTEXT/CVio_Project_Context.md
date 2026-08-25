# CVio Project Context - condensed authoritative handoff

Project: **Lightweight Deep Learning for Robust Shrimp Disease Detection on Mobile Devices under Diverse Environmental Conditions**.

Current team:
- Tran Huu Nhan - Project Leader - CE181655
- Le Thi Huynh Nhu - CA182007
- Nguyen Van Phong - CE192081
Supervisor: Nguyen Dinh Vinh.

## Report policy
- Only the integrated Final Report is the current editing target.
- Current baseline: `Report7.pdf` (155 A4 pages in the handoff environment).
- Final report should be a native XeLaTeX project; do not merge already-paginated standalone reports.
- FPT template controls institutional cover/layout. IEEE guidance is secondary for academic/editorial/math/citation quality, not two-column page geometry.

## Council Review 3
Council approved with revisions. Classification-relevant requests include:
- stronger ablation/evidence for ASL-LDAM + SimAM-DCFR;
- real-world images from different environments/cameras;
- explain what “lightweight” means in architecture, params, size, results, and phone configurations;
- explain why proposed Macro-F1 is 91.29 on EXT-3 but 82.18 integrated while CE is 86.51;
- improve parameter/model comparison presentation.

## Classification data/protocol
SDI-4 total 1,149 images:
- Healthy 403
- BG 198
- WSSV 328
- WSSV_BG 220

Fixed seed-42 split:
- train 804
- validation 172
- test 173
Approximately 70/15/15.
Train class counts: Healthy 282, BG 139, WSSV 229, WSSV_BG 154.

EXT-3-Original: 315 images; fixed 221/47/47 split; classes Healthy/BG/WSSV.
Integrated SDI-4 + EXT-3-Original: 1,025/219/220, merged partition-wise after source-wise splitting; WSSV_BG comes only from SDI-4.

## Official classification results
Primary SDI-4 final proposed:
- Macro-F1 91.01%
- Accuracy 91.33%
- Kappa 0.881593
CE baseline:
- Macro-F1 89.02%
- Accuracy 89.02%

EXT-3:
- CE Accuracy 70.21%, Macro-F1 72.30%
- Proposed Accuracy 91.49%, Macro-F1 91.29%

Integrated:
- CE Macro-F1 86.51%, Accuracy 87.727273%
- Proposed Macro-F1 82.18%, Accuracy 83.181818%
This negative result must remain visible.

## Final classifier architecture
YOLO26m-cls + ASL-LDAM + SimAM-DCFR.
Training configuration retained in source evidence:
- imgsz 224
- epochs 30
- seed 42
- batch 32
- workers 4
- AMP true
- AdamW
- lr0 0.00125
- lrf 0.01
- cosine LR
- RandAugment
- erasing 0.4
ASL-LDAM:
- gamma_pos 0
- gamma_neg 4
- label smoothing 0.1
- class counts [282,139,229,154]
- ldam_max_m 0.5
- ldam_scale 30
SimAM-DCFR:
- lambda 1e-4
- inserted before the original Classify head
- SimAM branch parameter-free; texture/channel gate trainable.

## Model-selection positioning
Engineering preference for the final lightweight candidate is approximately <=15M parameters. YOLO26m-cls is about 10.36M before the custom module. Historical broader screening may contain larger candidates; do not present a larger ceiling as the final mobile target without explicitly distinguishing the phases.

## TFLite artifacts
FP32 size about 39.52 MiB; FP16 about 19.81 MiB. Do not call FP32 “quantized”. FP16 is reduced precision/storage.
Mobile evidence remains prototype evidence; do not call Kaggle T4 FPS mobile FPS.

## Known Report7 defects to repair
- duplicate/local+global page numbering and old `9A` artifact;
- mixed fonts / non-embedded fonts;
- corrupted Objective #4;
- Table 1.4 displaced row text;
- Chapter II clipped opening sentence;
- WBS rows with displaced descriptions;
- broken Methodology inserted page;
- formula glyph overlap from prior reconstruction attempt;
- filenames incorrectly changed to monospace;
- merged/missing-space words from PDF extraction;
- too-tight line spacing and giant inter-subsection gaps;
- missing Figure 6.21 imagery in failed repair attempt;
- table cells not consistently centered;
- attention epoch order should be executed/configured;
- remove `3.5 Prioritized Future Work` from Conclusion in this repair baseline.

## Scientific wording boundaries
Use: “under the recorded protocol”, “supports”, “observed improvement”, “condition-dependent”, “does not establish”, “remains a limitation”.
Avoid universal/SOTA/clinical/real-world robustness claims not supported by evidence.
