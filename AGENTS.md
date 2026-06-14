# Shrimp Disease Segmentation Capstone - Agent Context

## Project Objective
Improve and optimize a YOLOv11n-seg baseline model for automated shrimp disease segmentation to achieve better detection accuracy, speed, and robustness under fair specimen-grouped evaluation protocol.

## Current State & Key Findings

### Baseline Established ✓
- **Model**: YOLOv11n-seg with clean-light YOLO augmentation
- **Dataset**: Hand-labeled shrimp disease segmentation (1149 images, 416 specimens, 1031 masks)
- **Split Protocol**: Stratified grouped-specimen split (leakage-safe)
- **Seed**: Primary 42, to be confirmed across multiple seeds
- **Validation Strategy**: Healthy-aware score (minimizes false positives on healthy specimens)

### Baseline Metrics (Reference)
| Metric | Value |
|---|---:|
| Full test mask mAP50 | 0.481532 |
| Labeled-only test mask mAP50 | 0.512002 |
| Labeled-only test mask mAP50-95 | 0.167785 |
| Healthy test false positive rate | 0.243902 |
| Healthy-aware labeled test score | 0.459581 |

### Dataset Quality Insights
- Hand-labeled dataset contains 220 co-infection images (both BG + WSSV)
- EWU prior dataset is unreliable: incomplete labeling, inconsistent masks per specimen
- Visual-hash analysis confirms EWU and hand-labeled show different annotation standards

## Current Focus: Preprocessing & Augmentation (Phase B: YOLO Policy Screening)

### Primary Hypothesis
> Shrimp disease segmentation improves under fair grouped-specimen evaluation when augmentation is tuned toward disease visibility, healthy false-positive control, and small-data generalization—not through generic preprocessing but through disease-aware YOLO policy search.

### Research & Screening Strategy (NEW)

**We have conducted a comprehensive literature review of recent (2023-2026) methods** across:
- Segmentation-specific augmentation (MaskMix, CopyPaste variants)
- Medical/domain-specific preprocessing (color normalization, CLAHE)
- Small-data optimization strategies
- YOLOv11+ ecosystem advances
- Frequency-domain methods
- AutoML policy search approaches

**Deliverables**:
1. RESEARCH_SYNTHESIS_WEB_BASED_2023_2026.md - 22 methods evaluated from CVPR/ICCV/ECCV/MICCAI 2024
2. PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md - Ready-to-use checklist with 16 methods across 6 phases

### Screening Checklist: 16 Methods in 6 Phases

**PHASE B (Weeks 1-2): YOLO NATIVE + QUICK WINS** ← You are here
- 9 methods | ~40-60 GPU hours | HIGH PRIORITY
- Key methods: Auto-Augment, Color Normalization (Macenko/Reinhard), CLAHE, GridMask, Mosaic, HSV, Geometric, Copy-Paste
- Expected gain: +0.02 to +0.06 mAP50

**PHASE C (Weeks 2-3): COPY-PASTE FOCUSED VARIANTS** (conditional)
- 3 methods | ~20-30 GPU hours
- Key methods: MaskMix, CopyPaste in Context, Augmentation Scheduling
- Expected gain: +0.01 to +0.04 mAP50

**PHASE D (Weeks 3-4): DOMAIN-SPECIFIC EXTENSIONS** (on-demand)
- Test only if Phase B+C combined gain < +0.03 mAP50

**PHASE E (Weeks 4-5): FREQUENCY-DOMAIN METHODS** (on-demand, low priority)
- Test only if plateau observed

**PHASE F (Weeks 5-6): MULTI-SEED CONFIRMATION** (critical)
- Top 2-3 methods across 5+ grouped-split seeds
- Report mean ± std, 95% CI

### Planned Experiment Phases (in order)

**Phase B: YOLO Augmentation Policy Screening** ← Current
- Run 9 candidates on baseline split and model
- Key hyperparameters: Auto-augment, color norm, CLAHE, mosaic, copy_paste, HSV, geometric
- Best 2-3 policies move to Phase C & multi-seed confirmation
- Success metric: +0.02 to +0.05 absolute labeled-only mAP50 improvement

**Phase C: Copy-Paste Focused Sweep** (if Phase B shows copy-paste promise)
- Fine-tune copy-paste modes and variants
- Visual validation on augmented training images

**Phase D: Segmentation-Safe Photometric & Domain-Specific** (if needed)
- Test color/brightness robustness policies
- Build robustness tables for noisy/compressed test images

**Phase E: Frequency/Wavelet Revisit** (lower priority, if plateau)
- Limited variants only: Fourier train-only, wavelet detail-enhancement
- Decide whether to include in main paper or appendix

**Phase F: Multi-Seed Confirmation** (critical for reproducibility)
- Top 2-3 methods run across 3+ grouped-specimen seeds
- Report mean, std, confidence intervals

## Working Patterns & Cues

### Notebook Development
- Keep Colab-ready; must run RUN ALL successfully
- Always download: main CSV, split manifests, best.pt, results.csv
- Include boolean flags to enable/disable experiment groups
- Include smoke-test mode for quick validation
- Version-lock Ultralytics to match baseline

### Evaluation Discipline
- **Always compare to baseline** before claiming improvement
- **Validate on held-out specimens** (grouped split)
- Healthy-aware score should not decrease (avoid false positive increase)
- Track both validation and test metrics separately

### Augmentation Testing on Shrimp
- Test on actual disease samples to ensure visibility isn't lost
- Color/photometric changes matter (disease appearance varies with lighting)
- Vertical flip should be tested carefully (not all orientations natural)
- Mosaic in early epochs OK; close it late for clean localization
- Copy-paste within same image safer than cross-image for small dataset

### Dataset Leakage Prevention
- Ensure split respects specimen groups (no same-shrimp in train/val/test)
- Grouped manifest should be deterministic and saved
- Confirm no visual duplicates between splits

### Using the Screening Checklist
1. Read the method's test card in the checklist
2. Mark Status = IN_PROGRESS
3. Run experiment with specified config/parameters
4. Record exact results (mAP50, mAP50-95, Healthy FP)
5. Decide: KEEP (+gain ≥ 0.005 mAP50) or DROP
6. Fill summary table after each phase
7. Proceed to next method or phase

## Project Structure

`
C:\Users\Admin\workspace\CVio_Shrimp_Disease_Classification_Capstone_SU26\
├── shrimp-leakage-aware-segmentation/
│   ├── notebooks/
│   │   ├── baseline/
│   │   ├── copy_paste_focused_sweep/
│   │   ├── frequency_wavelet_revisit/
│   │   ├── photometric_randaug/
│   │   ├── yolo_aug_policy_search/
│   │   ├── model-sweep/
│   │   └── split-policy/
│   └── paper/
│
├── augmentation_research/baseline_opt/
│   └── super_paper_fix_leakage_fourier_fair_optimization/
│       ├── augmentation_preprocessing_optimization_subplan.md  (master plan)
│       ├── research-findings.md  (dataset quality insights)
│       ├── research-plan.md  (full context)
│       └── [notebooks, results, checkpoints]
│
├── AGENTS.md  ← You are here (project context & working patterns)
├── RESEARCH_SYNTHESIS_WEB_BASED_2023_2026.md  ← 22 methods reviewed from literature
└── PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md  ← 16-method screening plan (Phase B-F)
`

## Quick Reference
- **Roboflow Dataset**: https://app.roboflow.com/lets-try-this/shrimpdishandsegv2
- **Model**: YOLOv11n-seg
- **Screening Checklist**: PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md (START HERE for Phase B)
- **Literature Review**: RESEARCH_SYNTHESIS_WEB_BASED_2023_2026.md (22 methods with papers)
- **Baseline mAP50**: 0.512 (labeled-only)
- **Primary Augmentation Hyperparams to Tune**: hsv_h, hsv_s, hsv_v, mosaic, close_mosaic, copy_paste, degrees, scale, translate
- **Validation Focus**: Healthy-aware score (balance mAP50 with false positive control)
- **Success Threshold**: +0.03 absolute labeled-only mAP50 with no healthy FP increase
- **Next Action**: Start Phase B testing with Method 1.1 (Auto-Augment) - quickest test

## Important: Before Starting Phase B Experiments
1. Confirm grouped-split seed 42 manifest exists and is deterministic
2. Verify Ultralytics version matches baseline runs
3. Confirm hidden Albumentations hook is enabled in YOLO config
4. Document exact training command and config before each run
5. Save split manifests alongside results for reproducibility
6. Read the relevant test card in PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md before implementing
