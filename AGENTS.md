# Shrimp Disease Segmentation Capstone - Agent Context

## Project Objective
Improve and optimize a YOLOv11n-seg baseline model for automated shrimp disease segmentation to achieve better detection accuracy, speed, and robustness under fair specimen-grouped evaluation protocol.

## Active Working Folder

The current GitHub-facing working folder is:

`shrimp-leakage-aware-segmentation/`

Use this folder as the primary location for paper-ready code, notebooks, manuscript files, repo documentation, and current research plans. Older files under `augmentation_research/baseline_opt/` are historical/exploratory context unless the user explicitly asks to work there.

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

## Current Focus: Preprocessing & Augmentation Optimization

### Primary Hypothesis
> Shrimp disease segmentation may improve under fair grouped-specimen evaluation when augmentation is tuned around the current hook-enabled clean-light baseline, disease visibility, healthy false-positive control, and small-data generalization.

### Research & Screening Strategy

The earlier broad literature review remains useful as an idea bank, but current experiments show the hook-enabled clean-light YOLO baseline is strong. The active optimization plan is therefore targeted rather than broad.

Primary current plan:

- Start from the accepted hook-enabled clean-light baseline.
- Prioritize low-risk YOLO-native policy tuning.
- Treat healthy false positives as a hard practical constraint.
- Move to hard-negative mining if generic augmentation does not improve the baseline.
- Confirm only successful candidates across multiple grouped-specimen seeds.

Current active research plan:

- `shrimp-leakage-aware-segmentation/research/optimization_deep_research_2026.md`

Earlier research package / idea bank:
- Segmentation-specific augmentation (MaskMix, CopyPaste variants)
- Medical/domain-specific preprocessing (color normalization, CLAHE)
- Small-data optimization strategies
- YOLOv11+ ecosystem advances
- Frequency-domain methods
- AutoML policy search approaches

**Deliverables**:
0. `shrimp-leakage-aware-segmentation/research/optimization_deep_research_2026.md` - active targeted optimization plan based on current evidence
1. RESEARCH_SYNTHESIS_WEB_BASED_2023_2026.md - 22 methods evaluated from CVPR/ICCV/ECCV/MICCAI 2024
2. PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md - Ready-to-use checklist with 16 methods across 6 phases

### Active Targeted Optimization Checklist

**Phase 1: YOLO-native policy tuning**
- low mosaic + close mosaic
- copy-paste + low mosaic
- conservative HSV/geometric middle policies
- hook remains enabled unless the experiment is explicitly a hook-ablation control

**Phase 2: Healthy hard-negative mining**
- collect healthy images where the baseline predicts false masks
- retrain with emphasized/duplicated healthy negatives
- accept only if healthy FP improves without collapsing diseased mAP

**Phase 3: Candidate combinations**
- combine only methods that individually help
- reject combinations that increase healthy FP, disease miss rate, or count error

**Phase 4: Threshold/calibration sweep**
- run for baseline and finalists
- report healthy FP, missed disease, and count MAE across confidence thresholds

**Phase 5: Multi-seed confirmation**
- run only after a candidate beats baseline on seed 42
- minimum seeds: 42, 123, 3407

### Planned Experiment Phases (in order)

Current recommended next notebook:

- `shrimp-leakage-aware-segmentation/notebooks/yolo_aug_policy_search/yolo_aug_policy_search_v2_targeted.ipynb`

It should test only a compact baseline-centered search:

1. baseline sanity row
2. low mosaic candidates
3. copy-paste + low mosaic candidates
4. conservative HSV/geometric middle policies

Do not expand into Fourier, wavelet, broad photometric copies, or diffusion augmentation unless the user explicitly changes direction.

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
│   ├── research/
│   │   └── optimization_deep_research_2026.md  (active targeted optimization plan)
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
- **Working repo**: shrimp-leakage-aware-segmentation/
- **Active optimization plan**: shrimp-leakage-aware-segmentation/research/optimization_deep_research_2026.md
- **Screening Checklist**: PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md (older broad checklist / idea bank)
- **Literature Review**: RESEARCH_SYNTHESIS_WEB_BASED_2023_2026.md (22 methods with papers)
- **Baseline mAP50**: 0.512 (labeled-only)
- **Primary Augmentation Hyperparams to Tune**: hsv_h, hsv_s, hsv_v, mosaic, close_mosaic, copy_paste, degrees, scale, translate
- **Validation Focus**: Healthy-aware score (balance mAP50 with false positive control)
- **Success Threshold**: improve healthy-aware score or achieve at least +0.010 labeled-only mAP50 with no meaningful healthy FP increase
- **Next Action**: generate/run targeted YOLO-native augmentation policy notebook

## Important: Before Starting Optimization Experiments
1. Confirm grouped-split seed 42 manifest exists and is deterministic
2. Verify Ultralytics version matches baseline runs
3. Confirm hidden Albumentations hook is enabled in YOLO config
4. Document exact training command and config before each run
5. Save split manifests alongside results for reproducibility
6. Read `shrimp-leakage-aware-segmentation/research/optimization_deep_research_2026.md` before implementing
