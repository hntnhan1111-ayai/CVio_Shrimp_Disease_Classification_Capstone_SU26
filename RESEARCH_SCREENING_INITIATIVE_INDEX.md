# Research & Screening Initiative Index
# Shrimp Disease Segmentation - 2023-2026 Literature Review
# Created: June 14, 2026

---

## QUICK START

**You are here**: Phase B - YOLO Native + Quick Wins

**Next action**: 
1. Open PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md
2. Read Method 1.1 (Auto-Augment) test card
3. Run first experiment with one-line YOLO config change
4. Record results in the checklist

**Expected outcome**: 12-16 methods tested, +0.03-0.08 mAP50 gain, 5-6 weeks

---

## DOCUMENT GUIDE

### 1. AGENTS.md
**What**: Project context, baseline, working patterns
**Who should read**: Everyone (start here)
**When to use**: Reference for overall strategy and team coordination

**Contains**:
- Project objective and current state
- Baseline metrics (mAP50: 0.512)
- 6-phase testing strategy overview
- Working patterns and cues
- File structure

**Action**: Already read? → Move to document 2

---

### 2. RESEARCH_SYNTHESIS_WEB_BASED_2023_2026.md
**What**: Comprehensive literature review of recent methods (2023-2026)
**Who should read**: Technical lead, those wanting deep understanding
**When to use**: To understand why each method is on the checklist

**Contains**:
- 22 methods across 5 domains:
  - Segmentation-specific augmentation (4 methods)
  - Small-data optimization (3 methods)
  - Medical/domain-specific preprocessing (6 methods)
  - YOLO ecosystem (4 methods)
  - Frequency-domain methods (3 methods)
  - AutoML policy search (2+ methods)
  
- For each method:
  - Paper title, authors, year, venue
  - Core idea (1-2 sentences)
  - Why it matters for segmentation
  - Expected performance gain
  - Implementation complexity
  - Status (published, research, cutting-edge)

- Summary table: Priority, expected gain, complexity, phase assignment

**Action**: Skim section headers. Read full sections for methods you're testing.

---

### 3. PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md ⭐ PRIMARY DOCUMENT
**What**: Ready-to-use testing checklist for 16 methods across 6 phases
**Who should read**: Implementation team (the one running experiments)
**When to use**: DURING and AFTER each experiment

**Contains**:
- **PHASE B** (9 methods, HIGH PRIORITY):
  - 1.1 Auto-Augment (YOLO RandAugment native)
  - 1.2 Color Normalization - Macenko
  - 1.3 Color Normalization - Reinhard
  - 1.4 Learnable CLAHE
  - 1.5 GridMask Augmentation
  - 1.6 Mosaic & Close-Mosaic Tuning
  - 1.7 HSV Color Space Tuning
  - 1.8 Geometric Augmentation
  - 1.9 Copy-Paste Native Tuning

- **PHASE C** (3 methods, CONDITIONAL):
  - 2.1 MaskMix
  - 2.2 CopyPaste in Context
  - 2.3 Augmentation Scheduling

- **PHASE D** (2 methods, ON-DEMAND):
  - Domain-specific extensions (test if plateau)

- **PHASE E** (2 methods, ON-DEMAND):
  - Frequency-domain methods (test if plateau)

- **PHASE F**:
  - Multi-seed confirmation (top 2-3 methods)

**Each method card includes**:
- Status (PENDING/IN_PROGRESS/DONE)
- Paper/source citation
- Core idea and why it matters
- Expected gain (from literature)
- Implementation details (config, code snippets, parameters)
- Test date, results, decision tracking
- Notes field for observations

**Summary tables** at phase end:
- Method | mAP50 | Δ mAP50 | Decision | Notes

**Decision rules**:
- KEEP if Δ mAP50 ≥ +0.005 AND Healthy FP doesn't increase
- DROP if Δ mAP50 < +0.003 OR visual artifacts
- COMBINE if no negative interaction

**Action**: Use this as your lab notebook. Fill in results as you test.

---

## PHASE-BY-PHASE SUMMARY

### PHASE B: YOLO NATIVE + QUICK WINS (Weeks 1-2)
**Methods**: 9 (1.1 to 1.9)
**GPU Hours**: 40-60
**Status**: ⭐ START HERE
**Expected Gain**: +0.02 to +0.06 mAP50 combined

**What to do**:
1. For each method, read the test card in CHECKLIST
2. Mark Status = IN_PROGRESS
3. Configure YOLO/preprocessing as specified
4. Train 1-3 models (depending on grid search)
5. Record results: mAP50, mAP50-95, Healthy FP, decision
6. Mark Status = DONE
7. Move to next method

**Easiest first**:
- 1.1 Auto-Augment (one-line YOLO config)
- 1.4 CLAHE (simple preprocessing)
- 1.5 GridMask (one-line Albumentations)

**High-impact methods**:
- 1.2 Macenko (color normalization - medical gold standard)
- 1.9 Copy-Paste (segmentation-specific)

**Decision point after Phase B**:
- If combined gain ≥ +0.03 mAP50 → Excellent! Move to Phase F (multi-seed)
- If combined gain +0.01 to +0.03 mAP50 → Good! Consider Phase C (copy-paste variants)
- If combined gain < +0.01 mAP50 → Investigate why; consider Phase D/E

---

### PHASE C: COPY-PASTE FOCUSED VARIANTS (Weeks 2-3)
**Methods**: 3 (2.1 to 2.3)
**GPU Hours**: 20-30
**Status**: CONDITIONAL (only if Phase B copy-paste shows promise)
**Expected Gain**: +0.01 to +0.04 mAP50

**Run only if**: Method 1.9 (Copy-Paste Native) achieves Δ mAP50 ≥ +0.01

**What to do**:
1. Read 2.1, 2.2, 2.3 test cards
2. Implement each variant
3. Compare against Phase B copy-paste result
4. Pick best; keep for Phase F confirmation

---

### PHASE D: DOMAIN-SPECIFIC EXTENSIONS (Weeks 3-4)
**Methods**: 2 (3.1, 3.2)
**Status**: ON-DEMAND
**Run only if**: Phase B+C combined gain < +0.03 mAP50

**What to test**:
- Augmentation-Invariant Parameter Scheduling
- Shadow & Illumination Correction (if dataset inspection shows issues)

---

### PHASE E: FREQUENCY-DOMAIN METHODS (Weeks 4-5)
**Methods**: 2 (4.1, 4.2)
**Status**: ON-DEMAND, LOW PRIORITY
**Run only if**: Plateau observed (no improvement from B/C/D)

**Note**: You tested Fourier extensively before. Only revisit if absolutely necessary.
- 4.1 Frequency Dropout (stochastic variant)
- 4.2 Wavelet Enhancement

---

### PHASE F: MULTI-SEED CONFIRMATION (Weeks 5-6) ⭐ CRITICAL
**Methods**: Top 2-3 from Phase B+C
**GPU Hours**: 20-30
**Status**: REQUIRED for publishable results

**What to do**:
1. Identify best 2-3 methods from earlier phases
2. Run each with seeds: 42, 123, 456 (or up to 5+ seeds)
3. Record mean ± std and 95% CI
4. Report in final paper with confidence bounds

**Why this matters**:
- Proves reproducibility
- Shows robustness across random seeds
- Reviewers will ask for this

---

## TIMELINE & EFFORT ESTIMATE

| Phase | Duration | Methods | GPU Hours | Status |
|-------|----------|---------|-----------|--------|
| **B** | Weeks 1-2 | 9 | 40-60 | ⭐ START |
| **C** | Weeks 2-3 | 3 | 20-30 | Conditional |
| **D** | Weeks 3-4 | 2 | 10-20 | On-demand |
| **E** | Weeks 4-5 | 2 | 10-20 | On-demand |
| **F** | Weeks 5-6 | Top 3 | 20-30 | Required |
| | | | | |
| **TOTAL** | **5-6 weeks** | **12-16** | **80-150** | |

---

## HOW TO TRACK PROGRESS

### Daily Checklist
- [ ] Reviewed today's method test card
- [ ] Ran training (or in progress)
- [ ] Saved results (best.pt, results.csv)
- [ ] Recorded mAP50 in checklist
- [ ] Recorded decision (KEEP/DROP)
- [ ] Updated phase summary table

### Weekly Summary
- [ ] Completed N methods
- [ ] Phase B progress: ___ / 9 complete
- [ ] Top performer so far: ___________ (Δ mAP50: _______)
- [ ] Any blockers? _____________________________________________________

### Monthly Report (Optional)
- Title:  Augmentation Research Progress - [Month]
- Include: Methods tested, best gains, next priorities
- Share with advisor/team

---

## KEY DECISION RULES

### KEEP a method if
✓ Δ mAP50 ≥ +0.005 (0.5 percentage point improvement)
✓ AND Healthy FP rate doesn't increase by > 0.02
✓ AND visual inspection shows reasonable augmentation

### DROP a method if
✗ Δ mAP50 < +0.003 (marginal improvement)
✗ OR Healthy FP rate increases significantly
✗ OR augmentation creates unrealistic images

### COMBINE methods if
✓ Both improve independently
✓ They don't conflict (e.g., two color norms might interfere)
✓ Combined effect ≥ sum of individuals (check for interaction)

---

## LITERATURE CITATIONS (For Your Paper)

**Cite when reporting results**:
- YOLO v11: Ultralytics (2024)
- MaskMix: Kim et al., ICCV 2024
- Color Normalization: Gamper et al., MICCAI 2024; Macenko et al. 2009
- Copy-Paste: Ghiasi et al. 2021; Wang et al. ICCV 2023
- GridMask: Chen et al., ICCV 2020
- Frequency Dropout: Lim et al., CVPR 2024
- Augmentation Scheduling: Zhang et al., CVPR 2024

---

## TROUBLESHOOTING

**Q: Method 1.1 (Auto-Augment) not improving, why?**
A: YOLO might not have auto_augment parameter. Check Ultralytics docs for your version. Fall back to manual HSV tuning (1.7).

**Q: CLAHE making images look weird, what to do?**
A: Reduce clip_limit (try 2.0 instead of 8.0) or reduce apply probability (p=0.3 instead of 0.5).

**Q: Copy-paste regions looking unnatural, how to fix?**
A: Add Gaussian blur to edges (boundary blending) - this is Method 2.2 (CopyPaste in Context).

**Q: Color normalization broke my images, what went wrong?**
A: May be applying too aggressively. Verify reference image is representative. Test Reinhard (simpler) before Macenko.

**Q: Plateau in Phase B, what now?**
A: Move to Phase C (copy-paste variants). If still plateau, try Phase D (parameter scheduling). Phase E (frequency) is lower priority.

---

## FILES IN THIS INITIATIVE

`
C:\Users\Admin\workspace\CVio_Shrimp_Disease_Classification_Capstone_SU26\
├── AGENTS.md (project context + references)
├── RESEARCH_SYNTHESIS_WEB_BASED_2023_2026.md (22 methods reviewed)
├── PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md (YOUR LAB NOTEBOOK)
├── This file: RESEARCH_SCREENING_INITIATIVE_INDEX.md
└── augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/
    ├── augmentation_preprocessing_optimization_subplan.md
    ├── research-findings.md
    └── [notebooks, configs, checkpoints]
`

---

## NEXT IMMEDIATE ACTIONS

1. **Open** PREPROCESSING_AUGMENTATION_CHECKLIST_FINAL.md
2. **Read** the PHASE B section header
3. **Go to** Method 1.1 (Auto-Augment test card)
4. **Create** a YOLO training script with:
   `yaml
   model: yolov11n-seg
   augmentation:
     auto_augment: 'randaugment'
   `
5. **Run** training (1 epoch for smoke test first)
6. **Record** results in the checklist table
7. **Mark** Status = DONE
8. **Move** to Method 1.2

---

## Questions or Blockers?

If you hit a blocker:
1. Check the method's test card in CHECKLIST - implementation details are there
2. Check RESEARCH_SYNTHESIS - understand why the method matters
3. Check troubleshooting section above
4. Document the blocker and skip to next method (can revisit later)

Remember: **The goal is systematic testing, not perfection on any single method.**

Good luck! 🦐🔬📊
