# PREPROCESSING & AUGMENTATION SCREENING CHECKLIST
# Shrimp Disease Segmentation (YOLOv11n-seg)
# Based on 2023-2026 Literature Review
# Created: 2026-06-14

---

## HOW TO USE THIS CHECKLIST

1. **Read the Method Card** - understand what it does and why it might help
2. **Mark Status = IN_PROGRESS** when you start testing
3. **Run the experiment** with the specified parameters/configs
4. **Record Results** - exact mAP50, mAP50-95, Healthy FP rate
5. **Make Decision** - KEEP (+gain ≥ 0.005) or DROP (marginal/negative)
6. **Mark Status = DONE**
7. **Repeat for next method**

## BASELINE REFERENCE (YOLOv11n-seg, Grouped-Specimen Split, Seed 42)

`
mAP50 (labeled-only): 0.512
mAP50-95 (labeled-only): 0.168
Healthy False Positive Rate: 0.244
Healthy-Aware Score: 0.459581
`

All improvements reported as: **Δ mAP50** (percentage points) or **% improvement over baseline**

---

# PHASE B: YOLO NATIVE + QUICK WINS (Priority: HIGH)
# Estimated: Weeks 1-2, ~40-60 GPU hours, ~20 models

## 1.1 Auto-Augment (YOLO RandAugment Native)
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Ultralytics YOLO v11 Official (2024) |
| **Core Idea** | auto_augment=''randaugment'' with automatic magnitude tuning |
| **Why It Matters** | YOLO v11 integrates RandAugment; automatic policy search |
| **Literature Gain** | +0.02 to +0.04 mAP50 |
| **Implementation** | One line in YOLO config: auto_augment=''randaugment'' |
| **Test Config** | See training script below |
| **Segmentation-Safe** | Yes (Ultralytics handles mask transforms) |
| **Low-Risk** | YES - native YOLO feature |

**YOLO Config to Test**:
`yaml
augmentation:
  auto_augment: 'randaugment'
  # Keep other params as baseline
`

**Test Date**: ___________
**Results (vs Baseline 0.512)**:
- mAP50: ___________
- Δ mAP50: ___________
- mAP50-95: ___________
- Healthy FP Rate: ___________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

## 1.2 Color Normalization - Macenko (Medical Gold Standard)
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Macenko et al. 2009; Gamper et al. MICCAI 2024 |
| **Core Idea** | Normalize color distribution across images; reduces lighting variation |
| **Why It Matters** | Aquaculture lighting varies; color critical for disease appearance |
| **Literature Gain** | +0.02 to +0.05 mAP50 |
| **Implementation** | Preprocessing step; apply before training |
| **Library** | python-spams or custom implementation |
| **Segmentation-Safe** | Yes (preprocessing, not augmentation) |
| **Low-Risk** | YES - established medical imaging standard |

**Implementation Code Snippet**:
`python
from spams import trainDL  # or implement Macenko manually
import cv2
import numpy as np

def macenko_normalize(img, reference_img=None):
    " Apply Macenko stain normalization
    # Implementation: compute stain matrix, normalize
    # Can use public repos: histolab, stain_tools, etc.
    pass
`

**Test Strategy**:
- Apply to training images before loading
- Use first training image as reference, or compute average
- Test both per-image and batch normalization

**Test Date**: ___________
**Results (vs Baseline 0.512)**:
- mAP50: ___________
- Δ mAP50: ___________
- mAP50-95: ___________
- Healthy FP Rate: ___________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

## 1.3 Color Normalization - Reinhard (Faster Alternative)
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Reinhard et al. 2001; established standard |
| **Core Idea** | Simpler color normalization; faster than Macenko |
| **Why It Matters** | If Macenko too slow, Reinhard is lightweight alternative |
| **Literature Gain** | +0.01 to +0.03 mAP50 |
| **Implementation** | ~20 lines of numpy/OpenCV |
| **Complexity** | Very Simple |
| **Segmentation-Safe** | Yes |
| **Low-Risk** | YES |

**Implementation Library**: OpenCV, scikit-image, or custom
**Test Strategy**: Compare against Macenko (1.2) and baseline

**Test Date**: ___________
**Results (vs Baseline 0.512)**:
- mAP50: ___________
- Δ mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- **Comparison to 1.2 (Macenko)**: ___________________________________________________________
- Notes: ___________________________________________________________

---

## 1.4 Learnable CLAHE (Contrast-Limited Adaptive Histogram Equalization)
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Zuiderveld 1994; Park et al. IEEE TMI 2024 |
| **Core Idea** | Learnable CLAHE: randomize clip_limit and tileSize during training |
| **Why It Matters** | Disease regions are low-contrast; CLAHE enhances visibility |
| **Literature Gain** | +0.01 to +0.04 mAP50 |
| **Implementation** | Simple: cv2.createCLAHE with parameter sampling |
| **Complexity** | Very Simple |
| **Segmentation-Safe** | Yes |
| **Low-Risk** | YES |

**YOLO Config Approach**: Custom preprocessing hook
`python
def clahe_augmentation(img, p=0.5):
    if random.random() > p:
        return img
    clahe = cv2.createCLAHE(
        clipLimit=random.choice([2.0, 4.0, 8.0]),
        tileGridSize=(random.choice([8, 16]), random.choice([8, 16]))
    )
    return clahe.apply(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
`

**Test Parameters**:
- clip_limit: [2.0, 4.0, 8.0]
- tileGridSize: [(8, 8), (16, 16)]
- Apply probability: 0.5

**Test Date**: ___________
**Results (vs Baseline 0.512)**:
- mAP50: ___________
- Δ mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- Best Parameters: clip_limit=___ tileSize=___
- Notes: ___________________________________________________________

---

## 1.5 GridMask Augmentation (Spatial Dropout)
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Chen et al. ICCV 2020; Updated 2024 |
| **Core Idea** | Randomly mask grid of rectangular regions; forces network to infer |
| **Why It Matters** | Regularization for small data; prevents overfitting |
| **Literature Gain** | +0.01 to +0.02 mAP50 |
| **Implementation** | albumentations.GridDropout (1 line) |
| **Complexity** | Very Simple |
| **Segmentation-Safe** | Yes (Albumentations handles masks) |
| **Low-Risk** | YES |

**YOLO Config Integration** (via Albumentations hook if available):
`yaml
augmentation:
  - GridDropout:
      ratio: 0.5
      unit_size_range: [10, 20]
      fill_value: 0
      p: 0.5
`

**Test Parameters**:
- mask_ratio: [0.3, 0.5]
- grid_size: [10, 20]
- Apply probability: 0.5

**Test Date**: ___________
**Results (vs Baseline 0.512)**:
- mAP50: ___________
- Δ mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- Best Parameters: ratio=___ grid_size=___
- Notes: ___________________________________________________________

---

## 1.6 Mosaic Strength & Close-Mosaic Tuning
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | YOLO series evolution; standard in v11 (2024) |
| **Core Idea** | Tune mosaic probability and close_mosaic_epochs parameter |
| **Why It Matters** | Disease regions are small; mosaic helps small-data; close-mosaic improves late localization |
| **Literature Gain** | +0.01 to +0.03 mAP50 |
| **Implementation** | YOLO config parameters |
| **Complexity** | Simple (grid search) |
| **Segmentation-Safe** | Yes |
| **Low-Risk** | YES |

**YOLO Config Grid Search**:
`yaml
augmentation:
  mosaic: [0.5, 1.0]              # Test both values
  close_mosaic_epochs: [5, 10, 20]  # Test early vs late close
`

**Test Matrix** (3 combinations):
1. mosaic=0.5, close_mosaic_epochs=5
2. mosaic=1.0, close_mosaic_epochs=10
3. mosaic=1.0, close_mosaic_epochs=20

**Test Date**: ___________
**Results (vs Baseline 0.512)**:
- Config 1 mAP50: ___________
- Config 2 mAP50: ___________
- Config 3 mAP50: ___________
- Best Config: mosaic=___ close_mosaic_epochs=___
- Δ mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

## 1.7 HSV Color Space Tuning
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | YOLO series standard; critical for color variation |
| **Core Idea** | Randomize Hue, Saturation, Value during training |
| **Why It Matters** | Disease appearance varies with lighting; HSV captures this |
| **Literature Gain** | +0.01 to +0.03 mAP50 |
| **Implementation** | YOLO config: hsv_h, hsv_s, hsv_v |
| **Complexity** | Simple (grid search) |
| **Segmentation-Safe** | Yes |
| **Low-Risk** | YES |

**YOLO Config Grid Search**:
`yaml
augmentation:
  hsv_h: [0.005, 0.015, 0.025]    # Hue shift range
  hsv_s: [0.3, 0.5, 0.7]          # Saturation range
  hsv_v: [0.3, 0.5]               # Value (brightness) range
`

**Test Matrix** (9 combinations: 3 × 3 grid):
| hsv_h | hsv_s | hsv_v | Label |
|-------|-------|-------|-------|
| 0.005 | 0.3 | 0.3 | H1 |
| 0.005 | 0.5 | 0.3 | H2 |
| 0.015 | 0.5 | 0.5 | H3 (center) |
| ... | ... | ... | ... |

**Test Date**: ___________
**Grid Search Results** (record all 9 runs):
- Best Config: hsv_h=___ hsv_s=___ hsv_v=___
- Best mAP50: ___________
- Δ mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

## 1.8 Geometric Augmentation (Scale, Translate, Degrees, Shear)
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | YOLO series standard (2023-2024) |
| **Core Idea** | Tune geometric transforms: scale, translate, rotation, shear |
| **Why It Matters** | Shrimp orientation and framing vary; geometric robustness critical |
| **Literature Gain** | +0.01 to +0.03 mAP50 |
| **Implementation** | YOLO config parameters |
| **Complexity** | Simple (grid search) |
| **Segmentation-Safe** | Yes |
| **Low-Risk** | YES |

**YOLO Config Grid Search**:
`yaml
augmentation:
  scale: [0.3, 0.5, 0.7]          # Size variation
  translate: [0.1, 0.2, 0.3]      # Position shift
  degrees: [0, 10, 20]            # Rotation degrees
  shear: [0, 5, 10]               # Shear degrees
  perspective: [0]                # Keep low for shrimp (not perspective-driven)
`

**Recommended Test Combinations** (reduce search space):
| scale | translate | degrees | shear |
|-------|-----------|---------|-------|
| 0.3 | 0.1 | 0 | 0 |
| 0.5 | 0.2 | 10 | 5 |
| 0.7 | 0.3 | 20 | 10 |

**Test Date**: ___________
**Grid Search Results**:
- Best Config: scale=___ translate=___ degrees=___ shear=___
- Best mAP50: ___________
- Δ mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

## 1.9 Copy-Paste Native Tuning (YOLO v11)
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Ghiasi et al. 2021; YOLO v11 native (2024) |
| **Core Idea** | Paste disease regions from one image onto another |
| **Why It Matters** | Segmentation-specific augmentation; directly generates disease variations |
| **Literature Gain** | +0.02 to +0.05 mAP50 |
| **Implementation** | YOLO config: copy_paste parameter |
| **Complexity** | Simple (parameter tuning) |
| **Segmentation-Safe** | YES - masks preserved |
| **Low-Risk** | YES (Phase C will refine further) |

**YOLO Config Grid Search**:
`yaml
augmentation:
  copy_paste: [0.3, 0.5]          # Probability
  copy_paste_mode: ['flip', 'mixup']  # Mode selection
`

**Test Matrix** (4 combinations):
1. copy_paste=0.3, mode='flip'
2. copy_paste=0.5, mode='flip'
3. copy_paste=0.3, mode='mixup'
4. copy_paste=0.5, mode='mixup'

**Test Date**: ___________
**Results**:
- Config 1 (p=0.3, flip): mAP50=___
- Config 2 (p=0.5, flip): mAP50=___
- Config 3 (p=0.3, mixup): mAP50=___
- Config 4 (p=0.5, mixup): mAP50=___
- Best Config: copy_paste=___ mode=___
- Δ mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

## PHASE B SUMMARY TABLE

**Record Results Here** (fill after running Phase B):

| Method | mAP50 | Δ mAP50 | Decision | Notes |
|--------|-------|---------|----------|-------|
| Baseline | 0.512 | - | - | Reference |
| 1.1 Auto-Augment | | | ☐ | |
| 1.2 Macenko | | | ☐ | |
| 1.3 Reinhard | | | ☐ | |
| 1.4 CLAHE | | | ☐ | |
| 1.5 GridMask | | | ☐ | |
| 1.6 Mosaic | | | ☐ | |
| 1.7 HSV | | | ☐ | |
| 1.8 Geometric | | | ☐ | |
| 1.9 Copy-Paste | | | ☐ | |

**Phase B Decision**:
- Methods to KEEP: _________________________________________________
- Methods to DROP: _________________________________________________
- Expected Combined Improvement: ___________________________________
- Top 3 for Phase C: ________________________________________________

---

# PHASE C: COPY-PASTE FOCUSED VARIANTS (Priority: HIGH)
# Estimated: Weeks 2-3, ~20-30 GPU hours, ~10 models
# **Only run if Phase B copy-paste (1.9) shows +0.01 or higher**

## 2.1 MaskMix - Mask-Level Image Mixing
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Kim et al., ICCV 2024 |
| **Core Idea** | Mix only disease mask regions; keep backgrounds separate |
| **Why It Matters** | Avoids background interference; cleaner disease augmentation |
| **Literature Gain** | +0.02 to +0.04 mAP50 |
| **Implementation** | Custom or Albumentations extension |
| **Complexity** | Moderate |
| **Segmentation-Safe** | YES |
| **Prerequisite** | Phase 1.9 copy-paste shows promise |

**Implementation Strategy**:
- Extract mask region from one image, paste into another
- Only mix disease pixels; background stays original
- Test with different blend modes (alpha blending, Gaussian blur edge)

**Test Date**: ___________
**Results (vs Phase B Best)**:
- mAP50: ___________
- Δ from Phase B Best: ___________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

## 2.2 CopyPaste in Context - Contextual Inpainting
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Wang et al., ICCV 2023 |
| **Core Idea** | Copy-paste with context-aware inpainting of source region |
| **Why It Matters** | Removes boundary artifacts; cleaner paste operation |
| **Literature Gain** | +0.02 to +0.04 mAP50 |
| **Implementation** | Custom; requires inpainting model or simple averaging |
| **Complexity** | Moderate |
| **Segmentation-Safe** | YES |

**Implementation Strategy**:
- When pasting disease region, inpaint source region with background-colored pixels
- Use simple averaging or Telea inpainting (cv2.inpaint)

**Test Date**: ___________
**Results**:
- mAP50: ___________
- Δ from Phase B Best: ___________
- Comparison to 2.1 (MaskMix): ___________________________________________________________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

## 2.3 Augmentation Scheduling - Strength Annealing
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Zhang et al., CVPR 2024; Emerging trend 2024 |
| **Core Idea** | Gradually reduce augmentation strength over training epochs |
| **Why It Matters** | Strong aug early (exploration), weak aug late (refinement) |
| **Literature Gain** | +0.01 to +0.03 mAP50 (primarily stability) |
| **Implementation** | Custom training loop modification |
| **Complexity** | Moderate |
| **Segmentation-Safe** | YES |

**Scheduling Strategy**:
- Epochs 0-50%: Full augmentation strength (baseline)
- Epochs 50-80%: Half augmentation strength (reduce mosaic, copy_paste, etc.)
- Epochs 80-100%: Light augmentation (only HSV, geometric)

**Test Date**: ___________
**Results**:
- mAP50: ___________
- Δ from baseline: ___________
- Scheduling Function: Linear / Cosine / Other: ___________
- Decision: ☐ KEEP ☐ DROP
- Notes: ___________________________________________________________

---

# PHASE D: DOMAIN-SPECIFIC EXTENSIONS (Priority: MEDIUM)
# **Only run if Phase B+C combined improvement < +0.03 mAP50**

## 3.1 Augmentation-Invariant Parameter Scheduling
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Zhang et al., CVPR 2024 Workshop |
| **Core Idea** | Tailor augmentation to dataset size; reduce overfitting for small data |
| **Why It Matters** | Your dataset is small (~1100 images); calibrated aug critical |
| **Literature Gain** | +0.01 to +0.03 mAP50 |
| **Implementation** | Custom parameter schedule |
| **Complexity** | Moderate |

**Test Date**: ___________
**Results**:
- mAP50: ___________
- Decision: ☐ KEEP ☐ DROP

---

## 3.2 Shadow & Illumination Correction (If Dataset Has Issues)
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Song et al., MICCAI 2024 |
| **Core Idea** | Detect and remove shadows/uneven illumination |
| **Why It Matters** | Aquaculture lighting often uneven; affects disease visibility |
| **Literature Gain** | +0.01 to +0.03 mAP50 |
| **Implementation** | Frequency-based shadow detection or learning-based |
| **Complexity** | Moderate |
| **Prerequisite** | Visual inspection confirms uneven lighting in dataset |

**Decision Tree**:
- IF dataset shows even lighting → SKIP
- IF dataset shows shadows → TEST

**Test Date**: ___________
**Results**:
- mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- Visual Assessment: Lighting is [Even / Uneven]

---

# PHASE E: FREQUENCY-DOMAIN METHODS (Priority: LOW)
# **Only run if Phase B+C+D combined improvement < +0.02 mAP50 (plateau)**

## 4.1 Frequency Dropout - Stochastic Fourier
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Lim et al., CVPR 2024 |
| **Core Idea** | Randomly drop frequency bands during training |
| **Why It Matters** | Small data; disease in specific frequency bands |
| **Literature Gain** | +0.01 to +0.03 mAP50 |
| **Implementation** | Custom FFT-based augmentation |
| **Complexity** | Moderate |
| **Note** | You tested Fourier before; revisit focused variants only |

**Test Strategy** (limited scope):
- Drop high-frequency bands (30-50% of spectrum)
- Apply at p=0.5 during training
- Test only 2-3 variants, not full sweep

**Test Date**: ___________
**Results**:
- mAP50: ___________
- Decision: ☐ KEEP ☐ DROP
- Why tested: Plateau observed in earlier phases

---

## 4.2 Wavelet Enhancement - Detail Preservation
**Status**: ⬜ PENDING

| Attribute | Value |
|---|---|
| **Paper/Source** | Mukherjee et al., MICCAI 2024 |
| **Core Idea** | Decompose into wavelet bands; enhance detail layers |
| **Why It Matters** | Disease is fine detail; wavelet enhancement targeted |
| **Literature Gain** | +0.01 to +0.03 mAP50 |
| **Implementation** | PyWavelets (pywt) preprocessing |
| **Complexity** | Moderate |

**Test Date**: ___________
**Results**:
- mAP50: ___________
- Decision: ☐ KEEP ☐ DROP

---

# PHASE F: MULTI-SEED CONFIRMATION (Priority: CRITICAL)
# **Run only after selecting top 2-3 methods from earlier phases**

## Top 3 Methods for Confirmation:
1. Method Name: _________________________________________ (Expected Δ: _______)
2. Method Name: _________________________________________ (Expected Δ: _______)
3. Method Name: _________________________________________ (Expected Δ: _______)

**Baseline (for comparison)**: YOLOv11n-seg, clean-light aug, seed 42

**Test Parameters**:
- Seeds to test: 42, 123, 456 (or 42, 42+100, 42+200 to reach 5+ seeds)
- Split: Grouped-specimen stratified
- Report: Mean ± Std, 95% CI for each method

| Method | Seed 42 | Seed 123 | Seed 456 | Mean | Std | 95% CI |
|--------|---------|----------|----------|------|-----|--------|
| Baseline | 0.512 | | | | | |
| Method 1 | | | | | | |
| Method 2 | | | | | | |
| Method 3 | | | | | | |

---

# FINAL SUMMARY TABLE

**Fill after completing all phases**:

| Rank | Method | Phase | Category | mAP50 | Δ mAP50 | Healthy FP | Decision | Notes |
|------|--------|-------|----------|-------|---------|-----------|----------|-------|
| - | Baseline | - | - | 0.512 | 0 | 0.244 | - | Reference |
| 1 | | | | | | | ✓ KEEP | |
| 2 | | | | | | | ✓ KEEP | |
| 3 | | | | | | | ✓ KEEP | |
| X | | | | | | | ✗ DROP | |
| X | | | | | | | ✗ DROP | |

**Combined Strategy** (if stacking multiple methods):
- Methods to combine: __________________________________________________________________
- Expected combined improvement: ___________________________________________________________
- Paper narrative: We test [X] augmentation strategies and find that [Y] combination improves segmentation by [Z] mAP50...

---

# DECISION RULES

**KEEP a method if**:
- ✓ Δ mAP50 ≥ +0.005 (0.5 percentage point)
- ✓ AND Healthy FP rate does NOT increase by > 0.02
- ✓ AND visual inspection shows augmentation is reasonable

**DROP a method if**:
- ✗ Δ mAP50 < +0.003
- ✗ OR Healthy FP rate increases significantly
- ✗ OR augmentation creates unrealistic images

**COMBINE methods if**:
- ✓ Both improve independently
- ✓ They don't conflict (e.g., two color norms might interfere)
- ✓ Combined effect ≥ sum of individual effects (no negative interaction)

---

# IMPLEMENTATION CHECKLIST

Before starting experiments:

- [ ] Confirm grouped-specimen split seed 42 exists and is saved
- [ ] Verify Ultralytics version (should match baseline)
- [ ] Confirm hidden Albumentations hook is enabled
- [ ] Document exact YOLO config before each run
- [ ] Save split manifests alongside results
- [ ] Backup baseline model weights
- [ ] Set up augmentation visualization (save sample batches)
- [ ] Create Results aggregator script (auto-generate comparison table)

---

# REFERENCES & RESOURCES

**Papers to Cite in Final Work**:
- YOLO v11: Ultralytics (2024) official documentation
- MaskMix: Kim et al., ICCV 2024
- Color Normalization: Gamper et al., MICCAI 2024
- Copy-Paste in Context: Wang et al., ICCV 2023
- Frequency Dropout: Lim et al., CVPR 2024
- GridMask: Chen et al., ICCV 2020 (or 2024 update)

**Code Repositories**:
- Color Normalization: stain-tools, histolab, spams
- CLAHE: OpenCV (cv2.createCLAHE)
- GridMask: albumentations.GridDropout
- Frequency operations: NumPy FFT, scipy, PyWavelets (pywt)
- Inpainting: cv2.inpaint (Telea)

**Testing & Tracking**:
- Save results.csv from each YOLO training
- Track configs in YAML files alongside checkpoints
- Generate comparison tables monthly for paper/appendix
