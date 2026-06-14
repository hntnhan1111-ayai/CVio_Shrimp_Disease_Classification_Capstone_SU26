# Preprocessing & Augmentation Screening Checklist
# Shrimp Disease Segmentation (YOLOv11n-seg)
# Last Updated: 2026-06-14

## How to Use This Checklist
1. Each row is a method to test in order of priority
2. Implementation cost (Simple/Moderate/Hard) + Expected gain determines test priority
3. When you implement a method, mark Status = IN_PROGRESS, record results
4. Decision: KEEP (if +gain ≥ 0.005 mAP50) or DROP (marginal/negative)
5. Record exact delta vs baseline in Results column
6. Generate comparison table quarterly for paper/appendix

## Baseline Reference
- **Model**: YOLOv11n-seg
- **Split**: Grouped-specimen stratified
- **Baseline mAP50**: 0.512 (labeled-only)
- **Baseline mAP50-95**: 0.168
- **Baseline Healthy FP Rate**: 0.244
- **All deltas reported as**: Δ mAP50 (percentage point increase) or % improvement

---

## TIER 1: High Priority Quick Wins (Test in Phase B.1)

### 1.1 RandAugment Auto-Tuning (YOLO v11 Native)
| Attribute | Value |
|---|---|
| Method | auto_augment='randaugment' in YOLO config |
| Category | AutoML / YOLO Ecosystem |
| Paper/Source | YOLO v11 (Ultralytics 2024) |
| Expected Gain | +0.02 to +0.04 mAP50 |
| Implementation | Very Simple (1 config param) |
| Prerequisite | None (YOLO v11 has this built-in) |
| Segmentation-Specific | Yes (Ultralytics handles mask transforms) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | One-line config change; should test immediately |

### 1.2 Color Normalization - Macenko Stain Normalization
| Attribute | Value |
|---|---|
| Method | Macenko color normalization (medical standard) |
| Category | Medical Imaging / Preprocessing |
| Paper/Source | Macenko et al., 2009; 2023-2025 revived for biomedical |
| Expected Gain | +0.02 to +0.05 mAP50 |
| Implementation | Simple (library: spams, custom, or skimage-color) |
| Prerequisite | Install: python-spams or custom implementation |
| Segmentation-Specific | N/A (preprocessing, not augmentation) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Normalizes stain/color across images; critical for aquaculture lighting variation. Test on training + validation pipeline. |

### 1.3 Color Normalization - Reinhard Color Normalization
| Attribute | Value |
|---|---|
| Method | Reinhard color normalization (simpler alternative) |
| Category | Medical Imaging / Preprocessing |
| Paper/Source | Reinhard et al., 2001; still used in 2023-2025 |
| Expected Gain | +0.01 to +0.03 mAP50 |
| Implementation | Simple (library: cv2 or skimage) |
| Prerequisite | None (both work with OpenCV) |
| Segmentation-Specific | N/A (preprocessing) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Faster than Macenko; test both, pick best. Consider stacking: Macenko → Reinhard. |

### 1.4 Learnable CLAHE (Contrast-Limited Adaptive Histogram Equalization)
| Attribute | Value |
|---|---|
| Method | CLAHE with augmentation-time parameter randomization |
| Category | Medical Imaging / Preprocessing |
| Paper/Source | CLAHE (Zuiderveld, 1994); learnable variants (2023-2024) |
| Expected Gain | +0.01 to +0.04 mAP50 |
| Implementation | Simple (library: cv2.createCLAHE; add parameter sweep) |
| Prerequisite | None (OpenCV standard) |
| Segmentation-Specific | N/A (preprocessing) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Enhances low-contrast disease regions. Test: clip_limit ∈ [2.0, 4.0, 8.0], tile_size ∈ [8, 16]. Apply at 50% probability during training. |

### 1.5 GridMask Augmentation (Spatial Dropout)
| Attribute | Value |
|---|---|
| Method | GridMask: randomly mask grid of rectangular regions |
| Category | Small-Data Regularization / Augmentation |
| Paper/Source | Chen et al., ICCV 2020; still effective 2023-2025 |
| Expected Gain | +0.01 to +0.02 mAP50 |
| Implementation | Simple (library: albumentations.GridDropout) |
| Prerequisite | albumentations library |
| Segmentation-Specific | Yes (Albumentations handles masks) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Forces network to infer disease regions from context. Test mask_ratio ∈ [0.3, 0.5], grid_size ∈ [10, 20]. Apply p=0.5. |

### 1.6 Label Smoothing for Segmentation
| Attribute | Value |
|---|---|
| Method | Soft mask targets (smooth boundaries) instead of binary |
| Category | Regularization / Training Strategy |
| Paper/Source | Szegedy et al., 2016; still standard in 2023-2025 |
| Expected Gain | +0.01 to +0.02 mAP50 |
| Implementation | Simple (YOLO parameter: label_smoothing) |
| Prerequisite | None (YOLO v11 supports this) |
| Segmentation-Specific | Yes (Ultralytics applies to mask targets) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Prevents overconfident predictions. Test label_smoothing ∈ [0.0, 0.1, 0.2]. |

---

## TIER 2: Phase B & C Integration (Medium Priority)

### 2.1 Mosaic Strength Tuning & close_mosaic Epoch
| Attribute | Value |
|---|---|
| Method | Vary mosaic probability + close_mosaic_epoch parameter |
| Category | Small-Data Augmentation / YOLO Ecosystem |
| Paper/Source | YOLO series (Redmon→Ultralytics); 2023-2024 refinements |
| Expected Gain | +0.01 to +0.03 mAP50 |
| Implementation | Simple (YOLO config: mosaic, close_mosaic_epochs) |
| Prerequisite | None (YOLO native) |
| Segmentation-Specific | Yes |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Test: mosaic ∈ [0.5, 1.0], close_mosaic_epochs ∈ [5, 10, 20]. Already partially in Phase B. |

### 2.2 Copy-Paste Augmentation Boundary Refinement
| Attribute | Value |
|---|---|
| Method | Copy-paste with Gaussian blur boundary blending |
| Category | Segmentation-Specific Augmentation |
| Paper/Source | Ghiasi et al., 2021; boundary refinements ICCV 2024 |
| Expected Gain | +0.02 to +0.05 mAP50 |
| Implementation | Moderate (integrate copy-paste with boundary smoothing) |
| Prerequisite | Albumentations or custom implementation |
| Segmentation-Specific | Yes (mask-aware cropping) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Phase C planned. Test: blur_sigma ∈ [1.0, 2.0], same_image vs cross_image modes. |

### 2.3 MaskMix - Mask-Level Image Mixing
| Attribute | Value |
|---|---|
| Method | Mix images at mask level only; preserve backgrounds |
| Category | Segmentation-Specific Augmentation |
| Paper/Source | 2024 (CVPR variant of copy-paste) |
| Expected Gain | +0.02 to +0.04 mAP50 |
| Implementation | Moderate (requires mask-aware mixing logic) |
| Prerequisite | Custom implementation or Albumentations extension |
| Segmentation-Specific | Yes |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Variant of copy-paste; avoids background interference. Test in Phase C. Consider as copy-paste vs MaskMix comparison. |

### 2.4 HSV Color Space Augmentation Tuning
| Attribute | Value |
|---|---|
| Method | Randomize HSV (hue, saturation, value) at training time |
| Category | Photometric Augmentation / YOLO Native |
| Paper/Source | YOLO series; essential for 2023-2025 |
| Expected Gain | +0.01 to +0.03 mAP50 |
| Implementation | Simple (YOLO params: hsv_h, hsv_s, hsv_v) |
| Prerequisite | None (YOLO v11 native) |
| Segmentation-Specific | Yes |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Critical for color-varying disease appearance. Phase B: grid search hsv_h ∈ [0.005, 0.015, 0.025], hsv_s ∈ [0.3, 0.5, 0.7], hsv_v ∈ [0.3, 0.5]. |

### 2.5 Geometric Augmentation Policy (Scale, Translate, Degrees)
| Attribute | Value |
|---|---|
| Method | Tune geometric transforms: scale, translate, rotation, shear |
| Category | Geometric Augmentation / YOLO Native |
| Paper/Source | YOLO series; standard for 2023-2025 |
| Expected Gain | +0.01 to +0.03 mAP50 |
| Implementation | Simple (YOLO params: scale, translate, degrees, shear, perspective) |
| Prerequisite | None (YOLO v11 native) |
| Segmentation-Specific | Yes |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Phase B: grid search scale ∈ [0.3, 0.5, 0.7], translate ∈ [0.1, 0.2, 0.3], degrees ∈ [0, 10, 20]. |

---

## TIER 3: Domain-Specific Extensions (Test after Tier 1+2)

### 3.1 Augmentation Scheduling - Strength Annealing
| Attribute | Value |
|---|---|
| Method | Gradually reduce augmentation strength over training epochs |
| Category | Training Strategy / Augmentation Scheduling |
| Paper/Source | 2023-2024 medical imaging; emerging in YOLO community |
| Expected Gain | +0.01 to +0.03 mAP50 (primarily stability) |
| Implementation | Moderate (custom training loop or YOLO config modification) |
| Prerequisite | Ability to modify YOLO training script |
| Segmentation-Specific | Yes |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Schedule: full augmentation epochs 0-50%, half augmentation 50-80%, light augmentation 80-100%. Test scheduling function (linear, cosine). |

### 3.2 Stochastic Frequency Dropout (Fourier-Based)
| Attribute | Value |
|---|---|
| Method | Randomly drop frequency bands during training (Fourier space) |
| Category | Frequency Methods / Advanced Augmentation |
| Paper/Source | 2023-2024 (stochastic variants) |
| Expected Gain | +0.01 to +0.03 mAP50 |
| Implementation | Moderate (FFT operations + custom augmentation) |
| Prerequisite | NumPy, scipy (FFT), albumentations-style integration |
| Segmentation-Specific | Yes (apply only on images, not masks) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | You tested Fourier before (mixed results). Revisit as focused variant: drop high-frequency bands (30-50%) at p=0.5. Phase E. |

### 3.3 Wavelet Preprocessing - Detail Enhancement
| Attribute | Value |
|---|---|
| Method | Decompose image into wavelet bands; enhance detail layers |
| Category | Frequency Methods / Preprocessing |
| Paper/Source | 2024 (medical imaging) |
| Expected Gain | +0.01 to +0.02 mAP50 |
| Implementation | Moderate (pywt library; custom preprocessing) |
| Prerequisite | pywt (PyWavelets) |
| Segmentation-Specific | N/A (preprocessing) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Test wavelet family (db2, db4, sym3); enhance detail coefficients (multiply 1.2-1.5). Phase E. |

### 3.4 Cutout / CutMask Augmentation
| Attribute | Value |
|---|---|
| Method | Randomly mask non-disease regions; keep disease visible |
| Category | Small-Data Regularization |
| Paper/Source | Devries & Taylor, 2017; segmentation variants 2023-2024 |
| Expected Gain | +0.01 to +0.02 mAP50 |
| Implementation | Simple (custom or albumentations.CoarseDropout) |
| Prerequisite | albumentations |
| Segmentation-Specific | Partial (standard Cutout not mask-aware; needs custom logic) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Test mask_ratio ∈ [0.1, 0.3], max_holes ∈ [8, 16]. Phase B or C. |

### 3.5 Denoising Preprocessing (Non-Local Means)
| Attribute | Value |
|---|---|
| Method | Remove noise using NLM (Non-Local Means) |
| Category | Medical Imaging / Preprocessing |
| Paper/Source | Classical (Buades et al., 2005); still used 2023-2025 |
| Expected Gain | +0 to +0.02 mAP50 (if noisy dataset) |
| Implementation | Simple (cv2.fastNlMeansDenoisingColored) |
| Prerequisite | None (OpenCV standard) |
| Segmentation-Specific | N/A (preprocessing) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Only test if dataset is noisy (check image SNR first). Params: h=10, templateWindowSize=7, searchWindowSize=21. Low priority. |

### 3.6 Shadow & Illumination Correction
| Attribute | Value |
|---|---|
| Method | Detect and remove shadows/gradients before training |
| Category | Medical Imaging / Preprocessing |
| Paper/Source | 2023-2024 (automatic shadow detection) |
| Expected Gain | +0.01 to +0.03 mAP50 |
| Implementation | Moderate (shadow detection + removal) |
| Prerequisite | Custom algorithm or library |
| Segmentation-Specific | N/A (preprocessing) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Aquaculture lighting is uneven. Test: simple illumination estimation (divide by smoothed image), or frequency-based shadow removal. Low priority. |

---

## TIER 4: Plateau-Breaking Methods (If Needed)

### 4.1 Fourier Augmentation Revisit (Constrained Variants)
| Attribute | Value |
|---|---|
| Method | Fourier enhancement focused on specific frequency bands |
| Category | Frequency Methods |
| Paper/Source | Prior work (your notebooks); 2023-2024 refinements |
| Expected Gain | +0.01 to +0.03 mAP50 (if well-tuned) |
| Implementation | Moderate (FFT, band selection) |
| Prerequisite | NumPy, scipy |
| Segmentation-Specific | Partial (apply on images) |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | You tested broad Fourier; only revisit focused variants (e.g., enhance mid-frequency band 5-20Hz). Phase E, test only if plateau. |

### 4.2 RandAugment Fast Policy Search (Custom Implementation)
| Attribute | Value |
|---|---|
| Method | Automated policy search via random/grid search |
| Category | AutoML / Policy Search |
| Paper/Source | 2023-2024 (fast variants: PopAug, OPS) |
| Expected Gain | +0.02 to +0.06 mAP50 |
| Implementation | Hard (requires search framework + multiple training runs) |
| Prerequisite | Custom training harness |
| Segmentation-Specific | Yes |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | High effort. Only implement if Phase B grid search plateaus and you want systematic policy optimization. |

### 4.3 Boundary-Aware Augmentation (Advanced)
| Attribute | Value |
|---|---|
| Method | Augmentations avoid disrupting object boundaries |
| Category | Segmentation-Specific (Advanced) |
| Paper/Source | CVPR 2024 (emerging) |
| Expected Gain | +0.01 to +0.03 mAP50 |
| Implementation | Hard (requires boundary detection + mask-aware transforms) |
| Prerequisite | Custom augmentation pipeline |
| Segmentation-Specific | Yes |
| **Status** | ❌ SKIP (low priority, complex) |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Complex to implement; marginal gain. Skip unless other methods plateau significantly. |

### 4.4 Healthy-Aware Loss Weighting
| Attribute | Value |
|---|---|
| Method | Penalize false positives on healthy specimens more in loss |
| Category | Loss Function / Training Strategy |
| Paper/Source | Domain-specific refinement, 2024 |
| Expected Gain | +0.01 to +0.02 mAP50 |
| Implementation | Moderate (custom loss weighting) |
| Prerequisite | Ability to modify YOLO loss function |
| Segmentation-Specific | Yes |
| **Status** | ⬜ PENDING |
| **Test Date** | |
| **Results (Δ mAP50)** | |
| **Decision** | |
| **Notes** | Your baseline uses healthy-aware score for validation. Loss-level implementation is refinement. Test after Tier 1-3. |

---

## SUMMARY: Testing Order & Estimated Timeline

### Phase B.1 (Weeks 1-2): YOLO Native + Quick Wins
- [ ] 1.1 RandAugment Auto-Tuning
- [ ] 1.2 Macenko Color Norm
- [ ] 1.3 Reinhard Color Norm
- [ ] 1.4 CLAHE
- [ ] 1.5 GridMask
- [ ] 1.6 Label Smoothing
- [ ] 2.1 Mosaic Tuning
- [ ] 2.4 HSV Tuning
- [ ] 2.5 Geometric Tuning

**Estimated Runs**: ~20 models (YOLO grid search)
**Expected Time**: 40-60 GPU hours
**Decision Point**: Compare all 9 methods. Keep top 3-4 for Phase B.2

### Phase B.2 (Weeks 2-3): Extended Policies
- [ ] 2.2 Copy-Paste Boundary Refinement
- [ ] 2.3 MaskMix
- [ ] 3.4 Cutout / CutMask
- [ ] 3.1 Augmentation Scheduling

**Estimated Runs**: ~10 additional models
**Expected Time**: 20-30 GPU hours
**Decision Point**: Pick best 2 methods → Phase F (multi-seed confirmation)

### Phase C (Weeks 3-4): Copy-Paste Focused Sweep
- [ ] 2.2 Extended (different blur sigmas, modes)

**Estimated Runs**: ~8 models
**Expected Time**: 16-24 GPU hours

### Phase D (Weeks 4-5): Photometric Extensions (if needed)
- [ ] 3.2 Stochastic Frequency Dropout
- [ ] 3.3 Wavelet
- [ ] 3.5 Denoising (if noisy)
- [ ] 3.6 Shadow Correction (if uneven lighting)

**Decision Point**: Test only if Tier 1-2 don't reach target (+0.03 gain).

### Phase E (Weeks 5-6): Fourier Revisit (if plateau)
- [ ] 4.1 Constrained Fourier
- [ ] 4.2 Policy Search (if needed)

**Decision Point**: Only if other methods stall; high effort for marginal gain.

### Phase F (Weeks 6-7): Multi-Seed Confirmation
- [ ] Top 2 methods from B.2 + baseline
- [ ] Run across 3-5 grouped-split seeds
- [ ] Report mean, std, 95% CI

---

## Metrics Template (Copy & Fill In)

For each test, record:

`
### [Method Name]
**Date Tested**: 2026-XX-XX
**Notebook/Config**: [filename]
**Parameters**: [exact YOLO config or preprocessing params]
**Seed**: 42 (unless Phase F)
**Results**:
- Baseline mAP50: 0.512
- Test mAP50: [value]
- Δ mAP50: [value - 0.512]
- Baseline mAP50-95: 0.168
- Test mAP50-95: [value]
- Baseline Healthy FP: 0.244
- Test Healthy FP: [value]
**Decision**: KEEP / DROP
**Reason**: [why you kept or dropped]
**Visual Inspection**: [any observations from augmented images]
`

---

## Decision Rules

**KEEP a method if**:
- Δ mAP50 ≥ +0.005 (0.5 percentage point improvement)
- AND Healthy FP rate does not increase by >0.02
- AND visual inspection shows augmentation is reasonable (no distortion)

**DROP a method if**:
- Δ mAP50 < +0.003
- OR Healthy FP rate increases significantly
- OR augmentation is visually unrealistic

**COMBINE methods if**:
- Both improve independently
- They don't conflict (e.g., two color norms might interfere)
- Test combined version

---

## Appendix: Implementations to Prepare

Before starting, prep these code pieces:

1. **Color Normalization Module** (Macenko + Reinhard)
   - Input: RGB image
   - Output: normalized RGB
   - Config: reference image or per-image normalization

2. **CLAHE Module** (Learnable)
   - Input: RGB image
   - Output: enhanced image
   - Config: clip_limit, tile_size (randomized during training)

3. **Augmentation Visualization** (View augmented batches)
   - Save samples of augmented training images
   - Compare before/after for each method

4. **Results Aggregator** (CSV summary table)
   - Auto-generate comparison table
   - Track which methods to combine

---

## References for Implementation

- Macenko color norm: https://github.com/mitkovetta/stain-normalization (Python)
- CLAHE: OpenCV docs (cv2.createCLAHE)
- GridMask: albumentations.GridDropout
- Copy-Paste: albumentations or Ultralytics built-in
- Fourier: NumPy FFT, scipy
- Wavelet: PyWavelets (pywt)

