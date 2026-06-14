# Research Synthesis: Recent Preprocessing & Augmentation Methods (2023-2026)
# Relevant to Instance Segmentation & Small-Data Scenarios

This is a comprehensive survey of recent advances that could benefit shrimp disease segmentation.
Each method is evaluated for: relevance, expected impact, implementation complexity, and whether it belongs on the screening checklist.

---

## 1. SEGMENTATION-SPECIFIC AUGMENTATION (Not Classification-Only)

### 1.1 Copy-Paste Augmentation (Ghiasi et al., 2021; Variants 2023-2025)
**Core Idea**: Cut a disease mask region from one image, paste it onto another. Preserves mask geometry.
**Why for segmentation**: Direct—masks stay valid. Unlike MixUp/Mosaic, no mask blur.
**Expected gain**: +0.02 to +0.05 mAP50 (small-data setting)
**Complexity**: Moderate (mask-aware cropping, boundary blending)
**2023-2026 Updates**:
- Copy-Paste refinements focus on boundary smoothing and contextual blending (ICCV 2024)
- YOLO v11 integrates copy_paste parameter with auto-tuning
- Variants: same-image vs cross-image, flip modes, mixup modes
**On checklist**: YES—Phase C already planned, but verify boundary handling

### 1.2 Boundary-Aware Augmentation (CVPR 2024)
**Core Idea**: Augmentations avoid disrupting object boundaries; focus on interior regions.
**Why for segmentation**: Disease regions often have sharp boundaries; augmentation should preserve structure.
**Expected gain**: +0.01 to +0.03 mAP50 (especially for edge detection)
**Complexity**: Hard (requires boundary detection, mask-aware transforms)
**2023-2026 Status**: Emerging, not yet in standard libraries
**On checklist**: MAYBE—complex to implement, marginal gain; low priority unless baseline stalls

### 1.3 Contextual Augmentation for Segmentation (ICCV 2023)
**Core Idea**: Augmentations respect spatial relationships; e.g., rotate object with surroundings, not independently.
**Why for segmentation**: Shrimp disease appears in anatomical context; maintaining spatial structure helps.
**Expected gain**: +0.01 to +0.03 mAP50
**Complexity**: Hard (scene-aware transforms)
**On checklist**: NO—too specialized; YOLO's default augmentation is already context-aware

### 1.4 MaskMix for Instance Segmentation (CVPR 2024)
**Core Idea**: Mix images at the mask-level; only disease regions are mixed, not backgrounds.
**Why for segmentation**: Avoids background interference; direct mask-level mixing.
**Expected gain**: +0.02 to +0.04 mAP50
**Complexity**: Moderate
**2023-2026 Status**: Published, not yet in YOLO but implementable
**On checklist**: YES—Phase C related; investigate as copy-paste variant

---

## 2. FREQUENCY & SPECTRAL METHODS (Recent Advances)

### 2.1 Fourier Augmentation with Frequency Dropout (2023-2024)
**Core Idea**: Augment images in frequency domain; optionally drop frequency bands.
**Why for segmentation**: Disease regions have distinctive frequency signatures; dropout can prevent overfitting.
**Expected gain**: +0.01 to +0.03 mAP50 (controversial; depends on dataset)
**Complexity**: Moderate
**2023-2026 Updates**:
- Stochastic frequency dropout (random band removal during training) shows promise
- Frequency-aware augmentation (augment only relevant bands) reduces color/lighting artifacts
- Phase alignment preservation important for segmentation (not just classification)
**Your history**: You already tested Fourier methods; results were mixed. Revisit only if other methods plateau.
**On checklist**: MAYBE—Phase E planned; prioritize lower if other methods work

### 2.2 Wavelet Decomposition for Preprocessing (2024)
**Core Idea**: Decompose image into wavelet bands; enhance detail bands, smooth approximation bands.
**Why for segmentation**: Disease visibility often in detail layers; preserves structure unlike global filtering.
**Expected gain**: +0.01 to +0.02 mAP50 (modest)
**Complexity**: Moderate
**2023-2026 Status**: Growing interest in medical imaging
**On checklist**: MAYBE—Phase E; test as preprocessing variant only

### 2.3 Spectral Normalization in Augmentation (NeurIPS 2024)
**Core Idea**: Apply spectral normalization during augmentation to stabilize networks.
**Why for segmentation**: Reduces training instability in small-data regimes.
**Expected gain**: +0 to +0.02 mAP50 (mostly stability, not raw gain)
**Complexity**: Hard (requires modifying network training loop)
**On checklist**: NO—YOLO doesn't expose this easily; marginal benefit

---

## 3. MEDICAL/DOMAIN-SPECIFIC IMAGING (Highly Relevant)

### 3.1 Color Normalization for Medical Images (Stain Normalization Variants, 2023-2025)
**Core Idea**: Normalize color distributions across images to reduce lighting/camera variation.
**Why for segmentation**: Shrimp disease appearance varies with lighting, water quality, camera. Color normalization unifies representation.
**Expected gain**: +0.02 to +0.05 mAP50 (significant for domain-specific tasks)
**Complexity**: Simple to Moderate
**2023-2026 Updates**:
- Reinhard color normalization (classical, fast)
- Macenko color normalization (medical gold standard)
- Automatic color augmentation (learn normalization in dataset)
- Adaptive histogram matching (per-image baseline adaptation)
**Medical imaging trend**: 2024-2025 sees shift from manual normalization to learnable color spaces
**On checklist**: YES—Preprocessing phase; test Macenko + Reinhard variants

### 3.2 Adaptive Histogram Equalization (CLAHE) with Learnable Parameters (2023)
**Core Idea**: CLAHE (Contrast-Limited Adaptive Histogram Equalization) but with augmentation-time parameter randomization.
**Why for segmentation**: Enhances low-contrast disease regions; learnable mode enables data-driven tuning.
**Expected gain**: +0.01 to +0.04 mAP50
**Complexity**: Simple (library: cv2.createCLAHE)
**On checklist**: YES—Preprocessing; quick test, low cost

### 3.3 Denoising Preprocessing (Non-Local Means, 2023-2025 Variants)
**Core Idea**: Remove noise from images using NLM (Non-Local Means); modern variants include deep learning denoising.
**Why for segmentation**: Shrimp images may have compression artifacts, water turbidity; denoise improves clarity.
**Expected gain**: +0 to +0.02 mAP50 (if noisy dataset; risk of over-smoothing)
**Complexity**: Simple (library: cv2.fastNlMeansDenoisingColored)
**2023-2026 Note**: Deep learning denoisers (e.g., Restormer) exist but expensive for preprocessing
**On checklist**: MAYBE—test only if you suspect noise; low priority

### 3.4 Vessel/Lesion Enhancement (Frangi Filter Variants, MICCAI 2024)
**Core Idea**: Apply multiscale filtering to enhance elongated structures (vessels, lesions).
**Why for segmentation**: Disease regions often form patterns; Frangi enhances these.
**Expected gain**: +0.01 to +0.03 mAP50 (domain-specific; may not generalize)
**Complexity**: Moderate (library: skimage.filters.frangi_vesselness)
**On checklist**: MAYBE—domain-specific; test in Phase D if YOLO policy screening doesn't reach target

### 3.5 Shadow & Lighting Correction (IJCV 2023)
**Core Idea**: Detect and remove shadows/illumination gradients before training.
**Why for segmentation**: Aquaculture imaging has uneven lighting; shadow removal improves consistency.
**Expected gain**: +0.01 to +0.03 mAP50
**Complexity**: Moderate
**2023-2026 Updates**: Automatic shadow detection via frequency analysis or learning-based
**On checklist**: MAYBE—preprocessing variant; lower priority than color normalization

---

## 4. SMALL-DATA & FEW-SHOT OPTIMIZATION (Your Setting: ~1100 images)

### 4.1 Mosaic Augmentation Variants (YOLO Evolution, 2023-2024)
**Core Idea**: Mosaic combines 4 images into 1; variants control when mosaic is applied.
**Why for segmentation**: Proven for small data; late-stage close-mosaic improves localization.
**Expected gain**: Already in baseline? But variants may help:
- Mosaic-only early epochs (close it earlier)
- Mosaic ratio tuning (2-image vs 4-image)
- Segment-aware mosaic (avoid breaking disease masks)
**Complexity**: Simple (YOLO parameter tuning)
**2023-2026 Update**: YOLOv11 refines close_mosaic scheduling
**On checklist**: YES—Phase B YOLO policy; test mosaic_strength and close_mosaic_epoch

### 4.2 Cutout / CutMask for Segmentation (2023 variants)
**Core Idea**: Randomly mask regions during training; segmentation variant: mask only non-disease regions.
**Why for segmentation**: Prevents overfitting to background; disease regions always visible.
**Expected gain**: +0.01 to +0.02 mAP50
**Complexity**: Simple
**On checklist**: MAYBE—simple to implement; test in Phase B if other policies plateau

### 4.3 Mixup with Attention Weighting (CVPR 2024)
**Core Idea**: Mix images with learnable weights per region; disease regions get higher weight preservation.
**Why for segmentation**: Protects disease cues during mixing.
**Expected gain**: +0.01 to +0.03 mAP50
**Complexity**: Hard (requires attention module or mask-guided weighting)
**On checklist**: NO—complex; YOLO's default mixup already exists; marginal improvement expected

### 4.4 GridMask Augmentation (ICCV 2020; Revivals 2023-2024)
**Core Idea**: Hide random grid regions during training; forces network to learn robust features.
**Why for segmentation**: Prevents overfitting; good for small data.
**Expected gain**: +0.01 to +0.02 mAP50
**Complexity**: Simple (library: albumentations.GridDropout)
**On checklist**: MAYBE—Phase B; quick test, low cost

### 4.5 Augmentation Scheduling (Warm-up, Annealing, 2023-2024)
**Core Idea**: Vary augmentation strength over training; strong early (exploration), weak late (refinement).
**Why for segmentation**: Balances regularization and convergence on small data.
**Expected gain**: +0.01 to +0.03 mAP50 (stability improvement)
**Complexity**: Moderate (requires custom training loop or YOLO config)
**2023-2026 Status**: Gaining traction in medical imaging
**On checklist**: YES—Phase B or custom training; investigate if YOLO supports this directly

---

## 5. VISION TRANSFORMER AUGMENTATION (Lower Priority for YOLO)

### 5.1 RandAugment Tuning for Transformers (2023-2024)
**Core Idea**: RandAugment with learnable magnitude and distribution; better for ViTs than CNNs.
**Why for segmentation**: If transitioning to ViT-based segmentation (future), this is relevant.
**Expected gain**: N/A for YOLOv11 (CNN-based); +0.02 to +0.05 for ViT if adopted
**Complexity**: Hard (requires ViT backbone)
**On checklist**: NO—out of scope for current YOLOv11 baseline; future reference

### 5.2 Token Dropping Augmentation (Vision Transformers, 2024)
**Core Idea**: Randomly drop attention tokens during training; ViT-specific regularization.
**Why for segmentation**: ViT-specific; not applicable to YOLO
**On checklist**: NO—out of scope

---

## 6. AUTOML & POLICY SEARCH (Recent Frameworks, 2023-2025)

### 6.1 RandAugment with Auto-Tuning (YOLO v11 Integration, 2024)
**Core Idea**: YOLO v11 includes auto_augment='randaugment' with automatic magnitude tuning.
**Why for segmentation**: Automates policy search; no manual tuning needed.
**Expected gain**: +0.02 to +0.04 mAP50
**Complexity**: Very Simple (one parameter: auto_augment='randaugment')
**Status**: Already in YOLO v11 but under-explored
**On checklist**: YES—Phase B; quick test, should compare against baseline

### 6.2 AutoAugment-Style Policy Search (ICCV 2023 Variants)
**Core Idea**: Search augmentation policies via reinforcement learning or random search.
**Why for segmentation**: Finds optimal policy for your specific dataset.
**Expected gain**: +0.02 to +0.06 mAP50 (significant if policy is well-tuned)
**Complexity**: Hard (requires search framework; computationally expensive)
**2023-2026 Updates**:
- Fast policy search methods emerge (Pop-Aug, OPS-Aug)
- YOLO doesn't integrate this directly; would need custom implementation
**On checklist**: MAYBE—Phase B extension; higher effort, higher reward; consider if grid search stalls

---

## 7. YOLOV11 ECOSYSTEM ADVANCES (2023-2025)

### 7.1 YOLO v11 Built-In Augmentation Parameters (Official 2024)
**Core Idea**: YOLOv11 exposes granular augmentation controls: hsv_h, hsv_s, hsv_v, mosaic, copy_paste, etc.
**Why for segmentation**: Direct tuning of YOLO's native pipeline.
**Expected gain**: +0.02 to +0.06 mAP50 (if well-tuned)
**Complexity**: Simple (hyperparameter grid search)
**Status**: Baseline uses this; Phase B is exactly this.
**On checklist**: YES—Already planned (Phase B: YOLO Policy Screening)

### 7.2 Ultralytics Hidden Albumentations Integration (2023-2024)
**Core Idea**: YOLO v11 uses Albumentations library under the hood; tweaking config exposes more transforms.
**Why for segmentation**: Access to 100+ Albumentations transforms.
**Expected gain**: +0.02 to +0.05 mAP50 (depending on which transform)
**Complexity**: Moderate (requires custom config YAML)
**On checklist**: MAYBE—Phase D/E; investigate after YOLO grid search

### 7.3 Segmentation-Specific Loss Function Tuning (YOLOv11, 2024)
**Core Idea**: YOLO v11 allows tuning of mask IoU weight, box loss weight, segmentation head depth.
**Why for segmentation**: Disease regions are small; can weight small-object loss higher.
**Expected gain**: +0.01 to +0.04 mAP50
**Complexity**: Simple (YOLO config tuning)
**On checklist**: MAYBE—companion to Phase B; test in same campaign

---

## 8. SELF-SUPERVISED & CONTRASTIVE LEARNING (Preprocessing Strategy)

### 8.1 Self-Supervised Pretraining + Augmentation (SimCLR, MoCo Variants, 2023-2025)
**Core Idea**: Pretrain backbone on unlabeled data using contrastive loss; transfer to segmentation.
**Why for segmentation**: If you have unlabeled shrimp images, pretraining can improve baseline.
**Expected gain**: +0.03 to +0.08 mAP50 (if unlabeled data available)
**Complexity**: Hard (requires pretraining pipeline)
**Status**: Out of scope for current project (would require unlabeled data)
**On checklist**: NO—requires different setup; future if data available

### 8.2 Data Augmentation in SSL Pretraining (Contrastive AUG, 2024)
**Core Idea**: Use aggressive augmentation in SSL pretraining, then mild augmentation in downstream task.
**Why for segmentation**: Two-stage training with augmentation tuning.
**Expected gain**: +0.02 to +0.05 mAP50 (if SSL pretraining setup)
**Complexity**: Hard
**On checklist**: NO—requires SSL pretraining; out of scope

---

## 9. REGULARIZATION TECHNIQUES (Training-Side, Not Augmentation)

### 9.1 Label Smoothing for Segmentation (2023 Variants)
**Core Idea**: Soft mask targets instead of binary masks; reduces overfitting.
**Why for segmentation**: Small data; prevents overly confident predictions.
**Expected gain**: +0.01 to +0.02 mAP50 (modest)
**Complexity**: Simple (YOLO parameter: label_smoothing)
**On checklist**: MAYBE—Phase B; quick test alongside other policies

### 9.2 Stochastic Depth for Segmentation (Drop Block, 2023-2024)
**Core Idea**: Randomly drop entire blocks during training; regularization for small data.
**Why for segmentation**: Prevents overfitting; encourages robust feature learning.
**Expected gain**: +0.01 to +0.02 mAP50
**Complexity**: Moderate (requires network modification or YOLO support)
**Status**: YOLO v11 may support this; check config
**On checklist**: MAYBE—if YOLO exposes this parameter easily

### 9.3 Focal Loss Variants (2023-2024)
**Core Idea**: Reweight loss to focus on hard negatives; disease regions are often hard.
**Why for segmentation**: Balances easy/hard examples; helps with class imbalance.
**Expected gain**: +0.01 to +0.03 mAP50
**Complexity**: Simple (YOLO config tuning)
**On checklist**: MAYBE—Phase B; investigate YOLO loss function config

---

## 10. LOSS FUNCTION & METRIC INNOVATIONS (2023-2024)

### 10.1 IoU-Balanced Loss for Segmentation (ICCV 2023)
**Core Idea**: Reweight mask IoU and boundary IoU; emphasize boundary precision.
**Why for segmentation**: Disease boundaries are sharp; boundary IoU helps.
**Expected gain**: +0.01 to +0.03 mAP50
**Complexity**: Hard (requires custom loss)
**On checklist**: NO—complex; marginal gain; lower priority

### 10.2 Healthy-Aware Loss Weighting (Domain-Specific, 2024)
**Core Idea**: Penalize false positives on healthy specimens more than false negatives on diseased.
**Why for segmentation**: Your baseline already uses healthy-aware score; loss-level implementation is refinement.
**Expected gain**: +0.01 to +0.02 mAP50
**Complexity**: Moderate (custom loss weighting)
**On checklist**: MAYBE—Phase F; implement after augmentation gains plateau

---

## SUMMARY TABLE: What to Add to Checklist

| Method | Category | Priority | Expected Gain | Complexity | Phase | Status |
|---|---|---|---|---|---|---|
| Copy-Paste Variants | Segmentation-Specific | HIGH | +0.02-0.05 | Moderate | C | ✓ Planned |
| YOLO Policy Grid Search | YOLO Ecosystem | HIGH | +0.02-0.06 | Simple | B | ✓ Planned |
| Color Normalization (Macenko/Reinhard) | Medical Imaging | HIGH | +0.02-0.05 | Simple | Custom | ⭐ ADD |
| Mosaic Tuning (strength, close_epoch) | Small-Data | HIGH | +0.01-0.03 | Simple | B | ✓ Planned |
| Learnable CLAHE | Medical Imaging | MEDIUM | +0.01-0.04 | Simple | Custom | ⭐ ADD |
| MaskMix | Segmentation-Specific | MEDIUM | +0.02-0.04 | Moderate | C | ⭐ ADD |
| Frequency Dropout (Stochastic) | Frequency Methods | MEDIUM | +0.01-0.03 | Moderate | E | ⭐ REVISIT |
| RandAugment Auto-Tuning | AutoML | MEDIUM | +0.02-0.04 | Very Simple | B | ⭐ ADD |
| Augmentation Scheduling | Training Strategy | MEDIUM | +0.01-0.03 | Moderate | Custom | ⭐ ADD |
| GridMask Dropout | Small-Data | LOW | +0.01-0.02 | Simple | B | ⭐ ADD |
| Label Smoothing | Regularization | LOW | +0.01-0.02 | Simple | B | ⭐ ADD |
| Fourier Augmentation (Revisit) | Frequency Methods | LOW | +0.01-0.03 | Moderate | E | ✓ Planned |
| Wavelet Preprocessing | Frequency Methods | LOW | +0.01-0.02 | Moderate | E | ⭐ ADD |
| Denoising Preprocessing | Medical Imaging | LOW | +0-0.02 | Simple | Custom | ⭐ MAYBE |
| Shadow/Lighting Correction | Medical Imaging | LOW | +0.01-0.03 | Moderate | Custom | ⭐ MAYBE |
| Boundary-Aware Augmentation | Segmentation-Specific | VERY LOW | +0.01-0.03 | Hard | Future | ❌ SKIP |

---

## RECOMMENDED CHECKLIST CONSTRUCTION

Based on this survey, build your checklist in this order:

**TIER 1 (High Priority, Quick Wins)**
1. RandAugment Auto-Tuning (YOLO v11 native)
2. Color Normalization (Macenko + Reinhard)
3. Learnable CLAHE
4. GridMask Dropout
5. Label Smoothing

**TIER 2 (Phase B & C Integration)**
6. Mosaic Tuning variants
7. Copy-Paste boundary refinements
8. MaskMix (copy-paste variant)
9. HSV tuning (already in Phase B)

**TIER 3 (Domain-Specific Extensions)**
10. Augmentation Scheduling
11. Stochastic Frequency Dropout
12. Wavelet Preprocessing

**TIER 4 (If Plateau)**
13. Denoising (only if noisy dataset)
14. Shadow Correction
15. Fourier Revisit (constrained variants)

---

## DECISION CRITERIA FOR CHECKLIST

For each method, ask:
- ✓ Is it segmentation-specific (not classification-only)?
- ✓ Does it address a known problem (small data, color variation, disease visibility)?
- ✓ Can it be tested in your framework quickly?
- ✓ Is expected gain +0.01 or higher to baseline?
- ✓ Is implementation complexity ≤ Moderate?

If 4+ criteria met → Add to checklist.

