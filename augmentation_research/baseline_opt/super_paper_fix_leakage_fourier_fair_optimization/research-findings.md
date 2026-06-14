# Research Findings: EWU vs Hand-Labeled Shrimp Segmentation Dataset

This file records evidence derived from the local dataset EDA and visual comparison reports.

Generated evidence folder:

[reports/dataset_style_comparison](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison)

Main figures:

- [EWU within-specimen label style](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/ewu_within_specimen_label_style_examples.png)
- [Hand-labeled within-specimen label style](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/hand_within_specimen_label_style_examples.png)
- [EWU vs hand-labeled visually similar image pairs](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/ewu_vs_hand_visual_hash_similar_pairs.png)

Source tables:

- [dataset_summary.csv](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/dataset_summary.csv)
- [class_and_coinfection_counts.csv](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/class_and_coinfection_counts.csv)
- [specimen_group_summary.csv](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/specimen_group_summary.csv)
- [visual-hash similar pairs CSV](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/ewu_hand_visual_hash_similar_pairs.csv)

## Finding 1: EWU Class Labeling Is Incomplete Relative To The Original Disease Identity

The EWU dataset and the new hand-labeled dataset contain many visually identical or near-identical shrimp images. However, their class treatment differs substantially.

Dataset-level counts:

| Dataset | Images | Specimens | Labeled images | Healthy/empty images | Total masks | Co-infection images |
|---|---:|---:|---:|---:|---:|---:|
| EWU | 1126 | 409 | 574 | 552 | 1061 | 0 |
| Hand-labeled | 1149 | 416 | 746 | 403 | 1031 | 220 |

Class/mask counts:

| Dataset | Class | Instances |
|---|---|---:|
| EWU | blackgill | 101 |
| EWU | whitespot | 960 |
| Hand-labeled | BG | 462 |
| Hand-labeled | WSSV | 569 |

Interpretation:

- EWU records no co-infection images.
- The hand-labeled dataset records 220 images containing both BG and WSSV masks.
- Because the hand-labeled dataset is derived from the original disease identity convention and explicit relabeling, the absence of co-infection in EWU suggests incomplete or oversimplified class treatment.

Paper-safe claim:

> The EWU segmentation dataset appears to under-represent co-infection cases. While the hand-labeled dataset contains 220 images with both BG and WSSV masks, EWU contains no co-infection images in the analyzed export. This indicates that EWU is not sufficiently reliable as the primary optimization dataset for disease-aware segmentation.

## Finding 2: EWU Masks Are Inconsistent Across Images Within The Same Specimen Group

Each shrimp specimen can have multiple images. For the same physical shrimp, disease regions should remain anatomically consistent across views, allowing for rotation, pose change, and partial occlusion. The mask location should not arbitrarily move to unrelated body regions.

Evidence:

- See [EWU within-specimen label style](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/ewu_within_specimen_label_style_examples.png).
- See [Hand-labeled within-specimen label style](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/hand_within_specimen_label_style_examples.png).

Examples from EWU group summary:

| EWU specimen group | Images | Total masks |
|---|---:|---:|
| `shrimp::318` | 3 | 6 |
| `shrimp::319` | 3 | 6 |
| `shrimp::322` | 3 | 9 |
| `shrimp::324` | 3 | 13 |

Interpretation:

- EWU often marks several small WSSV regions per image.
- Within the same specimen, mask locations and granularity can vary noticeably across the three images.
- Human inspection suggests that some of this variation is not explained by viewpoint alone.
- In contrast, the hand-labeled dataset shows more stable disease localization across same-specimen images, especially for co-infection examples where BG and WSSV masks are consistently separated.

Paper-safe claim:

> Qualitative inspection of same-specimen groups shows that EWU masks vary in relative anatomical location and granularity across images of the same shrimp. Since disease regions should remain spatially consistent for the same physical specimen, this suggests annotation noise. The hand-labeled dataset shows more consistent within-specimen localization.

## Finding 3: Visual Hashing Confirms EWU And Hand-Labeled Sets Share Near-Identical Images With Different Labeling Style

The visual-hash report found many EWU/hand-labeled image pairs with:

- `hash_hamming = 0`
- `color_l1 = 0.0`
- `score = 0.0`

This means the paired images are visually identical under the perceptual hash and color signature used by the script.

Examples:

| EWU image | EWU masks/classes | Hand-labeled image | Hand masks/classes |
|---|---|---|---|
| `Shrimp_318-1...jpg` | 2 / whitespot | `WSSV-1-img-1...jpg` | 1 / WSSV |
| `Shrimp_322-1...jpg` | 4 / whitespot | `WSSV-5-img-1...jpg` | 1 / WSSV |
| `Shrimp_326-1...jpg` | 3 / whitespot | `WSSV-9-img-1...jpg` | 1 / WSSV |
| `Shrimp_345-1...jpg` | 7 / whitespot | `WSSV-29-img-1...jpg` | 1 / WSSV |

Evidence:

- [EWU vs hand-labeled visually similar image pairs](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/ewu_vs_hand_visual_hash_similar_pairs.png)
- [visual-hash similar pairs CSV](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/ewu_hand_visual_hash_similar_pairs.csv)

Interpretation:

- The same or near-identical image can receive different mask counts and mask shapes.
- EWU tends to fragment WSSV into multiple smaller masks.
- The hand-labeled dataset tends to use a more conservative and consistent mask standard.
- This supports the need to avoid mixing EWU with the hand-labeled dataset for final optimization unless label standards are reconciled.

Paper-safe claim:

> Perceptual hashing identifies many visually identical or near-identical images across EWU and the hand-labeled dataset. These image pairs frequently have different mask counts and mask extents. This confirms that the two datasets are not merely different splits of the same annotation standard; they encode different labeling policies.

## Finding 4: EWU Is Useful As Prior Work, But Not As The Main Optimization Dataset

EWU should not be ignored. It is useful as:

- a prior public segmentation resource
- evidence that segmentation labels existed before our relabeling effort
- a comparison point for dataset quality

However, EWU is not ideal as the main optimization dataset because:

- class treatment appears incomplete relative to original disease identity
- co-infection is absent
- same-specimen masks are qualitatively inconsistent
- visually identical images show different mask counts compared with the hand-labeled dataset

Conclusion:

> EWU is appropriate to discuss as a prior segmentation dataset, but the new hand-labeled dataset is a more reliable basis for leakage-safe baseline construction and fair Fourier optimization.

## Recommended Figure Usage In Paper

Use these figures as follows:

1. Dataset section:
   - Use [EWU vs hand-labeled visually similar image pairs](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/ewu_vs_hand_visual_hash_similar_pairs.png)
   - Purpose: show that identical images receive different mask standards.

2. Annotation-quality section:
   - Use [EWU within-specimen label style](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/ewu_within_specimen_label_style_examples.png)
   - Use [Hand-labeled within-specimen label style](C:/Users/Admin/workspace/CVio_Shrimp_Disease_Classification_Capstone_SU26/augmentation_research/baseline_opt/super_paper_fix_leakage_fourier_fair_optimization/reports/dataset_style_comparison/figures/hand_within_specimen_label_style_examples.png)
   - Purpose: show within-specimen consistency difference.

3. Motivation section:
   - Use the dataset summary table.
   - Purpose: justify why the new hand-labeled dataset is the dataset used for fair optimization.

## Text Snippet For Manuscript Draft

> We analyzed the prior EWU segmentation dataset against our hand-labeled dataset. Although both datasets contain many visually identical shrimp images, their annotation behavior differs substantially. EWU contains 574 labeled images, 552 empty-label images, and 1061 mask instances, but no co-infection images. In contrast, the hand-labeled dataset contains 746 labeled images, 403 empty-label images, 1031 mask instances, and 220 images containing both BG and WSSV masks. Visual-hash matching further shows that identical or near-identical images across the two datasets often have different mask counts and mask extents. Qualitative inspection of same-specimen EWU groups also shows inconsistent mask placement and granularity across images of the same shrimp. These observations indicate that EWU is useful as a prior segmentation resource, but not sufficiently reliable as the primary dataset for fair model optimization. Therefore, we use the new single-annotator hand-labeled dataset for leakage-safe baseline construction and Fourier preprocessing experiments.

