# English Presentation Script - Nguyen Van Phong

## Scope and timing

This script covers the slides explicitly assigned to **Phong** in the final presentation:

- Project introduction: slides 1-9
- Segmentation data and leakage control: slides 19-20
- Fourier methodology: slides 28-37
- Fourier results: slides 52-54
- Joint expanded-dataset result: slide 56, Fourier half only
- Limitations and future work: slide 60

Target speaking time: **approximately 14-15 minutes** within the team's 45-minute presentation.

Slide 55 has no presenter name next to its slide number. An optional script is provided at the end in case Phong is asked to cover it.

## Delivery notes

- Do not read every bullet or table cell. State the question, the evidence, and the conclusion.
- Use "preliminary screening" rather than "clinical diagnosis."
- Use "single-seed evidence" where applicable to the Fourier results.
- On slide 20, the `Leak` values are counts of overlapping specimen groups, not percentages.
- W1 is deterministic image preprocessing. It has no learned parameters.
- W1 was retained as an interaction-study reference, not as the universal final method.

---

## Project Introduction

### Slide 1 - Title
**Target: 15 seconds**

Good morning, respected members of the Council, our supervisor, and everyone present. We are team CVio, and our capstone project is titled "Lightweight Deep Learning for Robust Shrimp Disease Detection on Mobile Devices under Diverse Environmental Conditions." Our project combines classification, disease-region segmentation, robustness evaluation, and mobile deployment.

### Slide 2 - Our Team
**Target: 15 seconds**

Our team has three members. Tran Huu Nhan is the project leader and leads the classification research. Le Thi Huynh Nhu leads the mobile application and attention-based segmentation work. I am Nguyen Van Phong, and I am responsible mainly for segmentation data annotation, leakage-aware splitting, segmentation baselines, and Fourier-domain experiments.

### Slide 3 - Supervisor
**Target: 10 seconds**

The project was conducted under the supervision of Doctor Nguyen Dinh Vinh, who provided research direction, milestone evaluation, and academic feedback throughout the project.

### Slide 4 - Presentation Structure
**Target: 15 seconds**

Our presentation has six parts: project introduction, project management, methodology, results and evaluation, discussion, and the final demonstration. I will first introduce the problem and objectives before handing over the project-management section.

### Slide 5 - Project Introduction
**Target: 5 seconds**

I will begin with the practical pain point, the research gap, and the objectives that shaped our final scope.

### Slide 6 - Problem Statement
**Target: 40 seconds**

Shrimp disease outbreaks can cause serious production loss, but laboratory testing may be delayed, expensive, or unavailable in resource-constrained farming environments. Farmers often depend on visual inspection, which can vary with experience and image quality. This is difficult because visible symptoms may be small, sparse, or visually overlapping. Their appearance can also change under poor lighting, blur, noise, different viewpoints, and cluttered backgrounds. Therefore, our goal is not to replace laboratory diagnosis. We investigate an accessible image-based tool for preliminary screening and localization, while clearly preserving the boundary between model output and confirmed disease diagnosis.

### Slide 7 - Research Gap
**Target: 50 seconds**

Existing shrimp-disease computer-vision studies mainly focus on image classification or bounding-box detection. Lightweight CNN and YOLO models have been studied, and attention, robustness, and explainability have also been explored, but usually as separate topics. Our project addresses several gaps together. First, disease-region masks for Black Gill and WSSV are limited. Second, repeated photographs of the same shrimp can leak across training and test sets, but this risk is rarely quantified. Third, segmentation errors on Healthy images are often hidden when only aggregate mAP is reported. Finally, there is limited combined evidence covering classification, instance segmentation, corruption robustness, Fourier processing, and an executable mobile workflow.

### Slide 8 - Objectives
**Target: 50 seconds**

Our practical objectives are to develop models for shrimp disease classification and localization, improve resilience under difficult image conditions, and keep the solution feasible for resource-constrained mobile inference. Our research objectives are more specific. We enrich the data with BG and WSSV segmentation masks, prevent repeated-specimen leakage, evaluate attention-based feature enhancement, and study whether Fourier-domain preprocessing can improve disease-region segmentation. The final Android application allows farmers to capture or select an image and receive a preliminary classification result, followed by segmentation when a disease class passes the application threshold.

### Slide 9 - Project Management Plan
**Target: 5 seconds**

With the project goals established, I will now hand over to Nhan for the project management plan.

**Handoff:** "Nhan will explain how the team organized these research and implementation workstreams."

---

## Segmentation Data and Preprocessing

### Slide 19 - Segmentation Data Pipeline
**Target: 55 seconds**

For the segmentation track, we started from raw shrimp images and created project-authored disease-region polygons. The positive mask classes are Black Gill and WSSV. Healthy is not a third mask class; Healthy images are retained as empty-label negatives, which allows us to measure false disease predictions on known negative images. Co-infection images can contain separate BG and WSSV instances, while any polygon overlap is resolved by one deterministic project rule so that the same pixel is not given conflicting targets.

The original segmentation inventory contains 1,149 images from 416 filename-derived specimen groups. It includes 746 images with disease masks, 403 Healthy empty-label images, and 1,031 mask instances. After annotation, we preserve image identity, derive the specimen key from the disease category and shrimp ID, and generate a grouped train, validation, and test split. Training images may receive a defined augmentation policy, while validation and test preprocessing remain deterministic and geometry-consistent.

### Slide 20 - Why Specimen-Grouped Splitting Matters
**Target: 60 seconds**

Each physical shrimp was photographed between one and three times. If we split individual images randomly, different views of the same shrimp can appear in both training and testing. The model is then evaluated partly on specimens it has effectively seen before.

The `Leak` column shows the average number of specimen groups shared between training and testing across three seeds. Plain random splitting produced about 98.7 overlapping groups, stratified image splitting produced about 94, and grouped-specimen splitting reduced the overlap to zero. The image-level splits achieved labeled mAP50 values around 67 percent, while the grouped split achieved about 47 percent. This does not mean grouping damaged the model. It shows that repeated-specimen leakage made the random-split task artificially easier. Therefore, our principal segmentation results use the stricter grouped-specimen protocol, with 905 training images, 115 validation images, and 129 test images.

**Handoff:** "After establishing this leakage-aware data protocol, the next section presents the model backbones and proposed segmentation methods."

---

## Fourier Methodology

### Slide 28 - What Is the Fourier Transform?
**Target: 30 seconds**

An image in the spatial domain is represented by pixel positions and intensities. The Fourier transform represents the same image through spatial frequencies, meaning how rapidly intensity changes across the image. Low frequencies mainly describe broad illumination and slowly changing regions. Higher frequencies contain more rapid changes, including edges, fine textures, small spots, but also noise. The inverse Fourier transform reconstructs the spatial image after frequency-domain processing.

### Slide 29 - Why Fourier Processing?
**Target: 30 seconds**

We selected Fourier processing because it lets us control which frequency components are emphasized or suppressed. This is relevant to shrimp disease because visible evidence may include small white spots, gill boundaries, and local texture changes. Importantly, image-space Fourier processing changes pixel intensity but does not translate, rotate, crop, or rescale the shrimp. Therefore, the existing segmentation polygons remain spatially aligned. The absolute-difference panel shows that the main changes occur around edges and fine structures.

### Slide 30 - Candidate Families
**Target: 35 seconds**

The common processing path is: input image, FFT, frequency filtering, inverse FFT, reconstructed image, and then the segmentation model. We screened six image-space families. High-pass enhancement strengthens fine detail. High-frequency damping suppresses excessive detail. Band-pass filtering focuses on a selected scale. Low-frequency flattening reduces broad illumination variation. Homomorphic filtering rebalances illumination and reflectance. Gaussian low-pass suppresses fine high-frequency content. These were controlled candidates, not methods assumed to be beneficial in advance.

### Slide 31 - Strict Fourier Screening and W1
**Target: 45 seconds**

To isolate Fourier effects, the first screen used YOLO11n-seg at 640 by 640 pixels with the grouped-specimen split and seed 42. All explicit YOLO augmentation, auto augmentation, and the hidden Albumentations hook were disabled. This strict control prevents conventional augmentation from being mistaken for a Fourier benefit.

The retained reference was W1, with Gaussian scale sigma equal to 50 and residual weight alpha equal to 0.10. For each color channel, we reconstruct a smooth low-frequency component, subtract it from the original to obtain the signed high-frequency residual, scale that residual by ten percent, and add it back to the original image before clipping. W1 is deterministic and has no learned parameters.

### Slide 32 - Complete W1 Decomposition
**Target: 10 seconds**

This overview shows the full W1 decomposition in both domains: original image, low-frequency reconstruction, high-frequency residual, scaled residual, final W1 image, their spectra, and the corresponding theoretical filter responses.

### Slide 33 - Low-Frequency Component and High Residual
**Target: 25 seconds**

The middle image is the Gaussian low-frequency reconstruction, which contains broad and slowly changing appearance. Subtracting it from the original creates the signed high-frequency residual on the right. Positive and negative values describe local intensity deviations. The strongest response appears around shrimp boundaries, appendages, texture transitions, and visible spots.

### Slide 34 - Scaling and Reconstruction
**Target: 20 seconds**

The raw residual would be too aggressive if used directly, so alpha scales it to ten percent. The original image remains the dominant component. We then add the scaled residual and clip values to the valid image range. This produces a modest enhancement rather than replacing the image with an edge map.

### Slide 35 - Frequency-Domain Interpretation
**Target: 20 seconds**

In the frequency domain, the Gaussian low-pass component concentrates energy near the center, where the low frequencies are located. The residual suppresses that central low-frequency contribution and retains relatively more energy away from the center. The low-frequency energy is reduced, not necessarily forced to exactly zero, because the Gaussian response changes smoothly.

### Slide 36 - Scaled Residual and Final W1 Image
**Target: 15 seconds**

Scaling the residual by 0.10 substantially reduces its energy. When it is added back to the original, the final W1 spectrum remains visually similar to the original because the enhancement is deliberately small. The shared color scale helps us avoid exaggerating the difference.

### Slide 37 - Interaction with Strong Augmentation
**Target: 45 seconds**

This comparison uses matched stochastic augmentation paths and shows the original, W1 alone, strong augmentation alone, W1 before strong augmentation, and strong augmentation before W1. Across 24 test images and 12 matched draws per image, we obtained 288 paired observations. W1 alone increased the high-frequency energy share, but after strong augmentation its incremental high-band effect was only about 22.5 percent of the W1-only effect. This suggests that the strong spatial and color pipeline already redistributes much of the frequency content that W1 would otherwise add. The two processing orders showed limited separation in radial power, although individual augmentation families can still respond differently. This is supporting evidence for overlap or redundancy, not a multi-seed causal proof.

**Handoff:** "The results section will now show whether these controlled spectral changes translated into useful segmentation performance."

---

## Fourier Results and Evaluation

### Slide 52 - Fourier Candidate Screening
**Target: 45 seconds**

Under the strict no-augmentation, hook-off control, the no-Fourier reference achieved 23.54 percent labeled mAP50. W1 achieved the highest labeled mAP50 at 34.51 percent, an improvement of 10.97 percentage points, and the highest displayed Healthy-aware score at 21.08 percent. High-frequency damping achieved the highest mAP50-95, but it also missed more than half of the diseased images.

The W1 improvement has an important cost that is not visible in this chart: its Healthy false-positive rate increased from 39.02 to 63.41 percent. Therefore, we selected W1 as a reproducible interaction-study reference, not as a universal deployment winner.

### Slide 53 - W1 Is Condition-Dependent
**Target: 55 seconds**

This matched table is the central Fourier conclusion. With no augmentation and the hidden hook disabled, W1 improved labeled mAP50 by 10.97 points. With clean-light augmentation and the hook enabled, it produced a smaller gain of 1.98 points. However, it reduced performance under no-augmentation with the hook enabled, under clean-light with the hook disabled, and under strong augmentation.

The highest labeled mAP50 in this table is 59.51 percent from strong augmentation without Fourier. Adding W1 reduced it to 57.71 percent. Combined with the spectrum analysis, our interpretation is that W1 can supply useful appearance variation when the pipeline is otherwise weak, but it becomes partly redundant or harmful when strong augmentation already changes edges, color, and frequency distribution. Therefore, Fourier is interaction-dependent and should not be described as an always-additive enhancement.

### Slide 54 - Synthetic Noise Conditions
**Target: 20 seconds**

We also tested controlled noisy inputs. These examples show Gaussian noise, blue color cast, contrast reduction, motion blur, and salt-and-pepper noise. They preserve the original labels and allow a matched stress test, but they are synthetic corruptions and should not be interpreted as field validation or as coverage of every farm environment.

### Slide 56 - Expanded Dataset, Fourier Half
**Target: 55 seconds**

This slide uses the expanded segmentation inventory, so it is a dataset-extension assessment rather than a direct continuation of the original grouped experiment. The additional 303 images do not expose compatible specimen IDs, so the expanded split combines grouped allocation for the original subset with stratified image-level allocation for the additions.

In the Fourier block, strong augmentation without W1 achieved the higher labeled mAP50: 61.23 versus 59.90 percent. However, strong augmentation with W1 improved mAP50-95 from 23.63 to 25.13 percent, reduced Healthy false positives from 46.34 to 29.27 percent, reduced the disease miss rate from 9.17 to 5.83 percent, and increased HScore from 53.48 to 54.65 percent. This shows a trade-off: W1 slightly reduced the loose-threshold localization score but improved stricter localization quality and diagnostic balance on the expanded dataset. Because the dataset and split changed, we do not compare these numbers directly with the original grouped table.

**Joint-slide handoff:** "Nhu will cover the attention results in the upper half of this expanded-dataset table."

---

## Discussion

### Slide 60 - Limitations and Future Work
**Target: 45 seconds**

Our conclusions are bounded by several limitations. The primary classification comparison and most Fourier studies rely on one main seed, so future work should repeat the strongest candidates across three to five independent seeds. The segmentation masks were created under one project annotation protocol without independent expert boundary review, so expert and multi-annotator validation is needed. The additional segmentation images are not fully specimen-grouped because compatible identities are unavailable; future work should use stronger duplicate detection or collect explicit specimen metadata. Finally, the 21 real-world photographs and three-phone runtime observations demonstrate operational feasibility only. They do not provide veterinary ground truth or a same-image hardware benchmark. Future work should collect expert-verified field data and perform stage-separated, same-image profiling across Android and iOS devices.

---

## Optional Slide

### Slide 55 - Noise-Resilience Heatmap
**Use only if Phong is assigned this currently unlabeled slide. Target: 45 seconds**

This heatmap compares labeled mAP50 across five synthetic corruption conditions. No preprocessing method dominates every row. W1 improves over the control under Gaussian noise, salt-and-pepper noise, blur, and motion blur, but it is weaker under low contrast. Bilateral filtering is strongest for the displayed blur condition, while low-pass filtering is strongest for Gaussian noise, salt-and-pepper noise, and motion blur in this panel. The result supports a corruption-specific conclusion: high-frequency enhancement can help preserve structure under some blur conditions, but it can also amplify high-frequency noise. This is a screening comparison, not evidence of one universally robust preprocessing method.

---

## Short Defense Answers

### Why reconstruct the low-pass image and subtract it instead of applying a high-pass mask directly?

The two operations are mathematically equivalent before clipping because the complementary high-pass response is `1 - M_sigma`. The residual form is easier to interpret: it explicitly separates the original image into a smooth component and a signed detail component. It also makes the final operation clear: preserve the original image and add only `alpha` times the detail residual.

### Is W1 learnable?

No. W1 has fixed sigma and alpha values and contains no trainable parameters. It is deterministic preprocessing. The separate FDDEM feature-domain experiment introduced learnable frequency gains, but no fully audited final FDDEM result was used in the main conclusion.

### Why did W1 improve the no-augmentation model but not the strong-augmentation model?

With no augmentation, W1 supplies a controlled appearance change that emphasizes edges and fine texture. Strong augmentation already changes color, orientation, scale, interpolation, and edge structure. In the paired spectrum study, W1's incremental high-band effect after strong augmentation was only 22.5 percent of its standalone effect. The measured overlap is consistent with diminishing benefit, although the single-seed result is not a universal causal proof.

### Does the grouped split reduce model quality?

No. The grouped split removes the easier repeated-view leakage condition. Random image splits produced around 94 to 99 overlapping specimen groups and inflated labeled mAP50. The grouped result is lower because it evaluates generalization to unseen specimen identities more honestly.

### Why retain Healthy images if they have no masks?

Healthy images are known empty-label negatives. They provide no positive box or mask target, but they train the detector against spurious disease predictions and make Healthy false-positive behavior measurable during evaluation.

### Is W1 the final best segmentation method?

No. On the original grouped dataset, strong augmentation without Fourier achieved the highest labeled mAP50 in the main Fourier interaction table. W1 remains useful as a controlled frequency reference and can improve other trade-offs in some regimes, including mAP50-95, Healthy false positives, disease misses, or HScore on the expanded dataset.

### Are the corruption results equivalent to real-world validation?

No. Synthetic corruption tests isolate selected disturbances under fixed labels. The 21 phone photographs support operational testing but have no veterinary or laboratory ground truth. Neither source establishes prospective field diagnostic accuracy.

---

## Evidence Notes

- Final presentation owner labels were read directly from the 64-slide submitted deck.
- Dataset semantics, split counts, Fourier equations, W1 interpretation, and limitations were checked against the final thesis.
- The Fourier candidate and interaction metrics match the final thesis tables for the strict screen and augmentation-regime comparison.
- Slide 20's `98.7` and `94.0` values originate from the recorded mean counts of overlapping specimen groups across three seeds. They should not be spoken as percentages.
- Slide 55 is not assigned to a presenter in the submitted deck and is therefore treated as optional.
