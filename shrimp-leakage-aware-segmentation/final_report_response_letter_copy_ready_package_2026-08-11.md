# Copy-Ready Final Report and Response Letter Package

Date: 2026-08-11

## 1. Purpose and use

This package converts the short notes in `Danh sách chỉnh lý.docx` into:

1. exact English wording for the Final Report;
2. exact tables, figures, captions, and explanatory paragraphs;
3. complete response-letter entries in Vietnamese and English;
4. stable section/table/figure locations and evidence paths.

The current PDF page numbers are included for navigation. The final response letter must use the page numbers from the recompiled revised PDF, because adding figures and paragraphs will change pagination.

No DOCX, PDF, slide, or XeLaTeX source is modified by this package.

## 2. Assessment of the selected checklist

| Checklist item | Verdict | What the short note is missing |
|---|---|---|
| Healthy images are actual empty-label negatives | Correct and required | The table and surrounding paragraph must explain both training supervision and Healthy false-positive diagnostics. |
| WSSV takes priority in overlapping annotated regions | Correct as the project's stated annotation policy | It must not be described as a general YOLO limitation or as a biological claim that WSSV replaces BG. |
| Expanded data use mixed grouped/random allocation | Correct and required | The report must state that the complete expanded split is not fully specimen-leakage-free and remains secondary to the original grouped dataset. |
| `3.1.1` subsection number is wrong | Correct | Use `3.2.1`; this preserves the following `3.3` section and avoids renumbering the rest of Chapter IV. |
| Random erasing was not active in YOLO segmentation | Correct and required | Remove it from both active and disabled operation lists. Add a footnote that the argument was recorded but ignored by Ultralytics 8.4.62 segmentation `v8_transforms`. |
| Replace the Fourier visualization | Correct | The embedded 18-panel figure is too dense for the main report. Use the clearer 8-panel figure in Methodology and move the full decomposition to the appendix. |
| Define train-test overlap | Correct and required | Include the mathematical denominator and state that the values in Table 6.10 are mean and standard deviation across runs/seeds. |

## 3. Exact Final Report amendments

### 3.1 Table 4.4: annotation semantics

**Current location:** Chapter IV, Section 3.1, PDF physical pages 70-71, printed pages 61-62.

Replace the current Table 4.4 with the following content.

**Table 4.4. Segmentation annotation rules and label semantics**

| Label/status | Positive annotation rule | Exclusion or handling rule |
|---|---|---|
| BG | Visibly darkened gill tissue with a coherent disease-region boundary. | Exclude general shadows, background darkness, and uncertain color changes. |
| WSSV | Visible white spots or white-lesion regions consistent with WSSV appearance. | Exclude glare, isolated reflections, and background white marks. |
| Co-infection image | Annotate spatially separable BG and WSSV regions as their respective classes. | When annotated disease regions overlap, retain one deterministic target; WSSV takes project-specific priority in the overlapping pixels. |
| Healthy | No positive disease-mask instance is assigned. The image is retained as an empty-label negative sample. | Include Healthy images in the train, validation, and test partitions. Use them for negative supervision and Healthy false-positive diagnostics. |
| Ambiguous region | None. | Omit uncertain pixels rather than forcing a disease label. |
| Display color | Visualization only. | Class identity is determined by the stored class ID, not by overlay color. |

Replace or extend the paragraph immediately after the table with:

> Ultralytics YOLO segmentation supports multiple instances and multiple classes in the same image. The project therefore retained separable BG and WSSV regions as independent class-specific instances in co-infection images. A multilabel semantic target was not assigned to the same annotated pixel. When the project-authored BG and WSSV regions overlapped, the overlap was resolved through one deterministic annotation rule in which WSSV took project-specific priority. This policy was adopted to keep the stored training targets consistent; it is not a claim that YOLO cannot represent multiple disease instances, nor is it a biological claim that one disease replaces the other.

Add the following paragraph for Healthy images:

> Healthy images are known negative samples rather than missing annotations. Their label files contain no positive disease polygon because no visible BG or WSSV region is present. In training, no matched positive instance contributes box or mask regression terms for these images, while the detection/classification branch still receives negative supervision against disease predictions. Healthy behavior is therefore reported separately through the Healthy false-positive rate and the number of false masks per Healthy image.

### 3.2 Dataset inventory and split wording

**Current location:** Chapter IV, Section 3.2 and the incorrectly numbered `3.1.1`, PDF physical page 72, printed page 63.

Change the heading to:

> **3.2.1 Instance-Segmentation Dataset Construction and 80/10/10 Partition Summary**

Use the following replacement text for the original dataset:

> The primary instance-segmentation dataset contained 1,149 images associated with 416 filename-derived specimen groups. It included 746 images with at least one project-authored BG or WSSV mask and 403 Healthy images retained as empty-label negatives. Across the dataset, 1,031 mask instances were recorded: 462 BG instances and 569 WSSV instances. A total of 220 images represented the WSSV_BG co-infection image category; these images retained BG and WSSV as the two mask classes rather than introducing a third co-infection mask class.

> The primary dataset was divided using disease-stratified grouped-specimen allocation. All views sharing the specimen key `Disease::ShrimpID` were assigned to the same partition. The resulting split contained 905/115/129 train/validation/test images and 331/40/45 specimen groups. No specimen key was shared between train and validation, train and test, or validation and test.

Use the following replacement text for the expanded dataset:

> The expanded dataset was evaluated only as an additional dataset-composition transfer assessment. It combined the 1,149 convention-matched images with 303 added images whose filenames did not provide compatible specimen identifiers. The convention-matched subset retained grouped-specimen allocation, whereas the 303 unmatched-name images were allocated by image-level stratified random splitting before the partitions were merged. The final expanded split contained 1,147 training images, 144 validation images, and 161 test images. Because specimen identity was unavailable for the unmatched subset, the complete expanded split cannot be described as fully specimen-leakage-free. The accurate scope is that group leakage was prevented for convention-matched images, while unmatched images were stratified at image level.

Keep the existing expanded inventory table, but add this note directly below it:

> **Note:** “Recognized specimen-group images” retain the grouped allocation. “Added unmatched images” use image-level stratified random allocation because compatible specimen identifiers are unavailable.

### 3.3 Table 4.6: augmentation and control policies

**Current location:** Chapter IV, Section 3.3, PDF physical page 74, printed page 65.

Replace Table 4.6 with:

**Table 4.6. Segmentation augmentation and control policies**

| Policy | Active photometric settings | Active geometric settings | Explicitly disabled components |
|---|---|---|---|
| Clean-light | HSV `h/s/v = 0.01/0.35/0.20`. | Horizontal flip `0.50`; translation `0.05`; scale `0.20`. | Vertical flip, rotation, shear, perspective, mosaic, MixUp, CutMix, copy-paste, and auto augmentation. |
| Strict control | All explicit photometric augmentation set to zero. | All explicit geometric augmentation set to zero. | Hidden Albumentations hook and all online augmentation. |
| Stronger candidate | HSV `h/s/v = 0.05/0.50/0.40`. | Horizontal/vertical flip `0.50/0.30`; rotation `10` degrees; translation `0.10`; scale `0.50`. | Mosaic, MixUp, CutMix, copy-paste, shear, perspective, auto augmentation, and the hidden Albumentations hook. |

Add this footnote beneath the table:

> **Implementation note:** The recorded strong-policy argument dictionary contained `erasing=0.15`; however, Ultralytics 8.4.62 applies this argument in the classification transform builder and does not use it in the standard segmentation `v8_transforms` builder. Random erasing was therefore not an active operation in this strong YOLO segmentation policy. This does not affect the separate N=1 experiments that implemented erasing explicitly as custom offline preprocessing.

### 3.4 W1 formulation and main Methodology figure

**Current location:** Chapter IV, Section 4.3, PDF physical pages 79-80, printed pages 70-71.

Use this exact formulation:

> For each image channel `x_c`, the method first computed the two-dimensional Fourier spectrum `F_c = FFT2(x_c)`. A centered Gaussian low-pass response was defined as `G_sigma(u,v) = exp[-D(u,v)^2/(2 sigma^2)]`, where `D(u,v)` is the radial distance from the shifted frequency origin. The low-frequency reconstruction was then obtained as `x_low,c = Re{IFFT2(G_sigma F_c)}`. The complementary high-frequency residual was calculated in image space as `x_high,c = x_c - x_low,c`, and the W1 output was produced by `x_W1,c = clip[x_c + alpha x_high,c, 0, 255]`.

> The selected W1 configuration used `sigma=50` and `alpha=0.10`. Its equivalent linear frequency gain before clipping is `H_W1(u,v) = 1 + alpha[1 - G_sigma(u,v)]`. The gain is approximately 1 near the low-frequency origin and approaches 1.10 at high frequencies. W1 therefore preserves the original low-frequency image content while mildly increasing the contribution of the complementary high-frequency residual; it is not a binary high-pass output that discards the original image.

> Computing the high-frequency component by subtracting the Gaussian low-frequency reconstruction is mathematically equivalent to applying the complementary response `1-G_sigma` to the spectrum before inverse transformation. The subtraction form was used because the same low-frequency reconstruction is directly interpretable in the spatial domain and makes the residual identity `x = x_low + x_high` easy to verify.

Replace the current Figure 4.10 image with:

`shrimp-leakage-aware-segmentation/fourier_vis_output/fourier_visualization_outputs/w1_defense_spatial_frequency_comparison.png`

Use this caption:

> **Figure 4.10. Representative spatial- and frequency-domain effect of W1 (`sigma=50`, `alpha=0.10`).** The panels compare the original and transformed image, the spatial absolute difference, the theoretical frequency gain, the original and W1 log-power spectra on a shared scale, the signed spectral change, and the radial mean power profile. The original image remains recognizable because W1 retains the full image and adds only a scaled high-frequency residual.

Do not use the 18-panel decomposition as the main Figure 4.10 because its labels become too small at report width. Place it in Appendix D using:

`shrimp-leakage-aware-segmentation/fourier_vis_output/fourier_visualization_outputs/w1_spatial_frequency_decomposition.png`

Appendix caption:

> **Figure D.x. Exact W1 decomposition in the spatial and frequency domains.** The figure shows the original image, Gaussian low-frequency reconstruction, signed high-frequency residual, scaled residual, W1 result before clipping, and W1 result after clipping and `uint8` quantization, together with their spectra and theoretical filter responses. The reconstruction audit verified `x = x_low + x_high` up to floating-point precision. Clipping and quantization are nonlinear and therefore do not have one linear shift-invariant frequency response.

### 3.5 Spectrum-interaction protocol in Methodology

**Current location:** Chapter IV, Section 4.6, PDF physical page 80, printed page 71.

Replace or extend Section 4.6 with:

> To examine the interaction between W1 and the recorded strong augmentation policy, a paired image-spectrum study was conducted on 24 held-out test images sampled with seed 42. The sample contained six Healthy, six BG, six WSSV, and six WSSV_BG images. Each image was processed under 12 matched augmentation realizations and five image paths: original, W1 only, strong augmentation only, W1 followed by strong augmentation, and strong augmentation followed by W1. The same sampled augmentation parameters were reused across the matched paths for each image-realization pair.

> The active strong-policy replica followed the Ultralytics 8.4.62 segmentation transform order and equations for letterbox resizing, random rotation, translation, scale, HSV transformation, vertical flip, and horizontal flip. Mosaic, MixUp, CutMix, copy-paste, shear, perspective, and the hidden Albumentations hook were disabled. Random erasing was excluded because the segmentation transform builder did not use the recorded `erasing` argument.

> The analysis produced 288 paired observations for each global image metric and 216 observations for lesion-region metrics because Healthy images contain no lesion mask. Recorded measurements included normalized low-, mid-, and high-band spectral-energy share, radial power, mean gradient magnitude, Laplacian variance, lesion-region gradient, background gradient, and clipping fraction. This experiment measured image-distribution and spectrum changes. It did not retrain the complete model under both strong-policy processing orders and therefore cannot by itself establish a causal relationship between spectral changes and segmentation mAP.

### 3.6 Definition of train-test overlap

**Current location:** Chapter VI, Section 10.2, Table 6.10, PDF physical pages 119-120, printed pages 110-111.

Add this definition before or immediately after Table 6.10:

> For a given split, train-test overlap was defined as `100 x |G_train intersect G_test| / |G_test|`, where `G_train` and `G_test` are the sets of filename-derived specimen keys in the training and test partitions. The measure is therefore the percentage of test specimen groups that also appear in training. It is not the percentage of image files that overlap and not the fraction of views from an individual specimen that leaked. Table 6.10 reports the mean and standard deviation of this group-level quantity across the three recorded seeds (42, 123, and 3407).

Recommended revised column header:

> **Test specimen groups also present in train (%)**

### 3.7 YOLO11n-seg selection paragraph

**Current location:** Chapter VI, Section 10.1, immediately after Table 6.9, PDF physical page 119, printed page 110.

Replace the current selection paragraph with:

> YOLO11n-seg was retained as the reference architecture because it provided the strongest overall localization-efficiency trade-off in the recorded grouped-specimen comparison, rather than because it dominated every metric. With 2.877 million parameters, it achieved the highest full-test mask mAP50 (48.2%), labeled-only mask mAP50 (51.2%), and HScore (46.0%) in the six-model panel. YOLO26s-seg achieved the highest labeled mAP50-95 (18.2%) but used 11.506 million parameters and produced lower main mAP50 and HScore. YOLO26n-seg achieved the lowest Healthy false-positive rate (19.5%) but lower full-test and labeled-only mAP50. YOLO11n-seg was therefore selected as the compact balanced baseline for subsequent attention, preprocessing, and augmentation experiments; the result is specific to the recorded grouped-split protocol and is not a universal architecture ranking.

### 3.8 New Results subsection for the council's Fourier question

**Insertion location:** Chapter VI, within Section 12.2, immediately after Table 6.16 and Figure 6.15. Use the subsection number `12.2.1` so the existing Section 12.3 does not need to be renumbered.

**Heading:**

> **12.2.1 Measured Spectrum Interaction with Strong Augmentation**

Insert this table:

**Table 6.x. Paired incremental effect of W1 before and after matched strong augmentation**

| Metric | W1 effect without augmentation | Incremental W1 effect after matched strong augmentation | Retained W1 effect |
|---|---:|---:|---:|
| Low-band energy share | -0.001158 | -0.000434 | 37.5% |
| Mid-band energy share | +0.000488 | +0.000283 | 58.0% |
| High-band energy share | +0.000670 | +0.000151 | 22.5% |
| Mean gradient | +0.010979 | +0.006157 | 56.1% |
| Laplacian variance | +0.003127 | +0.000660 | 21.1% |
| Lesion-region gradient | +0.012104 | +0.009561 | 79.0% |

Add this note below the table:

> **Note:** Values are paired mean changes produced by adding W1 to the corresponding base path; they are not absolute spectral-energy shares. “Retained W1 effect” is the ratio between the incremental W1 effect after augmentation and the W1-only effect. The lesion-region metric excludes Healthy images.

Use the following Results text:

> The paired spectrum analysis showed that strong augmentation did not completely remove the W1 effect, but substantially reduced or redistributed its incremental global high-frequency emphasis. Without augmentation, W1 increased the high-band energy share by 0.000670 on average. After the same images had undergone matched strong augmentation, the additional increase caused by W1 was 0.000151, corresponding to 22.5% of the original W1 effect. The retained effects were 21.1% for Laplacian variance and 56.1% for mean gradient. The lesion-region gradient retained approximately 79.0%, indicating that W1 continued to enhance some local disease-region boundaries even though most of its global spectral contribution overlapped with the augmented image distribution.

> The active strong policy introduced HSV variation, horizontal and vertical flipping, rotation, translation, and scale. Affine interpolation can attenuate fine detail, while rotation, translation, padding, and image boundaries introduce structured frequency components. HSV also changes intensity relationships between channels. In the recorded Path 15 training pipeline, W1 was applied offline before online strong augmentation; W1-enhanced edges were therefore subsequently resampled and color-transformed. The strong-augmentation-only model already learned from a broader distribution, so adding a fixed high-pass bias could become redundant or overemphasize texture and noise that were not useful for localization.

> This mechanism is consistent with the observed seed-42 model interaction: labeled mask mAP50 changed from 0.2354 to 0.3451 under strict no augmentation, but from 0.5951 to 0.5771 under strong augmentation. The spectrum analysis supports an overlap or redistribution explanation, but it does not prove that the measured spectral change caused the mAP reduction. Multi-seed matched training would be required before describing the negative interaction as statistically confirmed.

Use these two images in the new subsection:

1. `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_wssv_spatial_spectrum_orders.png`
2. `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_w1_band_energy_interaction.png`

Recommended captions:

> **Figure 6.x(a). Representative matched WSSV image under five processing paths.** The rows show the spatial image, log-power spectrum, and signed spectral change relative to the original path. Strong augmentation introduces larger geometric and color changes than W1 alone, while the difference between the two combined orders is visually smaller. This image is illustrative; the quantitative findings use all 24 images and 12 realizations.

> **Figure 6.x(b). Incremental W1 spectral-energy effect before and after matched strong augmentation.** Red bars show the mean change produced by W1 without augmentation, and green bars show the additional W1 change after the corresponding strongly augmented image. Strong augmentation retained only 22.5% of W1's original high-band energy-share effect.

### 3.9 Required limitations text

**Location:** Chapter VI, Section 16, and Chapter VII, Sections 3.2-3.5. Current printed pages 124 and 130-131.

Add this segmentation-specific limitation paragraph:

> The expanded dataset does not establish controlled field generalization across farms, environments, or camera devices. Device identity, native camera resolution, acquisition conditions, and independent expert or laboratory ground truth were not retained as controlled factors. The expanded experiment therefore measures transfer to a different data composition rather than field accuracy. Likewise, YOLO11n-seg's 2.877-million-parameter size supports describing the model as compact, but does not demonstrate mobile segmentation readiness. The final segmentation checkpoint was not exported and benchmarked end to end on Android or iOS, and the recorded W1 timing was obtained on a Kaggle GPU rather than a mobile device.

Add this U-Net limitation/future-work paragraph:

> A matched U-Net benchmark was not completed in the retained evidence set. The implemented task used two-class instance segmentation with Healthy empty-label negatives, whereas a standard U-Net produces semantic masks. A fair comparison would rasterize the same BG/WSSV polygons under the same grouped-specimen manifest, use a matched training budget, and report shared semantic Dice, IoU, and mIoU metrics. Native YOLO instance-level mAP should not be compared directly with U-Net semantic Dice without an equivalent instance-extraction protocol. This benchmark remains future work and no claim of YOLO superiority over U-Net is made.

## 4. Response Letter entries

### 4.1 Reviewer 2, Question 2: Fourier and strong augmentation

**Ý kiến gốc | Original feedback**

> “Results - when no augmentation is used, Fourier high-pass filtering provides a massive boost in labeled mAP50 (23.54% to 34.51%). Strong Augmentation policy (hook off), applying Fourier filtering actually decreased performance from 59.51% down to 57.71%. Why does a strong aug pipeline make Fourier freq domain enhancement redundant/harmful? What happens to the image spectrum when strong spatial/color aug are combined with a high pass Fourier filter? SV: Có check image spectrum, chưa show trong report.”

**Bản dịch tiếng Anh | English translation**

> Without augmentation, Fourier high-pass filtering improves labeled mAP50 from 23.54% to 34.51%. Under the strong-augmentation policy with the hook disabled, applying Fourier filtering reduces performance from 59.51% to 57.71%. Why can strong augmentation make Fourier-domain enhancement redundant or harmful, and what happens to the image spectrum when spatial/color augmentation is combined with a high-pass Fourier filter? The team stated that the image spectrum had been checked but was not shown in the report.

**Phản hồi của nhóm | Tiếng Việt**

> Nhóm cảm ơn góp ý của Cô. Nhóm đã bổ sung phân tích phổ trực tiếp thay vì chỉ suy luận từ mAP. Trong cùng protocol seed 42, cùng YOLO11n-seg và cùng phép chia grouped-specimen, W1 làm labeled mask mAP50 tăng từ 0.2354 lên 0.3451 khi tắt augmentation và hidden Albumentations hook, tương đương +10.97 điểm phần trăm. Tuy nhiên, khi dùng strong augmentation với hook vẫn tắt, W1 làm mAP50 giảm từ 0.5951 xuống 0.5771, tương đương -1.80 điểm phần trăm. Vì vậy, hiệu quả của W1 phụ thuộc vào augmentation regime chứ không phải là một cải thiện cố định.
>
> Phân tích mới sử dụng 24 ảnh test, gồm 6 ảnh cho mỗi nhóm Healthy, BG, WSSV và WSSV_BG, với 12 lần sinh augmentation có tham số ngẫu nhiên được ghép cặp. Trên 288 cặp quan sát, W1 khi đứng một mình làm tỷ trọng năng lượng dải cao tăng trung bình 0.000670, trong khi phần tăng bổ sung của W1 sau strong augmentation chỉ còn 0.000151, tương đương 22.5% hiệu ứng ban đầu. Mức tăng Laplacian variance còn 21.1%, mean gradient còn 56.1%, và lesion-region gradient còn khoảng 79.0%. Kết quả cho thấy strong augmentation không triệt tiêu hoàn toàn W1, nhưng đã chồng lấp hoặc tái phân phối phần lớn nhấn mạnh cao tần toàn cục của W1.
>
> Trong pipeline đã ghi nhận, W1 được áp dụng offline trước rồi strong augmentation được áp dụng online. Các phép affine có nội suy tiếp tục resample các biên đã được W1 tăng cường, trong khi HSV thay đổi quan hệ cường độ giữa các kênh màu. Mô hình strong-augmentation-only đã học từ một phân phối đa dạng hơn, nên high-pass bias cố định có thể trở nên dư thừa hoặc nhấn mạnh texture/nhiễu không có ích. Đây là cơ chế phù hợp với phổ ảnh và kết quả seed 42, nhưng nhóm không trình bày nó như bằng chứng nhân quả hoặc xác nhận đa seed.

**Authors' response | English**

> We thank the reviewer for this comment. We added direct spectrum analysis rather than inferring the mechanism from mAP alone. Under the same seed-42 protocol, YOLO11n-seg model, and grouped-specimen split, W1 increased labeled mask mAP50 from 0.2354 to 0.3451 when augmentation and the hidden Albumentations hook were disabled, a gain of 10.97 percentage points. Under strong augmentation with the hook still disabled, W1 reduced mAP50 from 0.5951 to 0.5771, a decrease of 1.80 percentage points. W1 is therefore augmentation-regime dependent rather than uniformly beneficial.
>
> The new analysis used 24 test images, with six images from each Healthy, BG, WSSV, and WSSV_BG stratum, and 12 matched random augmentation realizations. Across 288 paired observations, W1 alone increased high-band energy share by 0.000670 on average, whereas the incremental W1 effect after strong augmentation was 0.000151, or 22.5% of the original effect. The retained effects were 21.1% for Laplacian variance, 56.1% for mean gradient, and approximately 79.0% for lesion-region gradient. Strong augmentation therefore did not remove W1 completely, but overlapped with or redistributed most of W1's global high-frequency emphasis.
>
> In the recorded pipeline, offline W1 was followed by online strong augmentation. Affine interpolation subsequently resampled W1-enhanced edges, while HSV changed intensity relationships among color channels. The strong-augmentation-only model already learned from a broader distribution, so a fixed high-pass bias could become redundant or overemphasize unhelpful texture or noise. This mechanism is consistent with the measured spectra and seed-42 results, but we do not present it as causal proof or multi-seed confirmation.

**Thay đổi trong Final Report | Tiếng Việt**

> Nhóm đã sửa Section 4.6 để mô tả protocol phân tích phổ gồm 24 ảnh test, 4 strata, 12 matched realizations và 5 processing paths. Nhóm bổ sung Section 12.2.1 với bảng paired spectral effects, hình ảnh không gian/phổ của năm processing paths và biểu đồ thay đổi năng lượng low/mid/high band. Figure 4.10 cũng được thay bằng hình thể hiện đồng thời thay đổi trong miền không gian, phổ log-power, theoretical gain và radial power. Phần thảo luận được giới hạn ở kết luận condition-dependent và nêu rõ chưa có multi-seed causal confirmation.

**Revision in Final Report | English**

> Section 4.6 was revised to document the 24-image, four-stratum, 12-realization, five-path paired spectrum protocol. A new Section 12.2.1 was added with a paired spectral-effect table, spatial/spectrum processing-path figure, and low/mid/high-band interaction chart. Figure 4.10 was replaced with a visualization that jointly reports spatial change, log-power spectra, theoretical gain, spectral change, and radial power. The discussion now limits the claim to an augmentation-regime-dependent interaction and explicitly states that multi-seed causal confirmation was not performed.

**Vị trí chỉnh sửa | Location**

> Chapter IV, Sections 4.3 and 4.6, Figure 4.10; Chapter VI, Section 12.2 and new Section 12.2.1, Table 6.x and Figure 6.x. Current source locations: printed pages 70-71 and 114-115. Revised Final Report pp. **[fill after recompilation]**.

**Minh chứng | Evidence**

- `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_paired_w1_effects_aggregate.csv`
- `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_w1_band_energy_interaction.png`
- `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_wssv_spatial_spectrum_orders.png`
- `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/ultralytics_augmentation_contract.json`
- `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/11to20/015_fourier_attention_interaction/015_modeA-fouirer-hev-aug-attention.ipynb`
- `shrimp-leakage-aware-segmentation/ONLY_fourier/[hand-explanation]final_metrics_and_training_configs_registry.md`

### 4.2 Reviewer 3, Question 1: Parameters and Assessment column

**Ý kiến gốc | Original feedback**

> “Bổ sung table in slide 39/54, không được chỉnh sửa slide quá nhiều. SV có so sánh parameters của các models không? SV: có ⇒ làm thêm 1 cột để đánh giá.”

**Bản dịch tiếng Anh | English translation**

> Add the comparison table to slide 39/54 without changing the slide deck substantially. The team has compared model parameters, so add one assessment column.

**Phản hồi của nhóm | Tiếng Việt**

> Nhóm đã bổ sung cột Parameters/Assessment vào bảng segmentation trên slide và giữ nguyên cấu trúc chính của slide. Đánh giá không chỉ dựa trên parameter count mà kết hợp full-test mAP50, labeled-only mAP50, mAP50-95, Healthy false-positive rate và HScore. YOLO11n-seg có 2.877 triệu tham số và đạt full/labeled mAP50 cùng HScore cao nhất trong panel grouped-specimen. YOLO26s-seg đạt mAP50-95 cao nhất nhưng có 11.506 triệu tham số, còn YOLO26n-seg có Healthy FP thấp nhất nhưng main mAP50 thấp hơn. Vì vậy, cột Assessment mô tả trade-off theo đúng metric thay vì xếp hạng tuyệt đối.

**Authors' response | English**

> We added a Parameters/Assessment column to the segmentation comparison slide while preserving its original structure. The assessment combines parameter count with full-test mAP50, labeled-only mAP50, mAP50-95, Healthy false-positive rate, and HScore. YOLO11n-seg has 2.877 million parameters and achieved the highest full/labeled mAP50 and HScore in the grouped-specimen panel. YOLO26s-seg achieved the highest mAP50-95 but used 11.506 million parameters, while YOLO26n-seg achieved the lowest Healthy false-positive rate but lower main mAP50. The added column therefore reports metric-specific trade-offs rather than an absolute architecture ranking.

**Thay đổi trong Final Report | Tiếng Việt**

> Đoạn giải thích ngay sau Table 6.9 được viết lại để nêu đầy đủ parameter count và trade-off của YOLO11n-seg, YOLO26s-seg và YOLO26n-seg. Bảng trên slide được bổ sung cột Parameters/Assessment dựa trực tiếp trên Table 6.9 và đoạn giải thích này.

**Revision in Final Report | English**

> The paragraph following Table 6.9 was rewritten to report the parameter counts and metric-specific trade-offs of YOLO11n-seg, YOLO26s-seg, and YOLO26n-seg. The slide comparison table was updated with a Parameters/Assessment column derived directly from Table 6.9 and the revised explanation.

**Vị trí chỉnh sửa | Location**

> Chapter VI, Section 10.1, Table 6.9 and its following paragraph; current printed page 110; revised Final Report p. **[fill after recompilation]**. Revised presentation: slide **[fill final slide number]**.

**Minh chứng | Evidence**

- Final Report Table 6.9, six-model grouped-specimen comparison.
- Recorded YOLO11n-seg baseline row in `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/only_fourier_factorial_A_H_highpass_s50_a0p10_summary.csv`.
- The complete raw six-model panel should be archived centrally if the executed output is still available.

### 4.3 Reviewer 3, Question 2: Why YOLO11n-seg was selected

**Ý kiến gốc | Original feedback**

> “Tại sao chọn YOLO11? Có bản so sánh không? Bổ sung explanation cho table 6.3.”

**Bản dịch tiếng Anh | English translation**

> Why was YOLO11 selected? Is there a comparison? Add an explanation to the relevant comparison table.

**Phản hồi của nhóm | Tiếng Việt**

> YOLO11n-seg được chọn làm reference cho các thí nghiệm segmentation tiếp theo vì đạt trade-off tổng thể tốt nhất trong bảng so sánh sáu mô hình trên cùng grouped-specimen protocol, không phải vì đứng đầu mọi metric. Mô hình có 2.877 triệu tham số, đạt full-test mask mAP50 48.2%, labeled-only mask mAP50 51.2%, labeled-only mAP50-95 16.8%, Healthy FP 24.4% và HScore 46.0%. Trong panel này, YOLO11n-seg đạt full-test mAP50, labeled-only mAP50 và HScore cao nhất. YOLO26s-seg có mAP50-95 cao hơn, còn YOLO26n-seg có Healthy FP thấp hơn. Vì vậy, YOLO11n-seg được giữ làm compact balanced baseline cho các controlled experiments, không được diễn giải là tốt nhất trên mọi metric hoặc mọi dataset.

**Authors' response | English**

> YOLO11n-seg was retained for subsequent segmentation experiments because it provided the best overall trade-off in the six-model grouped-specimen comparison, not because it dominated every metric. With 2.877 million parameters, it recorded full-test mask mAP50 of 48.2%, labeled-only mask mAP50 of 51.2%, labeled-only mAP50-95 of 16.8%, a Healthy false-positive rate of 24.4%, and HScore of 46.0%. It achieved the highest full-test mAP50, labeled-only mAP50, and HScore in this panel. YOLO26s-seg achieved higher mAP50-95, while YOLO26n-seg achieved a lower Healthy false-positive rate. YOLO11n-seg was therefore selected as the compact balanced baseline for the controlled experiments, not as the best model for every metric or dataset.

**Thay đổi trong Final Report | Tiếng Việt**

> Nhóm đã viết lại đoạn ngay sau Table 6.9 để nêu rõ sáu mô hình được so sánh trên grouped-specimen split, các metric mà YOLO11n-seg đứng đầu, các metric mà YOLO26s-seg và YOLO26n-seg tốt hơn, và lý do lựa chọn dựa trên localization-efficiency trade-off. Số bảng được dẫn theo numbering hiện tại là Table 6.9 thay vì Table 6.3 trong biên bản.

**Revision in Final Report | English**

> The paragraph immediately following Table 6.9 was rewritten to identify the six grouped-specimen candidates, the metrics led by YOLO11n-seg, the metrics led by YOLO26s-seg and YOLO26n-seg, and the localization-efficiency trade-off used for selection. The revised response references the current table number, Table 6.9, rather than the council minute's earlier Table 6.3 numbering.

**Vị trí chỉnh sửa | Location**

> Chapter VI, Section 10.1, Table 6.9 and the following paragraph; current printed page 110; revised Final Report p. **[fill after recompilation]**.

**Minh chứng | Evidence**

- Final Report Table 6.9.
- Baseline metrics: full mAP50 `0.481532`, labeled mAP50 `0.512002`, labeled mAP50-95 `0.167785`, Healthy FP `0.243902`, HScore `0.459581`.
- `shrimp-leakage-aware-segmentation/research/optimization_deep_research_2026.md`.

### 4.4 Reviewer 3, Question 3: U-Net comparison if time permits

**Ý kiến gốc | Original feedback**

> “Segmentation bổ sung so sánh với U-Net nếu có thời gian.”

**Bản dịch tiếng Anh | English translation**

> Add a segmentation comparison with U-Net if time permits.

**Phản hồi của nhóm | Tiếng Việt**

> Nhóm ghi nhận đây là yêu cầu có điều kiện. Trong evidence set hiện tại, nhóm chưa hoàn thành một thí nghiệm U-Net được kiểm chứng trên cùng grouped-specimen split và training budget, vì vậy báo cáo không tuyên bố YOLO tốt hơn U-Net. Bài toán hiện tại được xây dựng dưới dạng instance segmentation hai lớp với ảnh Healthy không có positive mask, trong khi U-Net tiêu chuẩn là semantic segmentation. Một so sánh công bằng cần rasterize cùng polygon BG/WSSV theo cùng manifest, huấn luyện với ngân sách tương đương và báo cáo Dice, IoU và mIoU chung. Native instance-level mAP của YOLO không thể được so trực tiếp với semantic Dice của U-Net nếu chưa có instance-extraction protocol tương đương. Nội dung này được bổ sung vào Limitations and Future Work thay vì tạo số liệu chưa được xác minh.

**Authors' response | English**

> We acknowledge that this was a conditional request. No verified U-Net experiment was completed using the same grouped-specimen split and training budget, so the report does not claim that YOLO outperforms U-Net. The current task is two-class instance segmentation with Healthy images containing no positive masks, whereas a standard U-Net performs semantic segmentation. A fair comparison would rasterize the same BG/WSSV polygons under the same manifest, use a comparable training budget, and report shared Dice, IoU, and mIoU metrics. Native YOLO instance-level mAP should not be compared directly with U-Net semantic Dice without an equivalent instance-extraction protocol. We added this limitation and benchmark design to Future Work rather than introducing unverified numbers.

**Thay đổi trong Final Report | Tiếng Việt**

> Section 3.2 Segmentation Limitations và Section 3.5 Prioritized Future Work của Chapter VII được bổ sung để nêu rõ chưa có matched U-Net benchmark, lý do native metrics không trực tiếp tương đương, và protocol cần thiết cho một so sánh công bằng.

**Revision in Final Report | English**

> Chapter VII, Section 3.2 Segmentation Limitations and Section 3.5 Prioritized Future Work were extended to state that no matched U-Net benchmark was completed, explain why the native metrics are not directly equivalent, and define the protocol required for a fair comparison.

**Vị trí chỉnh sửa | Location**

> Chapter VII, Sections 3.2 and 3.5; current printed pages 130-131; revised Final Report pp. **[fill after recompilation]**.

**Minh chứng | Evidence**

- Repository and retained-artifact audit found no verified U-Net training/evaluation row under the same manifest.
- Original grouped split manifest and BG/WSSV polygon labels define the proposed future comparison protocol.

### 4.5 Reviewer 1, Question 2: real-world multi-camera evaluation

This is a shared group question. The text below is the segmentation-specific contribution; the classification/deployment owner must integrate any additional evidence.

**Ý kiến gốc | Original feedback**

> “Mô hình có thực sự tổng quát với điều kiện thực tế không? Bổ sung thử nghiệm trên ảnh thực tế ở các môi trường khác nhau, điện thoại có camera độ phân giải khác nhau không phải ảnh từ dataset đã tạo.”

**Bản dịch tiếng Anh | English translation**

> Does the model truly generalize to real-world conditions? Add experiments using new real photographs captured in different environments and with phones having different camera resolutions, rather than images from the prepared datasets.

**Phản hồi của nhóm | Tiếng Việt**

> Nhóm đồng ý đây là một giới hạn quan trọng. Trong phạm vi segmentation, evidence hiện tại chưa có một field test set độc lập được kiểm soát theo thiết bị, độ phân giải camera, môi trường chụp và ground truth chuyên gia hoặc xét nghiệm. Tập expanded 1,452 ảnh chỉ đánh giá sự thay đổi về thành phần dữ liệu; nó không thể được sử dụng để chứng minh multi-camera field generalization vì 303 ảnh bổ sung không có specimen identifier tương thích và metadata thiết bị không được lưu như một biến kiểm soát. Do đó, bản sửa đổi giới hạn kết luận ở dataset-composition transfer và bổ sung protocol field evaluation vào Future Work thay vì báo cáo field accuracy chưa được xác minh.

**Authors' response | English**

> We agree that this is an important limitation. For segmentation, the current evidence does not contain an independent field test set controlled by device, native camera resolution, acquisition environment, and expert or laboratory ground truth. The 1,452-image expanded dataset evaluates a different data composition; it cannot establish multi-camera field generalization because 303 added images lack compatible specimen identifiers and device metadata were not retained as controlled variables. The revised report therefore limits the claim to dataset-composition transfer and adds a controlled field-evaluation protocol to Future Work rather than reporting unverified field accuracy.

**Thay đổi trong Final Report | Tiếng Việt**

> Phần mô tả expanded split được sửa để nêu rõ giới hạn specimen identity. Chapter VI Section 16 và Chapter VII Sections 3.2-3.5 được bổ sung để phân biệt dataset-composition transfer với field generalization và liệt kê các yêu cầu cho field test set độc lập.

**Revision in Final Report | English**

> The expanded-split description was revised to disclose its specimen-identity limitation. Chapter VI Section 16 and Chapter VII Sections 3.2-3.5 were extended to distinguish dataset-composition transfer from field generalization and to specify the requirements for an independent field test set.

**Vị trí chỉnh sửa | Location**

> Chapter IV, Section 3.2.1; Chapter VI, Section 16; Chapter VII, Sections 3.2-3.5; revised Final Report pp. **[fill after recompilation]**.

**Minh chứng | Evidence**

- `shrimp-leakage-aware-segmentation/train_log/seg_yolo11n_new_dataset_mixed_split_baseline_paper_row (1).csv`
- Expanded split counts: 1,149 recognized images plus 303 unmatched images; merged 1,147/144/161 split.
- No controlled segmentation field-image/device manifest exists in the retained artifacts.

### 4.6 Reviewer 1, Question 3: lightweight and mobile configurations

This is also a shared group question. The text below is limited to segmentation.

**Ý kiến gốc | Original feedback**

> “‘Lightweight’ có thực sự đồng nghĩa với hiệu quả trên thiết bị mobile không? Thử nghiệm trên các thiết bị iPhone, Android chưa? Chưa, bổ sung thông tin model cuối cùng (model architecture, model size, number of params., key obtained results) và chạy được trên những thiết bị mobile cấu hình như thế nào.”

**Bản dịch tiếng Anh | English translation**

> Does “lightweight” actually imply efficiency on mobile devices? Has the system been tested on iPhone and Android devices? Add information about the final model architecture, model size, number of parameters, key results, and the mobile configurations on which it can run.

**Phản hồi của nhóm | Tiếng Việt**

> Nhóm đồng ý rằng parameter count thấp không tự động chứng minh hiệu quả trên mobile. Đối với segmentation, YOLO11n-seg có 2.877 triệu tham số nên chỉ được mô tả là compact so với các candidate lớn hơn. Final segmentation checkpoint chưa được export và benchmark end to end trên Android hoặc iPhone. Timing của W1 được đo trên Kaggle GPU: Fourier preprocessing 48.62 ms/ảnh và effective W1 path 87.38 ms/ảnh; các số này không phải mobile latency. Bản sửa đổi tách rõ model complexity khỏi device-level evidence và không suy rộng classification mobile prototype thành segmentation deployment validation.

**Authors' response | English**

> We agree that a low parameter count does not automatically establish mobile efficiency. For segmentation, YOLO11n-seg has 2.877 million parameters and is described only as compact relative to larger candidates. The final segmentation checkpoint was not exported and benchmarked end to end on Android or iPhone. W1 timing was measured on a Kaggle GPU, with 48.62 ms/image for Fourier preprocessing and 87.38 ms/image for the effective W1 path; these values are not mobile latency. The revision separates model complexity from device-level evidence and does not generalize the classification mobile prototype as validation of segmentation deployment.

**Thay đổi trong Final Report | Tiếng Việt**

> Đoạn chọn YOLO11n-seg sau Table 6.9 được bổ sung parameter count và key metrics. Table 6.18 tiếp tục báo cáo riêng Fourier preprocessing và effective path. Chapter VI Sections 14 và 16 cùng Chapter VII Sections 3.2-3.5 được sửa để nêu rõ final segmentation checkpoint chưa được benchmark trên mobile.

**Revision in Final Report | English**

> The YOLO11n-seg selection paragraph following Table 6.9 was extended with parameter count and key metrics. Table 6.18 continues to report Fourier preprocessing separately from the effective path. Chapter VI Sections 14 and 16 and Chapter VII Sections 3.2-3.5 were revised to state explicitly that the final segmentation checkpoint was not benchmarked on a mobile device.

**Vị trí chỉnh sửa | Location**

> Chapter VI, Sections 10.1, 12.4/recorded timing, 14, and 16; Chapter VII, Sections 3.2-3.5; revised Final Report pp. **[fill after recompilation]**.

**Minh chứng | Evidence**

- YOLO11n-seg parameter count: 2.877 million.
- `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/41to50/46/46_inference_timing_all_methods_seed42.json`.
- Final Report Table 6.18.
- No Android/iOS segmentation checkpoint benchmark exists in the retained evidence.

## 5. Questions outside this user's answer ownership

The following response-letter entries should remain with the classification owner:

- Reviewer 1, Question 1: ASL-LDAM + SimAM-DCFR ablation evidence.
- Reviewer 2, Question 1: performance drop in the integrated classification regime.

The global font/size consistency response belongs to the report integrator. The segmentation owner should only verify that newly added tables and figures follow the final XeLaTeX style.

## 6. Final implementation order

1. Apply the six factual corrections from `Danh sách chỉnh lý.docx` using the full wording in Section 3.
2. Replace Figure 4.10 with the 8-panel W1 figure; place the 18-panel decomposition in Appendix D.
3. Add the exact spectrum protocol to Chapter IV Section 4.6.
4. Add Chapter VI Section 12.2.1 with the paired metrics table and the two specified figures.
5. Revise the Table 6.9 selection paragraph and Table 6.10 overlap definition.
6. Add the bounded U-Net, field-generalization, and mobile-segmentation limitations.
7. Recompile the report and verify all table/figure cross-references and captions.
8. Fill final report page numbers in each response-letter entry only after pagination is stable.
9. Integrate the bilingual response entries into the response-letter template.
10. Perform a final evidence audit: every quantitative claim must resolve to a retained CSV, JSON, executed notebook output, or report table.
