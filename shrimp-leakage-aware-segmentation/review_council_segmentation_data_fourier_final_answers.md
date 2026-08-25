# Final Evidence-Based Responses: Segmentation, Data, and Fourier Transform

## 1. Scope and evidence status

This document finalizes the responses to review-council questions that concern the segmentation dataset, split protocol, YOLO11n-seg selection, Fourier preprocessing, and the interaction between W1 and strong augmentation. The conclusions below are based on recorded training rows and the newly executed spectrum-analysis artifacts, rather than on the earlier AI-generated draft answers.

The new spectrum study used 24 test images sampled with seed 42: six Healthy, six BG, six WSSV, and six WSSV_BG images. Each image was evaluated under 12 matched augmentation realizations and five processing paths: original, W1 only, strong augmentation only, W1 followed by strong augmentation, and strong augmentation followed by W1. This produced 288 paired observations per metric, except lesion-region metrics, which used 216 observations because healthy images have no lesion mask.

Evidence status:

| Evidence | Status | Interpretation boundary |
|---|---|---|
| `49.ipynb` | All 11 code cells executed; no recorded runtime error | Canonical executed notebook for the paired spectrum analysis |
| `049_*.csv` and figures | Present and internally consistent | Primary quantitative and visual evidence for the spectrum response |
| `fourier_vis.ipynb` in Downloads | Malformed/truncated JSON near byte 41.4 MB | Do not archive as the canonical notebook; re-export it from Kaggle |
| `w1_frequency_band_energy.csv` and W1 figure | Present | Valid single-image illustration, not population-level evidence |
| Training comparisons | Recorded at seed 42 | Show an observed interaction, not multi-seed statistical confirmation |

## 2. Council question: Why does W1 help without augmentation but hurt with strong augmentation?

### Final answer for the response letter

Trong thí nghiệm seed 42 sử dụng cùng YOLO11n-seg và cùng phép chia theo cá thể, W1 không tạo ra một mức cải thiện cố định trong mọi chế độ huấn luyện. Khi tắt augmentation và tắt hidden Albumentations hook, W1 làm labeled mask mAP50 tăng từ 0.2354 lên 0.3451, tương đương +0.1097 hay +10.97 điểm phần trăm. Tuy nhiên, trong chế độ strong augmentation với hook vẫn tắt, thêm W1 làm labeled mask mAP50 giảm từ 0.5951 xuống 0.5771, tương đương -0.0180 hay -1.80 điểm phần trăm. Difference-in-differences giữa hai chế độ là -0.1277. Vì vậy, kết luận đúng là hiệu quả của W1 phụ thuộc vào augmentation regime; không nên kết luận rằng Fourier luôn cải thiện mô hình.

Phân tích phổ mới cho thấy strong augmentation không triệt tiêu hoàn toàn W1, nhưng làm giảm đáng kể phần thay đổi phổ bổ sung do W1 tạo ra. Trên 288 cặp quan sát, W1 khi đứng một mình làm high-frequency energy share tăng trung bình 0.000670, trong khi hiệu ứng bổ sung của W1 sau strong augmentation chỉ còn 0.000151, tương đương khoảng 22.5% mức ban đầu. Tương tự, mức tăng Laplacian variance chỉ còn khoảng 21.1%, và mức tăng mean gradient còn khoảng 56.1%. Mức tăng gradient trong vùng tổn thương vẫn giữ khoảng 79.0%, cho thấy W1 vẫn nhấn mạnh biên tổn thương nhưng phần lớn thay đổi phổ toàn cục của nó đã bị chồng lấp hoặc tái phân phối bởi augmentation.

Strong augmentation trong thí nghiệm đã tự tạo ra biến thiên lớn về màu sắc, tư thế, tỷ lệ và vị trí. Các phép affine có nội suy có thể làm suy giảm một phần chi tiết cao tần, trong khi xoay, dịch chuyển và vùng padding lại tạo ra các cấu trúc phổ có hướng. HSV thay đổi tương phản giữa các kênh màu. Trong pipeline huấn luyện đã ghi nhận, W1 được áp dụng offline trước, sau đó strong augmentation được áp dụng online. Vì vậy, tín hiệu biên do W1 tăng cường tiếp tục bị resampling và thay đổi màu. Mô hình strong-augmentation-only đã học từ một phân phối ảnh đa dạng hơn; thêm một high-pass bias cố định có thể trở nên dư thừa, làm tăng nhấn mạnh texture hoặc nhiễu không liên quan, và giảm sự cân bằng với thông tin màu hoặc tần số thấp. Đây là cơ chế phù hợp với số liệu phổ, nhưng chưa phải bằng chứng nhân quả trực tiếp cho mức giảm mAP.

### Recorded model results

| Training regime | No Fourier labeled mAP50 | W1 labeled mAP50 | W1 delta | No Fourier healthy FP | W1 healthy FP |
|---|---:|---:|---:|---:|---:|
| No augmentation, hook off | 0.235374 | 0.345105 | +0.109731 | 0.390244 | 0.634146 |
| Strong augmentation, hook off | 0.595095 | 0.577126 | -0.017969 | 0.341463 | 0.317073 |

W1 therefore improved disease-image mAP strongly in the bare regime but also raised the healthy false-positive rate by 0.243902. Under strong augmentation, W1 slightly reduced healthy FP by 0.024390 while reducing labeled mAP50. The result is a trade-off, not a universally beneficial Fourier effect. The current best labeled mAP50 among these matched baseline-architecture modes is strong augmentation without W1.

## 3. Council question: What happens to the image spectrum?

### W1 operation

For each channel, W1 reconstructs a Gaussian low-pass image and adds a fraction of the residual back to the original image:

`x_W1 = clip(x + alpha * (x - x_low), 0, 255)`

The selected setting is `sigma=50` and `alpha=0.10`. In frequency-domain terms, the amplitude gain approaches 1 near the low-frequency center and approaches `1 + alpha = 1.10` at high frequencies. W1 is therefore a mild high-frequency boost, not a binary high-pass image that discards low-frequency content.

### Single-image visualization

For the representative WSSV image in the visualization notebook, W1 changed the normalized band-energy distribution as follows:

| Frequency band | Original energy | W1 energy | Absolute energy gain |
|---|---:|---:|---:|
| Low, radius 0.00-0.15 | 99.8981% | 99.8734% | -0.0328 dB |
| Mid, radius 0.15-0.40 | 0.0768% | 0.0951% | +0.8992 dB |
| High, radius 0.40-1.00 | 0.0251% | 0.0314% | +0.9409 dB |

This figure makes the intended W1 behavior visible: the spatial image remains recognizable, edge and texture differences become visible in the absolute-difference map, and radial spectral power rises progressively away from the low-frequency center. These values illustrate one image only.

### Paired multi-image evidence

| Metric | W1 effect without augmentation | Incremental W1 effect after matched strong augmentation | Retained effect |
|---|---:|---:|---:|
| Low-frequency energy share | -0.001158 | -0.000434 | 37.5% |
| Mid-frequency energy share | +0.000488 | +0.000283 | 58.0% |
| High-frequency energy share | +0.000670 | +0.000151 | 22.5% |
| Mean gradient | +0.010979 | +0.006157 | 56.1% |
| Laplacian variance | +0.003127 | +0.000660 | 21.1% |
| Lesion-region gradient | +0.012104 | +0.009561 | 79.0% |
| Clipping fraction | +0.002381 | +0.000264 | 11.1% |

The safest interpretation is that strong augmentation makes W1's global high-frequency redistribution substantially smaller relative to the already augmented image. It does not remove all local lesion-edge enhancement. This supports a redundancy/overlap explanation, while the model metrics indicate that the remaining fixed emphasis was not beneficial to seed-42 mAP under the strong policy.

### Direct low/high decomposition verification

The subsequently executed decomposition export provides a direct check of the W1 construction on the representative WSSV image. It separately records the original image, Gaussian low-frequency reconstruction, signed high-frequency residual, scaled residual, and W1 result in both the spatial and frequency domains.

| Decomposition quantity | Recorded value |
|---|---:|
| Maximum error in `x = x_low + x_high` | 0.0 |
| Maximum error in `x_W1 = x + alpha * x_high` before clipping | 0.0 |
| High-residual spatial mean | approximately 0 (`-1.42e-6`) |
| High-residual spatial standard deviation | 6.3234 |
| Scaled-residual spatial standard deviation, `alpha=0.1` | 0.6323 |
| Scaled/original residual spectral-energy ratio | 0.0100 |
| Values outside `[0,255]` before clipping | 0.0193% |

The energy ratio is an internal consistency check: multiplying residual amplitude by `0.1` should multiply spectral power by `0.1^2 = 0.01`, which is exactly what the audit records. Before clipping and integer conversion, W1 changed the representative image's low/mid/high energy shares by `-0.000191`, `+0.000145`, and `+0.000046`, respectively. Thus, W1 preserved almost all low-frequency content while transferring a small relative share toward the mid- and high-frequency bands.

Three interpretation cautions apply. First, the exported panel titled “W1 after clipping” is the actual notebook output after both clipping and conversion to `uint8`. Its mean intensity is about `0.494` lower than the pre-clipping floating-point result, even though only `0.0193%` of values required clipping; most of that mean shift is therefore attributable to integer truncation rather than saturation. The defensible caption is “W1 after clipping and uint8 quantization.” Second, the reported low/mid/high bands are fixed radial bins (`0-0.15`, `0.15-0.40`, and `0.40-1.00`), whereas the Gaussian cutoff is controlled by `sigma=50`. Consequently, energy classified in the broad `0-0.15` reporting band can still belong to the Gaussian high-pass residual. The band labels describe measurement bins, not exact pass/stop regions of the Gaussian filter. Third, the decomposition audit averages spectral power across the three RGB channels, while the earlier `w1_frequency_band_energy.csv` first converts the image to grayscale luminance. Their numerical band shares are therefore not expected to be identical and should not be combined in one quantitative table without stating the measurement definition.

## 4. Council question: Does processing order matter?

The recorded Path 15 training order was **offline W1 -> online strong augmentation**. The spectrum notebook also generated the counterfactual **strong augmentation -> W1** using the same image and matched random augmentation parameters.

Relative to W1 -> strong augmentation, placing W1 last produced small average increases of `+0.000020` in high-frequency energy share, `+0.001052` in mean gradient, and `+0.000857` in clipping fraction. This is expected because the final W1 operation is no longer followed by affine interpolation. However, notebook 49 is an image-distribution analysis and did not retrain the model under both orders. It proves that order changes the final pixels and spectrum; it does not prove which order has better model accuracy under the complete strong policy.

For the report, write: “Processing order measurably changes the resulting spectrum. Applying W1 last preserves slightly more high-frequency and gradient emphasis, but it also increases clipping. A model-level preference between the two complete strong-policy orders was not established by this visualization experiment.”

## 5. Strong-augmentation contract correction

The recorded strong policy under Ultralytics 8.4.62 was:

| Component | Recorded value | Active in the segmentation replica |
|---|---:|---|
| HSV hue / saturation / value | 0.05 / 0.50 / 0.40 | Yes |
| Rotation | 10 degrees | Yes |
| Translation | 0.10 | Yes |
| Scale | 0.50 | Yes |
| Horizontal / vertical flip | 0.50 / 0.30 | Yes |
| Mosaic, MixUp, CutMix, CopyPaste | 0 | Disabled |
| Shear, perspective | 0 | Disabled |
| Hidden Albumentations hook | Off | Disabled |
| Erasing | Train argument recorded as 0.15 | Not used by the Ultralytics segmentation transform builder |

The report must not describe `erasing=0.15` as an active operation in this strong YOLO segmentation pipeline. In Ultralytics 8.4.62, erasing is used by the classification transform builder, not the standard `v8_transforms` segmentation builder. This does not invalidate the separate N=1 offline erasing experiments, because those notebooks explicitly implemented erasing outside the YOLO builder.

## 6. Council question: Why was YOLO11n-seg selected?

### Final answer for the response letter

YOLO11n-seg was selected from the grouped-split baseline comparison because it provided the best overall balance for the project's main objective rather than winning every individual metric. It has 2.877 million parameters and recorded full-test mask mAP50 of 0.4815, labeled-only mask mAP50 of 0.5120, labeled-only mask mAP50-95 of 0.1678, healthy false-positive rate of 0.2439, and healthy-aware test score of 0.4596. In the six-model baseline panel reported in the thesis, it achieved the highest full-test mAP50, labeled-only mAP50, and healthy-aware score. YOLO26s achieved a better mAP50-95 and YOLO26n achieved a lower healthy false-positive rate, so YOLO11n-seg should be described as the best balanced baseline for subsequent controlled experiments, not as uniformly superior on all metrics.

An evidence caveat should be retained internally: the final report contains the six-model table, but the canonical raw CSV for that complete model-sweep panel is not as clearly centralized as the Fourier experiment rows. Archive that raw output before final submission if it is still available.

## 7. Council questions about segmentation data and leakage

### Original segmentation dataset

| Item | Count |
|---|---:|
| Images | 1,149 |
| Specimens | 416 |
| Labeled diseased images | 746 |
| Healthy empty-label images | 403 |
| Mask instances | 1,031 |
| BG masks | 462 |
| WSSV masks | 569 |
| Co-infection images | 220 |

The final stratified grouped-specimen split contains 905/115/129 train/validation/test images and 331/40/45 specimen groups. The specimen key is `Disease::ShrimpID` because numeric IDs may be reused across disease folders. All images of one specimen are assigned to one partition. The recorded overlap is zero between train, validation, and test specimen groups.

The “leak” value in a split-policy comparison is the proportion of evaluation specimen groups that also occur in the training partition, not the percentage of images from each individual specimen that leaked. The exact denominator must be stated in the table caption if the report includes a percentage.

### Expanded dataset

The expanded dataset is an additional transfer assessment, not a replacement for the primary grouped dataset. It contains 1,452 images: 1,149 convention-matched images that can use grouped splitting and 303 unmatched-name images that use stratified random splitting. The merged split contains 1,147/144/161 train/validation/test images. Because specimen identity is unavailable for the unmatched subset, the complete expanded split cannot be described as specimen-leakage-free. The accurate wording is: “group leakage was prevented for convention-matched images; unmatched images were stratified at image level.”

### Healthy images with empty labels

Healthy images are valid negative examples. With no positive target instance, positive box and mask terms do not receive a matched target contribution, while the detection/classification branch still receives negative supervision against disease predictions. Their importance is therefore evaluated explicitly through healthy false-positive rate and false masks per healthy image. They should not be removed from segmentation training merely because their label files contain no polygons.

## 8. Council question: Why not compare with U-Net?

No verified U-Net experiment has been completed in the current evidence set. The correct response is therefore to acknowledge this as a scope limitation, not to imply that YOLO was empirically superior to U-Net.

Suggested response: “The implemented task was formulated as two-class instance segmentation with healthy negative images, so the project prioritized YOLO-seg models that directly produce instance masks and confidence-scored detections. A fair U-Net comparison would require rasterizing the same grouped-split polygons into semantic masks and evaluating both systems with a shared semantic metric such as Dice and IoU. Native YOLO instance mAP and U-Net semantic Dice are not directly interchangeable. This benchmark is proposed as future work unless it can be completed under the same split and training budget.”

## 9. Questions about field generalization and mobile deployment

The current expanded-dataset experiment does not establish real-world multi-camera generalization because device identity, acquisition conditions, and a separately annotated field test set were not recorded as controlled factors. It only shows behavior under a different data composition.

Likewise, YOLO11n-seg having 2.877 million parameters supports describing it as compact, but does not establish mobile readiness. Mobile deployment claims require exported-model benchmarks on the target device, including latency, memory, model size, and preferably energy use. The current Android prototype concerns classification and should not be presented as validation of segmentation deployment.

## 10. Required report and slide changes

1. Add `w1_defense_spatial_frequency_comparison.png` to show original/W1 images, absolute spatial difference, spectra, theoretical gain, spectral change, and radial power.
2. Add `049_w1_band_energy_interaction.png` with a table containing the paired aggregate values from Section 3.
3. State explicitly that the performance interaction is a seed-42 observation; avoid “proven” or “statistically significant.”
4. Replace “Fourier improves segmentation” with “W1 is regime-dependent: it improves the bare no-augmentation setting but slightly reduces mAP under the strongest augmentation baseline.”
5. Identify strong augmentation without W1 as the current best labeled-mAP50 baseline among the matched Path 15 modes.
6. Correct the strong-policy description so that erasing is not listed as an active Ultralytics segmentation transform.
7. Explain that the counterfactual order analysis measures pixels and spectra but does not include matched retraining of the complete strong policy.
8. Preserve the healthy-FP trade-off: the no-augmentation W1 gain was accompanied by a large increase in false positives on healthy images.
9. Re-export `fourier_vis.ipynb` from Kaggle because the downloaded copy is malformed; keep the generated figure and CSV as evidence in the meantime.
10. Do not claim U-Net superiority/inferiority, full leakage freedom for the expanded dataset, field generalization, or mobile segmentation deployment without the missing experiments.

## 11. Additional experiments ranked by defense value

| Priority | Experiment | What it would resolve |
|---:|---|---|
| 1 | Multi-seed 2x2 confirmation: no/strong augmentation x no W1/W1 | Whether the observed negative interaction generalizes beyond seed 42 |
| 2 | Full strong-policy order retraining: W1 -> augmentation versus augmentation -> W1 | Whether the small spectrum-order difference changes model performance |
| 3 | Shared semantic U-Net benchmark on the same grouped manifest | The architecture-comparison question using compatible Dice/IoU metrics |
| 4 | Dataset-integrity and acquisition audit for expanded data | Exact source/device coverage and leakage limits |
| 5 | ONNX/TensorRT or TFLite benchmark on the intended edge device | Whether the segmentation pipeline is practically deployable |

## 12. Canonical evidence files

- Spectrum experiment notebook: `C:/Users/Admin/Downloads/49.ipynb`
- Paired spectrum summary: `fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_paired_w1_effects_aggregate.csv`
- Raw paired observations: `fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_paired_w1_effects.csv`
- Training interaction rows: `fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_recorded_training_interaction_metrics.csv`
- Augmentation contract: `fourier_vis_output/049_w1_strong_aug_spectrum_interaction/ultralytics_augmentation_contract.json`
- W1 single-image band table: `fourier_vis_output/fourier_visualization_outputs/fourier_visualization_outputs/w1_frequency_band_energy.csv`
- W1 spatial/frequency figure: `fourier_vis_output/fourier_visualization_outputs/fourier_visualization_outputs/w1_defense_spatial_frequency_comparison.png`
- Interaction figure: `fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_w1_band_energy_interaction.png`
- Main metrics registry: `ONLY_fourier/[hand-explanation]final_metrics_and_training_configs_registry.md`
