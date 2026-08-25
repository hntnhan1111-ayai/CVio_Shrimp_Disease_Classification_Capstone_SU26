# Final Report Revision Plan: Segmentation Data and Fourier Evidence

Date: 2026-08-11

## 1. Audit boundary and verdict

This audit covers the user's contribution scope:

- the instance-segmentation dataset and annotation protocol;
- grouped-specimen and expanded-dataset splitting;
- segmentation baseline selection;
- image-level Fourier experiments, especially W1;
- Fourier interaction with conventional augmentation and processing order;
- segmentation robustness and inference-cost claims.

The current report is usable, but it is not yet ready for the final council response. Most recorded results are already reported truthfully. The required work is concentrated in four areas:

1. add the newly executed spatial/frequency-domain evidence that answers the augmentation-Fourier question;
2. correct the strong-augmentation contract because `erasing=0.15` was recorded but was not active in the Ultralytics segmentation transform builder;
3. clarify Healthy empty-label supervision and the leakage limit of the expanded mixed split;
4. state openly which requested evidence does not exist, especially U-Net, field-image generalization, and mobile segmentation benchmarks.

No report, response-letter, slide, or XeLaTeX source has been edited by this audit.

## 2. Council-question ownership

| Council item | Ownership for this user | Evidence status | Required action |
|---|---|---|---|
| Reviewer 1: ASL-LDAM + SimAM-DCFR ablation | Not in scope | Classification evidence owned by teammates | Do not answer on behalf of the classification owner |
| Reviewer 1: real-world images from different environments/cameras | Shared; segmentation portion only | Insufficient for a field-generalization claim | State the current limitation; do not present the expanded dataset as a multi-camera field test |
| Reviewer 1: lightweight/mobile configuration | Shared; segmentation portion only | Insufficient for segmentation mobile readiness | Report YOLO11n-seg complexity as compact, but do not claim mobile deployment without device benchmarks |
| Reviewer 2: integrated classification performance drop | Not in scope | Classification evidence owned by teammates | Do not answer on behalf of the classification owner |
| Reviewer 2: why strong augmentation reduces the benefit of W1 and what happens to the spectrum | Direct responsibility | Sufficient for a bounded, evidence-based answer; not sufficient for a causal or multi-seed claim | Add the new paired spectrum analysis and revise the response-letter answer |
| Reviewer 3: add Parameters/Assessment to the comparison slide | Shared | Sufficient for the segmentation rows; raw six-model provenance should be centralized | Add the segmentation trade-off assessment without changing the slide's scientific meaning |
| Reviewer 3: why YOLO11 was selected for segmentation | Direct responsibility | Sufficient in the report table; provenance partially centralized | Use the existing six-model comparison and explicitly state that YOLO11n-seg did not win every metric |
| Reviewer 3: compare segmentation with U-Net if time permits | Direct responsibility | No verified U-Net run | Record as unfinished conditional work/future work; never invent a comparison |
| Council: global font and format consistency | Report-production owner | Requires full-document XeLaTeX/render QA | Supply corrected segmentation content, but leave global typography to the report integrator |

The council letter does not ask a separate annotation question, but annotation, negative-label handling, and split wording must still be corrected because they affect the credibility of the segmentation results.

## 3. Current-report findings and exact amendments

Page references below use the physical PDF page first and the printed report page in parentheses.

| Location | Current status | Required amendment |
|---|---|---|
| PDF 70-72 (printed 61-63), segmentation dataset construction | Mostly correct | Keep BG and WSSV as the two positive mask classes. State explicitly that Healthy images are retained in train/validation/test as empty-label negative samples, not merely as an optional diagnostic subset. |
| PDF 70-71 (printed 61-62), overlap policy | Incomplete | State the actual project policy: separate non-overlapping BG and WSSV regions may coexist in one image; when annotated regions overlap, one deterministic label is retained, with WSSV taking project-specific priority. Do not claim that YOLO cannot represent multiple classes or instances in one image. |
| PDF 71 (printed 62), Table 4.4 | Misleading Healthy row | Replace “may be used only as a negative diagnostic subset” with wording that Healthy images provide negative supervision and are also used for Healthy false-positive diagnostics. |
| PDF 72 (printed 63), dataset inventory | Correct counts, incomplete explanation | Retain 1,149 images and 416 specimens. Add 1,031 masks: 462 BG and 569 WSSV; 220 images are co-infection images. Preserve the limitation that no pathologist review or inter-annotator agreement was recorded. |
| PDF 72 (printed 63), expanded inventory | Counts are correct, leakage boundary too implicit | State that 1,149 convention-matched images use grouped-specimen allocation, while 303 unmatched-name images use image-level stratified random allocation; the merged 1,147/144/161 split is therefore not fully specimen-leakage-free. |
| PDF 72 (printed 63), subsection heading | Numbering error | Change `3.1.1 Instance-Segmentation Dataset Construction...` to the correct hierarchy, most likely `3.2.1`, or promote it to the next top-level subsection consistently. |
| PDF 74 (printed 65), Table 4.6 augmentation policies | One factual error | Remove random erasing as an active component of the strong segmentation policy. Keep a footnote that `erasing=0.15` was present in recorded arguments but was ignored by Ultralytics 8.4.62 `v8_transforms`; separate custom offline erasing experiments are unaffected. |
| PDF 79-80 (printed 70-71), W1 method | Substantively correct | Present the equations unambiguously: `x_low = IFFT(G * FFT(x))`, `x_high = x - x_low`, and `x_W1 = clip(x + alpha*x_high, 0, 255)`, with `sigma=50`, `alpha=0.10`. |
| PDF 79 (printed 70), Figure 4.10 | Useful but insufficient for the council question | Add or replace with the new decomposition figure showing the original, Gaussian low reconstruction, signed high residual, scaled residual, W1 result, spectra, and theoretical filters. Caption “W1 after clipping” as “W1 after clipping and uint8 quantization.” |
| PDF 80 (printed 71), Fourier-augmentation interaction method | Needs reproducibility details | Add the spectrum-analysis protocol: 24 test images, six per Healthy/BG/WSSV/WSSV_BG stratum; 12 matched augmentation realizations; five image paths; 288 paired observations per global metric and 216 lesion observations. |
| PDF 81-82 (printed 72-73), metrics | Largely correct | Keep full-test, labeled-only, Healthy FP, disease miss, count MAE, and HScore distinct. Continue identifying HScore as an internal criterion and distinguish AP confidence (`0.001`) from diagnostic confidence (`0.25`). |
| PDF 119 (printed 110), Table 6.9 | Scientifically adequate | Keep the six-model table. Add an Assessment column only if space permits; otherwise the existing paragraph already gives the necessary trade-off. Ensure the raw six-model output is archived centrally before submission. |
| PDF 119-120 (printed 110-111), Table 6.10 | Result is useful, denominator is unclear | Define “train-test overlap” as the percentage of evaluation specimen groups whose specimen key also appears in training. It is not the fraction of images from an individual specimen that leaked. |
| PDF 123-124 (printed 114-115), Tables 6.15-6.16 | Correct and should remain | Preserve the negative result: W1 is condition-dependent. It improved the strict no-augmentation row but reduced labeled mAP50 under the strongest augmentation row. |
| After Table 6.16 / Figure 6.15 | Major missing evidence | Insert a new “Spectrum interaction analysis” subsection using the paired aggregate table and `049_w1_band_energy_interaction.png`. State that strong augmentation reduced or redistributed most of W1's incremental global high-frequency effect rather than completely canceling it. |
| PDF 125 (printed 116), Table 6.17 order study | Needs a comparability caveat | Keep the trained N=1 order results, but distinguish them from the new image-only counterfactual spectrum analysis. The spectrum experiment did not retrain both full strong-policy orders. |
| PDF 125 (printed 116), Table 6.18 timing | Values can remain with a stronger caveat | Report 48.62 ms/image as the measured Fourier preprocessing component and 87.38 ms/image as the observed end-to-end W1 path. Do not interpret the different YOLO `predict()` components as pure architecture-forward-time differences. |
| PDF 126 (printed 117), corruption results | Correctly separated from W1 | Keep Table 6.19 described as selected attention-based results. Do not relabel it as direct Fourier robustness evidence. |

## 4. Copy-ready answer: augmentation-Fourier spectrum interaction

### Vietnamese

Nhóm cảm ơn góp ý của Cô. Nhóm đã bổ sung phân tích phổ trực tiếp thay vì chỉ suy luận từ mAP. Trong cùng protocol seed 42, cùng YOLO11n-seg và cùng phép chia theo cá thể, W1 làm labeled mask mAP50 tăng từ 0.2354 lên 0.3451 khi tắt augmentation và tắt hidden Albumentations hook, tương đương +10.97 điểm phần trăm. Tuy nhiên, khi sử dụng strong augmentation với hook vẫn tắt, W1 làm labeled mask mAP50 giảm từ 0.5951 xuống 0.5771, tương đương -1.80 điểm phần trăm. Vì vậy, hiệu quả của W1 phụ thuộc vào chế độ augmentation chứ không phải là một cải thiện cố định.

Phân tích bổ sung sử dụng 24 ảnh test, gồm 6 ảnh cho mỗi nhóm Healthy, BG, WSSV và WSSV_BG, với 12 lần sinh augmentation có tham số ngẫu nhiên được ghép cặp. Trên 288 cặp quan sát, W1 khi áp dụng riêng làm tỷ trọng năng lượng dải cao tăng trung bình 0.000670, trong khi phần tăng bổ sung của W1 sau strong augmentation chỉ còn 0.000151, tương đương 22.5% hiệu ứng ban đầu. Mức tăng Laplacian variance còn 21.1%, mean gradient còn 56.1%, trong khi gradient trong vùng tổn thương còn khoảng 79.0%. Kết quả cho thấy strong augmentation không loại bỏ hoàn toàn W1, nhưng đã chồng lấp hoặc tái phân phối phần lớn nhấn mạnh cao tần toàn cục của W1.

Các phép xoay, dịch chuyển và thay đổi tỷ lệ sử dụng nội suy nên có thể làm suy giảm một phần chi tiết cao tần, đồng thời tạo cấu trúc phổ mới ở biên và vùng padding. HSV làm thay đổi quan hệ cường độ giữa các kênh màu. Trong pipeline huấn luyện đã ghi nhận, W1 được áp dụng trước rồi strong augmentation được áp dụng online; do đó tín hiệu biên được W1 tăng cường tiếp tục bị resampling và biến đổi màu. Mô hình chỉ dùng strong augmentation đã học từ một phân phối đa dạng hơn, nên thêm một high-pass bias cố định có thể trở nên dư thừa hoặc làm nhấn mạnh texture/nhiễu không có ích. Đây là cơ chế phù hợp với số liệu phổ và kết quả seed 42, nhưng chưa phải bằng chứng nhân quả hoặc xác nhận đa seed.

### English

We thank the reviewer for this comment. We added direct spectrum analysis rather than inferring the mechanism from mAP alone. Under the same seed-42 protocol, YOLO11n-seg model, and grouped-specimen split, W1 increased labeled mask mAP50 from 0.2354 to 0.3451 when augmentation and the hidden Albumentations hook were disabled, a gain of 10.97 percentage points. Under strong augmentation with the hook still disabled, W1 reduced labeled mask mAP50 from 0.5951 to 0.5771, a decrease of 1.80 percentage points. W1 is therefore augmentation-regime dependent rather than uniformly beneficial.

The added analysis used 24 test images, with six images from each Healthy, BG, WSSV, and WSSV_BG stratum, and 12 matched random augmentation realizations. Across 288 paired observations, W1 alone increased high-band energy share by 0.000670 on average, whereas the incremental W1 effect after strong augmentation was 0.000151, or 22.5% of the original effect. The retained effects were 21.1% for Laplacian variance, 56.1% for mean gradient, and approximately 79.0% for lesion-region gradient. Strong augmentation therefore did not remove W1 completely, but it overlapped with or redistributed most of W1's global high-frequency emphasis.

Rotation, translation, and scale operations use interpolation, which can attenuate some high-frequency detail while introducing new edge and padding structures. HSV changes the intensity relationships among color channels. In the recorded training pipeline, offline W1 was followed by online strong augmentation, so W1-enhanced edges were subsequently resampled and color-transformed. The strong-augmentation-only model already learned from a broader image distribution; adding a fixed high-pass bias could therefore become redundant or overemphasize unhelpful texture or noise. This mechanism is consistent with the measured spectra and seed-42 model results, but it is not presented as causal proof or multi-seed confirmation.

## 5. Copy-ready answer: why YOLO11n-seg was selected

### Vietnamese

YOLO11n-seg được chọn làm kiến trúc nền cho các thí nghiệm segmentation tiếp theo vì đạt trade-off tổng thể tốt nhất trong bảng so sánh sáu mô hình trên cùng phép chia grouped-specimen, không phải vì đứng đầu ở mọi metric. Mô hình có 2.877 triệu tham số, đạt full-test mask mAP50 0.4815, labeled-only mask mAP50 0.5120, labeled-only mask mAP50-95 0.1678, Healthy false-positive rate 0.2439 và HScore 0.4596. Trong panel này, YOLO11n-seg đạt full-test mAP50, labeled-only mAP50 và HScore cao nhất. YOLO26s đạt mAP50-95 cao hơn, còn YOLO26n có Healthy false-positive rate thấp hơn. Vì vậy, kết luận chính xác là YOLO11n-seg cung cấp sự cân bằng tốt nhất giữa localization performance, khả năng kiểm soát false positive và độ phức tạp mô hình cho chuỗi thí nghiệm có kiểm soát của dự án.

### English

YOLO11n-seg was retained for subsequent segmentation experiments because it provided the best overall trade-off in the six-model grouped-specimen comparison, not because it dominated every metric. With 2.877 million parameters, it recorded full-test mask mAP50 of 0.4815, labeled-only mask mAP50 of 0.5120, labeled-only mask mAP50-95 of 0.1678, a Healthy false-positive rate of 0.2439, and HScore of 0.4596. It achieved the highest full-test mAP50, labeled-only mAP50, and HScore in this panel. YOLO26s achieved higher mAP50-95, while YOLO26n produced a lower Healthy false-positive rate. YOLO11n-seg was therefore selected as the best balance of localization performance, false-positive control, and model complexity for the project's controlled experiments.

## 6. Copy-ready answer: U-Net comparison

### Vietnamese

Nhóm ghi nhận đây là yêu cầu có điều kiện. Trong phạm vi bằng chứng hiện tại, nhóm chưa hoàn thành một thí nghiệm U-Net được kiểm chứng trên cùng split và training budget, vì vậy báo cáo không tuyên bố YOLO tốt hơn U-Net. Bài toán hiện tại được xây dựng dưới dạng instance segmentation hai lớp với ảnh Healthy không có positive mask, trong khi U-Net tiêu chuẩn là semantic segmentation. Một so sánh công bằng cần rasterize cùng polygon BG/WSSV theo cùng grouped-specimen manifest, huấn luyện với ngân sách tương đương và báo cáo Dice/IoU/mIoU chung. Native instance-level mAP của YOLO không thể được so trực tiếp với semantic Dice của U-Net nếu chưa có protocol chuyển đổi instance tương đương. Nội dung này nên được ghi là limitation và future work nếu không kịp hoàn thành thí nghiệm trước khi khóa báo cáo.

### English

We acknowledge that this was a conditional request. No verified U-Net experiment has been completed using the same split and training budget, so the report does not claim that YOLO outperforms U-Net. The current task is two-class instance segmentation with Healthy images containing no positive masks, whereas a standard U-Net is a semantic-segmentation model. A fair comparison would rasterize the same BG/WSSV polygons under the same grouped-specimen manifest, use a comparable training budget, and report shared Dice/IoU/mIoU metrics. Native YOLO instance-level mAP should not be compared directly with U-Net semantic Dice without an equivalent instance-conversion protocol. Unless this experiment is completed before the report is frozen, it should be recorded as a limitation and future work.

## 7. Shared-question response boundaries

### Real-world multi-camera evaluation

The expanded 1,452-image dataset is not evidence of controlled real-world multi-camera generalization. Device identity, camera resolution, acquisition environment, specimen identity for 303 unmatched images, and independent expert/laboratory ground truth were not recorded as controlled evaluation factors. The segmentation response should say that the expanded set is an additional dataset-composition transfer assessment only. A field-generalization claim requires a separately collected and annotated test set with device and acquisition metadata.

### Lightweight/mobile segmentation

The 2.877-million-parameter YOLO11n-seg checkpoint can be described as compact relative to larger candidates. This is not proof of mobile readiness. The current W1 timing was collected on a Kaggle GPU and includes 48.62 ms/image of Fourier preprocessing; it is not an Android or iPhone benchmark. A mobile claim requires an exported segmentation model, target-device RAM/SoC/OS details, end-to-end preprocessing and inference latency, memory, and preferably energy measurements. The classification mobile prototype must not be used as segmentation deployment evidence.

### Parameters/Assessment slide column

The segmentation slide can use the following concise assessment:

| Model | Parameters (M) | Assessment |
|---|---:|---|
| YOLO11n-seg | 2.877 | Selected: highest full/labeled mAP50 and HScore in the grouped panel with low complexity |
| YOLO26s-seg | 11.506 | Highest mAP50-95, but approximately four times the parameters and lower main mAP50/HScore |
| YOLO26n-seg | 3.126 | Lowest Healthy FP rate, but lower full/labeled mAP50 than YOLO11n-seg |
| YOLOv8n-seg | 3.410 | Lower localization metrics and higher Healthy FP than the selected baseline |
| YOLOv8s-seg | 11.821 | Larger without a corresponding gain in the main grouped metrics |
| YOLO11s-seg | 10.113 | Larger and substantially weaker under this recorded training run |

These assessments are configuration-specific, not universal rankings of the architectures.

## 8. Dataset and annotation wording for likely defense questions

### Healthy empty-label images

Healthy images are not discarded and are not “unlabeled data.” They are known negative examples with no disease polygon. Consequently, there is no matched positive object for box and mask regression on those images, while the detection/classification/objectness pathway still receives negative supervision against disease predictions. Their behavior is measured separately through Healthy false-positive rate and false masks per Healthy image.

### Co-infection overlap

The correct statement is not that YOLO segmentation can only contain one class per image. It supports multiple labeled instances and multiple classes in the same image. The project instead adopted a deterministic annotation policy: spatially separate BG and WSSV regions remain separate instances, while genuinely overlapping pixels are assigned one project-defined target, with WSSV priority. This avoids assigning two semantic disease labels to the same annotated pixel in the project's targets. The choice is an annotation rule, not a general YOLO limitation or a biological claim that one disease replaces the other.

### Group leakage

For the original filename convention `<Disease>-<ShrimpID>-img-<ViewNumber>`, the specimen key is `Disease::ShrimpID` because numeric IDs may repeat across disease folders. All views from one key are placed in one partition. The original split contains 905/115/129 images and 331/40/45 specimen groups, with zero train-test and validation-test specimen overlap.

For the expanded assessment, the 1,149 convention-matched images retain grouped allocation and the 303 unmatched-name images use image-level stratified random allocation before the partitions are merged. The merged split contains 1,147/144/161 images. Leakage freedom is guaranteed only for the convention-matched subset.

## 9. Evidence-sufficiency matrix

| Claim/question | Sufficiency | Available evidence | Remaining gap |
|---|---|---|---|
| W1 helps without augmentation but slightly hurts under strong augmentation | Sufficient for seed 42 | Matched Path 15 metrics and report Table 6.16 | Multi-seed confirmation for statistical stability |
| Strong augmentation reduces/redistributes W1's incremental global spectral effect | Sufficient for a mechanistic association | 24 test images, 12 matched realizations, paired spectrum CSVs and figures | Does not prove the spectral change caused the mAP difference |
| W1 construction and low/high components | Sufficient | Direct spatial/frequency decomposition and exact reconstruction audit | Single representative image is illustrative, not population-level |
| Processing order changes the resulting image spectrum | Sufficient for image-level effect | Matched counterfactual paths in notebook 49 | No full matched retraining of both strong-policy orders |
| YOLO11n-seg is the best balanced baseline in the recorded panel | Substantively sufficient | Report Table 6.9 and recorded baseline metrics | Canonical raw CSV for the complete six-model panel is not clearly centralized |
| Original split is specimen leakage-safe | Sufficient | Deterministic grouped split, manifest fingerprint, zero group overlap | Caption must define the overlap denominator |
| Expanded split is fully specimen leakage-safe | Unsupported and must not be claimed | Mixed-split manifest explicitly separates recognized/unmatched data | Specimen identity unavailable for the unmatched subset |
| WSSV-priority overlap rule was the annotation policy | Sufficient as a project-method statement | Team annotation protocol and retained two-class labels | Preserve a representative Roboflow screenshot or annotation-history example if independent auditing is required |
| Annotation protocol is expert-validated | Unsupported | Team annotation protocol and counts are available | No pathologist review or inter-annotator agreement was recorded |
| U-Net is weaker or stronger than YOLO | Unsupported | No verified run | Requires a same-manifest semantic benchmark |
| Segmentation generalizes across real cameras/environments | Unsupported | Expanded dataset transfer results only | Requires independent field acquisition and annotation |
| Segmentation is mobile-ready | Unsupported | Parameter count and Kaggle GPU timing only | Requires export and target-device benchmark |

## 10. Recommended amendment order

1. Correct Methodology facts first: Healthy negative handling, overlap policy, mixed-split limitation, subsection numbering, and inactive erasing.
2. Fix the W1 equations and add the direct decomposition figure in the Fourier Methodology section.
3. Add the paired 24-image spectrum-interaction protocol to Methodology.
4. Insert the paired spectrum table and figure immediately after Table 6.16/Figure 6.15 in Results.
5. Add the seed-42/causality boundary and distinguish image-only order analysis from model retraining.
6. Strengthen the timing caveat and retain Fourier preprocessing time separately.
7. Confirm the YOLO11n selection paragraph and add the slide Assessment column.
8. Add explicit limitations for U-Net, field images, and mobile segmentation.
9. Replace AI-generated example responses in the response letter with the evidence-based text above.
10. Compile the revised report, then fill the response-letter “Revision,” “Location,” and page-number fields from the final PDF rather than from the current draft.

## 11. Canonical evidence to attach or archive

- Current report: `CVio_Codex_Final_Report_Handoff_2026-08-11/CVio_Final_Report_FINAL_CLEAN_2026-08-11.pdf`
- Response letter template: `CVio_Codex_Final_Report_Handoff_2026-08-11/CVio_Response_Letter_Template_Review_Council_2026-08-09 (1).docx`
- Existing evidence-based answer audit: `shrimp-leakage-aware-segmentation/review_council_segmentation_data_fourier_final_answers.md`
- W1/strong-augmentation spectrum notebook: `shrimp-leakage-aware-segmentation/fourier_vis_output/49.ipynb`
- Spectrum summary: `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_w1_strong_aug_spectrum_interaction_summary.md`
- Paired effects: `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_paired_w1_effects_aggregate.csv`
- Strong-policy implementation contract: `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/ultralytics_augmentation_contract.json`
- Spectrum interaction figure: `shrimp-leakage-aware-segmentation/fourier_vis_output/049_w1_strong_aug_spectrum_interaction/049_w1_band_energy_interaction.png`
- Direct W1 decomposition: `shrimp-leakage-aware-segmentation/fourier_vis_output/fourier_visualization_outputs/w1_spatial_frequency_decomposition.png`
- Decomposition audit: `shrimp-leakage-aware-segmentation/fourier_vis_output/fourier_visualization_outputs/w1_spatial_frequency_decomposition_audit.csv`
- Fourier canonical registry: `shrimp-leakage-aware-segmentation/ONLY_fourier/[hand-explanation]final_metrics_and_training_configs_registry.md`
- Matched factorial output: `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/only_fourier_factorial_A_H_highpass_s50_a0p10_summary.csv`
- Inference timing: `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/41to50/46/46_inference_timing_all_methods_seed42.json`
- Expanded baseline row: `shrimp-leakage-aware-segmentation/train_log/seg_yolo11n_new_dataset_mixed_split_baseline_paper_row (1).csv`

Before submission, the complete raw six-model segmentation baseline panel should be added to the canonical evidence area if the executed output still exists. The report table alone is adequate for explanation, but weaker for provenance auditing.
