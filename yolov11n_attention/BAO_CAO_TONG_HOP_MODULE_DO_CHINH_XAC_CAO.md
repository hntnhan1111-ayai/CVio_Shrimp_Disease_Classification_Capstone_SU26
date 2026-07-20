# Báo cáo tổng hợp các hướng attention có độ chính xác cao — YOLO11n-Seg

**Thời điểm tổng hợp:** 20/07/2026  
**Phạm vi:** toàn bộ thư mục thử nghiệm `yolov11n_attention`, bao gồm các notebook đã có output, CSV/Markdown tổng hợp, YAML/module tự định nghĩa và các hướng cải tiến tiếp theo. Thư mục `ultralytics/` là mã framework được vendoring; không được xem là một hướng thí nghiệm độc lập.

## 1. Cách đọc đúng kết quả

Đây là bài toán **instance segmentation** BG/WSSV, vì vậy “độ chính xác” trong báo cáo được hiểu là **mask mAP**, không phải classification accuracy.

- **Full test mask mAP50:** mAP mask trên đủ 129 ảnh test, gồm 41 ảnh healthy.
- **Disease/labeled-only mask mAP50:** mAP mask trên 88 ảnh có nhãn bệnh; phù hợp để đo chất lượng phát hiện tổn thương.
- **mAP50-95:** chất lượng mask ở dải IoU nghiêm ngặt hơn; giá trị cao hơn nghĩa là đường biên/chất lượng vùng mask tốt hơn.
- **Healthy FP:** tỷ lệ 41 ảnh healthy bị dự đoán có mask bệnh; càng thấp càng tốt.
- **HA (healthy-aware score):**

  ```text
  HA = disease_mAP50
       - 0.05 × mask_count_MAE
       - 0.15 × disease_miss_rate
       - 0.10 × healthy_FP_rate
  ```

Clean baseline dùng để tham chiếu có full mAP50 = **0.441**, disease mAP50 = **0.466**, disease mAP50-95 ≈ **0.169**, healthy FP = **36.6%**, HA = **0.393**. Các run dùng strong augmentation chỉ được so sánh trực tiếp khi có baseline strong cùng protocol.

## 2. Kết luận ngắn gọn

Không có một module tối ưu tuyệt đối cho mọi KPI. Các cấu hình tốt nhất đã có output là:

1. **Tốt nhất cho disease mAP50, mAP50-95 và HA: `SimAM + CA` với strong augmentation.** Disease mAP50 = **0.5507**, disease mAP50-95 = **0.1964**, HA = **0.4808** — đều là mức cao nhất trong các kết quả định lượng đã kiểm tra. Full mAP50 = 0.4971 và healthy FP = 34.1%.
2. **Tốt nhất cho full-test mAP50: `LKA → SimAM head`.** Full mAP50 = **0.5003**, disease mAP50 = 0.5249, disease mAP50-95 = 0.1812 và HA = 0.4518.
3. **Module attention mới tốt nhất: `DPCA` (Dual-Polarity Contrast Attention).** Disease mAP50 = **0.5225**, healthy FP = **26.8%** và HA = **0.4590**. Đây là ứng viên attention mới đáng ưu tiên xác minh bằng multi-seed.
4. **Tốt nhất trong hướng CoTE refinement: `CoTE + BoundaryLite` P4 strong.** Full/disease mAP50 = **0.4921 / 0.5174**, HA = **0.454**. Đây là đối chứng CoTE mạnh nhất hiện có, nhưng không vượt SimAM+CA strong về mAP50-95 toàn cục.
5. **Tối ưu giảm báo động giả: `SimAM+CA + WIoU v3`.** Healthy FP = **17.1%**, thấp hơn clean baseline 19.5 điểm %, nhưng disease mAP50 giảm còn 0.4539; chỉ nên chọn khi chi phí false positive đặc biệt cao.

## 3. Bảng xếp hạng các hướng đã có kết quả đáng tin

| Hướng | Cấu hình/module tốt nhất | Full mAP50 | Disease mAP50 | Disease mAP50-95 | Healthy FP | HA | Kết luận |
|---|---|---:|---:|---:|---:|---:|---|
| Tổ hợp attention + strong augmentation | **SimAM + CA** | 0.4971 | **0.5507** | **0.1964** | 34.1% | **0.4808** | Lựa chọn tốt nhất nếu ưu tiên hiệu năng tổng thể trên ảnh bệnh. Có baseline strong cùng nguồn để đối chiếu. |
| Tinh chỉnh head/attention | **LKA → SimAM head** | **0.5003** | 0.5249 | 0.1812 | 36.6% | 0.4518 | Full-test mAP50 cao nhất; cải thiện chủ yếu nhờ mask và recall trên ảnh bệnh. |
| Attention mới | **DPCA strong** | 0.4880 | 0.5225 | 0.1580 | **26.8%** | 0.4590 | Cân bằng tốt mAP/FP. Kết luận nhân quả còn cần baseline strong và lặp seed. |
| CoTE refinement | **CoTE + BoundaryLite P4 strong** | 0.4921 | 0.5174 | 0.1660 | 34.1% | 0.4540 | Ứng viên CoTE tốt nhất; hợp khi cần một hướng P4-only có kết quả mạnh. |
| Huấn luyện/checkpoint averaging | **SimAM+CA + SWA** | 0.4616 | 0.5093 | 0.1718 | 41.5% | 0.4328 | Tăng disease mAP nhưng healthy FP cao; không vượt SimAM+CA strong gốc. |
| CoTE gốc | **CoTE strong** | 0.4329 | 0.4865 | 0.1360 | 31.7% | 0.4207 | Có tín hiệu cân bằng tốt hơn clean baseline, nhưng thấp hơn các refinement mới. |
| Attention mới, đối chứng gần clean baseline | **ODAA light** | 0.4050 | 0.4660 | — | 29.3% | 0.3980 | Gần giữ disease mAP baseline và giảm FP 7.3 điểm %, nhưng lợi ích HA nhỏ. |
| Single custom attention | **Triplet Attention Segment Head** | 0.4220 | 0.4560 | 0.1460 | 34.1% | 0.3810 | Tốt nhất nhóm custom single-run; vẫn dưới clean baseline. |
| Giảm FP bằng loss | **SimAM+CA + WIoU v3** | 0.4314 | 0.4539 | 0.1473 | **17.1%** | 0.3954 | Phù hợp nếu ưu tiên loại healthy; không phải lựa chọn mAP cao nhất. |

`—` nghĩa là nguồn tổng hợp không dùng giá trị đó để xếp hạng.

## 4. So sánh các ứng viên dẫn đầu

| Cấu hình | Điểm mạnh chính | Điểm đánh đổi | Khuyến nghị dùng |
|---|---|---|---|
| **SimAM + CA strong** | Dẫn đầu disease mAP50, mAP50-95, HA; healthy FP thấp hơn baseline strong. | Là một tổ hợp 2 attention, cần kiểm tra độ ổn định nhiều seed trước khi chốt. | Mô hình ứng viên chính cho chất lượng segmentation tổng thể. |
| **LKA → SimAM head** | Full mAP50 cao nhất và disease miss chỉ 7.95%. | Healthy FP không giảm (36.6%); là kết quả một run. | Khi KPI chính là hiệu năng trên toàn bộ tập test. |
| **DPCA strong** | Cơ chế attention mới rõ ràng cho tổn thương sáng/tối cục bộ; FP 26.8%. | Disease mAP50-95 thấp hơn SimAM+CA strong; protocol strong chưa có đối chứng hoàn toàn tương ứng trong report. | Ưu tiên ablation DPCA-P4 và DPCA-P3/P4/P5. |
| **CoTE + BoundaryLite** | Mạnh nhất trong dòng CoTE refinement; disease mAP50 gần DPCA. | FP 34.1%; không phải top toàn cục về mAP50-95. | Benchmark CoTE khi cần hướng P4-only dễ kiểm soát. |
| **WIoU v3** | Healthy FP thấp nhất trong các cấu hình còn giữ HA xấp xỉ baseline. | Đánh đổi trực tiếp mAP50 và disease recall. | Môi trường vận hành rất nhạy với false alarm. |

## 5. Kết quả theo từng nhánh dự án

### 5.1 Nhánh attention nền tảng (`yolov11n_grouped_attention`)

- Trong các module đơn lẻ, **CA** và **SimAM** là các tín hiệu sớm tốt nhất; các module CBAM/EMA không cho lợi ích ổn định.
- Trong các tổ hợp, **SimAM+CA** là tổ hợp vượt trội. Run lịch sử đạt full mAP50 0.4950; run strong augmentation có export CSV đầy đủ đạt full/disease mAP50 0.4971/0.5507.
- Không nên tiếp tục thêm EMA vào SimAM+CA: `EMA+SimAM+CA` giảm còn full mAP50 0.4046 trong run combined lịch sử.

### 5.2 Nhánh vị trí đặt attention và backbone (`att_backbone`, `att_position_experiments`)

- Các thử nghiệm vị trí P3/head/neck/backbone đã được rà soát trong notebook và archive kết quả.
- Tín hiệu tốt nhất ở backbone chỉ tăng nhẹ so với baseline (SimAM backbone single run có full mAP50 khoảng 0.4489); các tổ hợp backbone/neck khác chưa vượt các ứng viên ở Bảng 3.
- Kết quả cho thấy **thay đổi đúng loại attention và mask head quan trọng hơn chỉ đổi vị trí chèn attention**. Đây là lý do LKA→SimAM, SimAM+CA và DPCA được ưu tiên hơn các biến thể vị trí đơn thuần.

### 5.3 Nhánh custom attention và research modules

- Custom single-run tốt nhất là **Triplet Attention Segment Head**, nhưng dưới baseline về mAP và có disease miss cao hơn.
- CoTE/LPSC/SCSG bản clean không vượt baseline. `CoTE strong` là tín hiệu tốt nhất của bộ research module gốc; `LPSC strong` giảm FP tốt (26.8%) nhưng disease miss lên 21.6%.
- Sáu biến thể P4 strong của CoTE/LPSC cho thấy thêm suppression mạnh thường đổi FP thấp lấy miss cao. `CoTE-BL` có FP 22.0% nhưng miss 38.6%, nên không nên dùng làm model chính.

### 5.4 Nhánh SimAM nhóm B/C/D: loss, mask head và train protocol

- **Nhóm B/loss:** WIoU v3 là phương án low-FP tốt nhất; BoundaryIoU chỉ tăng nhẹ disease mAP50 (0.4699) nhưng giảm full mAP50; Asymmetric Focal Loss là regression rõ rệt.
- **Nhóm C/kiến trúc:** `LKA→SimAM head` là kết quả full mAP50 cao nhất dự án. Bilinear refine giảm mAP; `CA dual` quá bảo thủ; `SimAM backbone + CA head` giảm FP nhưng không đủ mAP.
- **Nhóm D/train protocol:** SWA là cách train tốt nhất của nhóm này. Gradient accumulation cho FP rất thấp (14.6%) nhưng mAP thấp; label smoothing/calibration bị loại do regression.

### 5.5 Nhánh cải tiến CoTE và attention mới

- Trong `improving_labeled_test`, **CoTE+BoundaryLite** đứng đầu: full/disease mAP50 = 0.4921/0.5174.
- Các hướng `hard-negative fine-tune`, `image-level disease gate`, threshold/quality rescoring trong `improving_full_and_labeled_test` là **hậu xử lý hoặc chiến lược huấn luyện, không phải attention module**. Một số notebook có chênh giữa bảng tóm tắt và ô evaluate cuối; vì vậy không dùng để xếp hạng chính thức.
- Trong `new_research_attention`, **DPCA** dẫn đầu rõ ràng. ODAA-light là candidate tốt nhất khi đối chứng gần clean baseline; LACA và SREA chưa đạt yêu cầu về mAP/recall.

## 6. Điều kiện tin cậy và các điểm chưa hoàn tất

1. **Không trộn lẫn các protocol.** Chênh lệch có thể đến từ augmentation, số epoch, checkpoint và threshold, không chỉ từ module. Các kết quả strong không nên dùng để kết luận nguyên nhân nếu thiếu baseline strong matched.
2. **Đa số kết quả là single seed.** Test set chỉ có 129 ảnh/119 instance, vì thế chênh lệch vài phần nghìn đến vài phần trăm chưa đủ để tuyên bố thắng tuyệt đối.
3. **Pipeline `custom_attention_multiseed` chưa có kết quả hợp lệ.** `top4_multiseed_summary.csv` hiện có `n_seeds=0` cho cả Triplet, CESA-Lite, Context-Suppression và SGE-ECA; bảng số liệu cũ trong Markdown không được dùng làm bằng chứng multi-seed.
4. Các notebook nhóm `yolov11n_custom_attention_NhomA`–`NhomE` là pipeline/thiết kế thực nghiệm. Output đã chạy được tổng hợp từ các nhánh `yolov11n_simam_Nhom*`, `improving_*` và `new_research_attention`; các biến thể chỉ có notebook chưa được xếp hạng.

## 7. Đề xuất chốt hướng tiếp theo

Chạy lại tối thiểu 3 seed với cùng split, image size, strong augmentation và threshold cho năm cấu hình sau:

1. Baseline strong.
2. SimAM+CA strong.
3. LKA→SimAM head.
4. DPCA (P4 và P3/P4/P5).
5. CoTE+BoundaryLite P4.

Chốt model theo **mean ± std của disease mAP50, disease mAP50-95, full mAP50, healthy FP và HA**. Với kết quả hiện tại, thứ tự ưu tiên thực nghiệm là: **SimAM+CA strong → LKA→SimAM → DPCA → CoTE+BoundaryLite → WIoU v3 (nếu cần low-FP)**.

## 8. Nguồn số liệu đã đối chiếu

- `yolov11n_simam_NhomA/augmentation/.../reports/summary_all_models.csv`
- `yolov11n_simam_NhomB/result/group_b_vs_baseline_comparison.csv`
- `yolov11n_simam_NhomC/group_c_vs_baseline_comparison.csv`
- `yolov11n_simam_NhomD/group_d_comparison.csv`
- `custom_attention_comparison_vs_baseline.md`
- `custom_attention_research_modules/results/module_comparison_summary.csv`
- `custom_attention_research_modules/results/improvement_results_extracted.csv`
- `improving_labeled_test/result/*.ipynb` (output evaluation cuối)
- `improving_full_and_labeled_test/results/*.ipynb` (đã đánh dấu trạng thái cần xác minh)
- `new_research_attention/attention_module_new_report.md` và `new_research_attention/result/*.ipynb`
- `yolov11n_grouped_attention/*.ipynb`, `att_backbone/*.ipynb`, `att_position_experiments/**/*.ipynb`

