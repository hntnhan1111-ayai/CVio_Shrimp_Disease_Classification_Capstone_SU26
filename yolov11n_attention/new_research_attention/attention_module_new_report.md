# Báo cáo tổng hợp các module attention mới cho YOLO11n-Seg

## 1. Phạm vi và cách đọc kết quả

Báo cáo tổng hợp bốn module mới được triển khai trong `new_research_attention`: **DPCA**, **LACA**, **ODAA** và **SREA**. Baseline tham chiếu chính là notebook `aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb` (light augmentation, 60 epoch; best epoch 52). Tập test dùng chung gồm 129 ảnh, 119 đối tượng bệnh và 41 ảnh healthy.

Các phép chạy *strong augmentation* của module mới không hoàn toàn đối chứng với baseline light: phần chênh lệch có thể đến từ cả augmentation, số epoch và module. Vì vậy bảng strong dùng để chọn ứng viên; kết luận nhân quả về module cần baseline strong đã chuẩn bị chạy cùng cấu hình.

## 2. Baseline tham chiếu

| Chỉ số | Baseline clean fix-leakage (light) |
|---|---:|
| Full mask mAP50 | 0.441 |
| Full mask mAP50-95 | 0.161 |
| Disease mask mAP50 | 0.466 |
| Disease mask mAP50-95 | 0.169 |
| Healthy false-positive rate | 36.6% (15/41) |
| Healthy-aware score (HA) | 0.393 |

HA là thước đo cân bằng giữa phát hiện bệnh và hạn chế dự đoán nhầm trên ảnh healthy; nó phù hợp với bài toán sàng lọc bệnh tôm hơn mAP đơn thuần.

## 3. Cơ chế và phân loại module mới

| Module | Gate / tín hiệu chính | Nhóm cơ chế | Vị trí | Ý nghĩa |
|---|---|---|---|---|
| **DPCA** | Độ lệch khỏi ngữ cảnh cục bộ; chọn cực dương/cực âm theo từng kênh | Attention tương phản không gian cục bộ, có điều kiện theo kênh | P3/P4/P5 trước Segment | Làm nổi bất thường sáng hoặc tối so với vùng lân cận, phù hợp tổn thương nhỏ/không đồng nhất. |
| **LACA** | Mức đồng thuận của lân cận cục bộ | Attention không gian dựa trên tính nhất quán | Backbone/neck feature | Ưu tiên tín hiệu ổn định trong vùng lân cận; có xu hướng bảo thủ, giảm nhiễu nhưng có thể bỏ sót dấu hiệu yếu. |
| **ODAA** | Bằng chứng theo các hướng ngang, dọc, chéo | Attention định hướng không gian | Backbone/neck feature | Nhấn cấu trúc kéo dài, biên và texture có hướng; nhạy với nhiễu hình học khi augment mạnh. |
| **SREA** | Bằng chứng dư theo nhiều thang đo | Attention không gian đa tỉ lệ | Backbone/neck feature | Phát hiện phần còn lại khác biệt so với nền ở nhiều receptive field; đổi lại có nguy cơ làm mờ chi tiết segmentation. |

DPCA không phải attention kênh thuần (như SE/ECA): gate của nó có kích thước `[B,C,H,W]`. Việc chọn cực dương/cực âm là theo từng kênh, còn quyết định cuối cùng vẫn giữ vị trí không gian. LACA, ODAA và SREA chủ yếu là attention không gian; ODAA bổ sung trục hướng, SREA bổ sung trục tỉ lệ.

## 4. Kết quả các module mới

### 4.1 Light augmentation — đối chứng gần hơn baseline

| Model | Full mAP50 | Disease mAP50 | Healthy FP | HA | Nhận xét |
|---|---:|---:|---:|---:|---|
| Baseline clean | 0.441 | 0.466 | 36.6% | 0.393 | Mốc tham chiếu |
| DPCA | 0.328 | 0.373 | 46.3% | 0.284 | Chưa ổn định khi train light. |
| LACA | 0.353 | 0.372 | 31.7% | 0.297 | Giảm FP nhưng mất nhiều recall. |
| ODAA | 0.405 | 0.466 | 29.3% | **0.398** | Gần như giữ disease mAP, đồng thời giảm FP 7.3 điểm %. |
| SREA | 0.342 | 0.386 | 29.3% | 0.288 | Giảm FP nhưng giảm recall mạnh. |

ODAA-light là tín hiệu đáng chú ý nhất trong đối chứng gần baseline: HA tăng nhẹ 0.004, chủ yếu nhờ healthy FP giảm từ 36.6% xuống 29.3%, trong khi disease mAP50 gần như không đổi.

### 4.2 Strong augmentation — chọn ứng viên để ablation tiếp

| Model | Full mAP50 | Full mAP50-95 | Disease mAP50 | Disease mAP50-95 | Healthy FP | HA |
|---|---:|---:|---:|---:|---:|---:|
| DPCA | 0.488 | 0.148 | **0.522** | 0.158 | **26.8%** | **0.459** |
| LACA | 0.383 | 0.119 | 0.431 | 0.134 | 34.1% | 0.370 |
| ODAA | 0.385 | 0.120 | 0.438 | 0.137 | 48.8% | 0.352 |
| SREA | 0.403 | 0.118 | 0.448 | 0.131 | 39.0% | 0.375 |

Trong nhóm này DPCA là ứng viên rõ ràng: disease mAP50 đạt 0.522, healthy FP 26.8% và HA 0.459. So với baseline light, các chênh lệch quan sát là +0.056 disease mAP50, -9.8 điểm % FP và +0.066 HA. Tuy nhiên mAP50-95 vẫn thấp hơn baseline 0.011, nên DPCA hiện giúp ưu tiên phát hiện/loại healthy hơn là làm sắc nét biên mask.

## 5. DPCA so với CoTE + BoundaryLite

CoTE + BoundaryLite là đối chứng mạnh hiện có: full mAP50 0.492, disease mAP50 0.517, healthy FP 34.1%, HA 0.454 và disease mAP50-95 0.166.

| Phương diện | DPCA | CoTE + BoundaryLite |
|---|---|---|
| Bản chất | Attention tương phản cục bộ, có chọn cực sáng/tối theo kênh | Attention lai channel–spatial–cross-dimension; BoundaryLite là loss phụ, không phải attention |
| Gate | `[B,C,H,W]`, cục bộ, per-channel | Đồng thuận TripletLite + ECA + disagreement smoothing |
| Vị trí | P3/P4/P5 | P4 |
| Tác động chính | Nâng recall bệnh và giảm FP healthy | Tối ưu chất lượng mask/biên, giữ cân bằng detection–segmentation |
| Disease mAP50 | **0.522** | 0.517 |
| Disease mAP50-95 | 0.158 | **0.166** |
| Healthy FP | **26.8%** | 34.1% |
| HA | **0.459** | 0.454 |

Kết luận thực dụng: chọn **DPCA** nếu hệ thống ưu tiên không bỏ sót bệnh và hạn chế báo động giả trên healthy; chọn **CoTE + BoundaryLite** nếu cần mask chính xác hơn ở các ngưỡng IoU cao hoặc cần đường biên để đo diện tích tổn thương. Chênh HA chỉ 0.005 nên chưa nên tuyên bố hơn hẳn khi chưa chạy baseline strong và seed lặp.

## 6. Vị trí các kết quả khác trong hệ thống

Trong nhóm `improving_labeled_test`, CoTE P4 refinement (HA 0.426), CoTE ECA Rescue (0.407), CoTE residual calibration (0.399) và CoTE BCE-Dice P4 (0.386) đứng sau DPCA/BoundaryLite. Một số phương án `improving_full_and_labeled_test` là kỹ thuật huấn luyện hoặc hậu xử lý (hard-negative fine-tune, image-level disease gate), không phải attention; các notebook có chênh giữa bảng tóm tắt và ô evaluate cuối. Vì thế chúng chỉ được đưa vào workbook ở trạng thái **cần xác minh**, không dùng làm xếp hạng định lượng chính thức.

## 7. Kết luận và đề xuất thực nghiệm tiếp theo

1. **DPCA** là module attention mới tốt nhất theo HA ở kết quả hiện có (0.459). Nó đáng được ưu tiên ablation vì cơ chế có động cơ rõ ràng với tổn thương cục bộ.
2. **ODAA-light** là module mới có đối chứng gần baseline tốt nhất; cần thử ODAA P4-only hoặc giảm độ mạnh augment hình học để kiểm tra độ nhạy hướng.
3. **CoTE + BoundaryLite** là đối chứng tốt nhất về mask chất lượng cao; dùng làm benchmark ngoài nhóm module mới.
4. Chạy tối thiểu năm cấu hình cùng strong augmentation và seed: baseline, DPCA-P4, DPCA-P3/P4/P5, CoTE-P4, CoTE-P4+BoundaryLite. Báo cáo mean ± std của mAP50, mAP50-95, healthy FP và HA.
5. Giữ threshold/evaluation protocol cố định và xác minh lại các notebook hậu xử lý trước khi đưa vào bảng xếp hạng cuối.

## 8. Nguồn số liệu

- `yolov11n_attention/new_research_attention/result/*.ipynb`
- `yolov11n_attention/yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb`
- `yolov11n_attention/improving_labeled_test/result/*.ipynb`
- `yolov11n_attention/improving_full_and_labeled_test/results/*.ipynb`

