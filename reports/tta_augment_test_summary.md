# Tổng hợp kết quả TTA và Augmented Test

Nguồn: 8 notebook trong `tta_inference.zip` và `augment_test_results.zip`.

## 1. TTA fused trên test gốc (129 ảnh)

| Hạng theo Mask mAP50 | Mô hình | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | SimAM-CA Strong | 0.7525 | 0.5142 | **0.6516** | **0.3307** | 0.6730 | **0.4704** | **0.5787** | **0.2426** |
| 2 | CoTE-BoundaryLite-P4 Strong | 0.7207 | **0.5185** | 0.6355 | 0.3102 | 0.6138 | 0.4502 | 0.5279 | 0.2052 |
| 3 | LKA-SimAM Head | 0.7464 | 0.3028 | 0.5257 | 0.3099 | **0.7029** | 0.2730 | 0.4919 | 0.2135 |
| 4 | DPCA Strong | 0.6716 | 0.4897 | 0.5618 | 0.2307 | 0.5603 | 0.4214 | 0.4668 | 0.1726 |

### TTA theo lớp (Mask)

| Mô hình | BG P | BG R | BG mAP50 | BG mAP50-95 | WSSV P | WSSV R | WSSV mAP50 | WSSV mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SimAM-CA Strong | 0.5833 | 0.2692 | 0.4035 | 0.2113 | 0.7627 | **0.6716** | **0.7538** | **0.2739** |
| CoTE-BoundaryLite-P4 Strong | 0.5556 | 0.2885 | **0.4355** | 0.2022 | 0.6721 | 0.6119 | 0.6202 | 0.2083 |
| DPCA Strong | 0.4800 | 0.2308 | 0.3172 | 0.1450 | 0.6406 | 0.6119 | 0.6165 | 0.2002 |
| LKA-SimAM Head | **0.6667** | 0.0385 | 0.3578 | **0.2147** | 0.7391 | 0.5075 | 0.6261 | 0.2122 |

Ghi chú: BG có 52 instance và WSSV có 67 instance. LKA-SimAM Head có precision BG cao nhưng recall BG chỉ 0.0385 (3 dự đoán), cho thấy mô hình bỏ sót BG rất nhiều.

## 2. Augmented test (129 ảnh × 6 biến đổi = 774 ảnh)

| Hạng theo Mask mAP50 | Mô hình | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | SimAM-CA Strong | **0.7174** | **0.4848** | **0.4563** | **0.1675** | **0.6525** | **0.4411** | **0.3999** | **0.1249** |
| 2 | CoTE-BoundaryLite-P4 Strong | 0.6237 | 0.4765 | 0.4103 | 0.1557 | 0.5439 | 0.4191 | 0.3335 | 0.1058 |
| 3 | DPCA Strong | 0.5288 | 0.4458 | 0.3422 | 0.1196 | 0.4552 | 0.3941 | 0.2819 | 0.0816 |
| 4 | LKA-SimAM Head | 0.5386 | 0.4119 | 0.3441 | 0.1177 | 0.4042 | 0.3790 | 0.2786 | 0.0871 |

## 3. Kết luận

- **SimAM-CA Strong tốt nhất và ổn định nhất**: đứng đầu toàn bộ metric chính trong augmented test, đồng thời đứng đầu TTA về Box mAP50, Box mAP50-95, Mask recall, Mask mAP50 và Mask mAP50-95.
- **CoTE-BoundaryLite-P4 Strong đứng thứ hai rõ ràng**, có TTA Box recall cao nhất (0.5185) và TTA Mask mAP50 BG cao nhất (0.4355).
- **DPCA Strong và LKA-SimAM Head ở nhóm sau**. DPCA nhỉnh hơn LKA trên augmented-test Mask mAP50; LKA có TTA Mask mAP50 tổng cao hơn DPCA nhưng recall rất thấp, đặc biệt với BG.
- WSSV dễ nhận diện hơn BG ở cả bốn notebook TTA. Đây là điểm nghẽn chính cần ưu tiên khi cải thiện mô hình.

## 4. Lưu ý khi diễn giải

- TTA fused và augmented test không phải cùng một phép đo: TTA hợp nhất nhiều dự đoán rồi chấm trên 129 ảnh gốc; augmented test chấm riêng 774 ảnh đã biến đổi.
- Vì vậy, không nên lấy hiệu số giữa hai bảng để tuyên bố mức tăng/giảm do TTA. Hai bảng phù hợp để đánh giá lần lượt chất lượng ensemble khi inference và độ bền trước biến đổi ảnh.
- Notebook `ca-sim-strong-augment-test.ipynb` được ghép với `simam-ca-strong-tta-inference-1.ipynb` theo tên và cấu hình mô hình; tên được chuẩn hóa thành **SimAM-CA Strong** trong báo cáo.
