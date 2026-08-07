# Tổng hợp kết quả CA→SimAM và LKA→SimAM theo vị trí attention

Nguồn dữ liệu:

- `ca_simam_all_position_results.zip`
- `lka_simam_all_position_results.zip`

Ngày tổng hợp: 2026-08-07.

## Lưu ý phương pháp

- CA→SimAM được huấn luyện bằng **strong augmentation**.
- LKA→SimAM dùng **clean augmentation** đúng theo notebook nguồn.
- Hai nhóm dùng cùng split theo shrimp ID và cùng bộ test 129 ảnh/119 instances; labeled-only test có 88 ảnh/119 instances.
- Vì augmentation khác nhau, bảng phản ánh hiệu quả của toàn pipeline, không cô lập hoàn toàn tác động của attention.
- Gói CA→SimAM thiếu ZIP kết quả con của `03_head_p4`. Các metric mAP và training vẫn được lấy từ output notebook, nhưng `healthy_aware_labeled_test_mask_map50` không còn trong bảng HTML rút gọn nên được ghi là `N/A`.

## Bảng xếp hạng theo full-test mask mAP50

| Rank | Kiến trúc | Vị trí | Full box mAP50 | Full mask mAP50 | Full mask mAP50-95 | Labeled mask mAP50 | Labeled mask mAP50-95 | Healthy-aware score | Train (phút) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | CA→SimAM | Head P4 | 0.564 | **0.477** | 0.159 | 0.505 | 0.168 | N/A | 67.29 |
| 2 | LKA→SimAM | Head P3 | 0.529 | **0.474** | **0.167** | **0.508** | **0.180** | **0.444831** | 46.19 |
| 3 | CA→SimAM | Neck P3/P4 | 0.536 | 0.443 | 0.151 | 0.473 | 0.159 | 0.397888 | 69.95 |
| 4 | CA→SimAM | Sau C2PSA | 0.525 | 0.438 | 0.162 | 0.458 | 0.169 | 0.389839 | 70.30 |
| 5 | LKA→SimAM | Neck P3/P4 | 0.524 | 0.420 | 0.156 | 0.437 | 0.161 | 0.361539 | 53.20 |
| 6 | LKA→SimAM | Sau C2PSA | 0.475 | 0.416 | 0.130 | 0.471 | 0.151 | 0.398165 | 46.60 |
| 7 | CA→SimAM | Neck concat | 0.533 | 0.409 | 0.134 | 0.433 | 0.142 | 0.364885 | 69.25 |
| 8 | CA→SimAM | Head P3 | 0.504 | 0.409 | 0.137 | 0.434 | 0.146 | 0.356079 | 71.89 |
| 9 | LKA→SimAM | Head P4 | 0.479 | 0.393 | 0.116 | 0.445 | 0.131 | 0.370672 | 42.16 |
| 10 | CA→SimAM | Backbone P3 | 0.462 | 0.385 | 0.128 | 0.409 | 0.136 | 0.329605 | 67.39 |
| 11 | LKA→SimAM | Neck concat | 0.403 | 0.355 | 0.091 | 0.409 | 0.105 | 0.338228 | 83.09 |
| 12 | LKA→SimAM | Backbone P3 | 0.414 | 0.304 | 0.101 | 0.324 | 0.107 | 0.221613 | 69.62 |

## Bảng tổng hợp module CA→SimAM

| Vị trí | Full box mAP50 | Full box mAP50-95 | Full mask mAP50 | Full mask mAP50-95 | Labeled box mAP50 | Labeled mask mAP50 | Labeled mask mAP50-95 | Healthy-aware score | Train (phút) | Epoch chạy | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Neck concat | 0.533 | 0.223 | 0.409 | 0.134 | 0.564 | 0.433 | 0.142 | 0.364885 | 69.25 | 100 | 87 |
| Head P3 | 0.504 | 0.198 | 0.409 | 0.137 | 0.536 | 0.434 | 0.146 | 0.356079 | 71.89 | 100 | 97 |
| **Head P4** | **0.564** | 0.219 | **0.477** | 0.159 | **0.591** | **0.505** | 0.168 | N/A | 67.29 | 100 | 72 |
| Sau C2PSA | 0.525 | 0.220 | 0.438 | **0.162** | 0.547 | 0.458 | **0.169** | 0.389839 | 70.30 | 100 | 98 |
| Backbone P3 | 0.462 | 0.191 | 0.385 | 0.128 | 0.498 | 0.409 | 0.136 | 0.329605 | **67.39** | 100 | 78 |
| Neck P3/P4 | 0.536 | **0.232** | 0.443 | 0.151 | 0.567 | 0.473 | 0.159 | **0.397888** | 69.95 | 100 | 96 |

**Tốt nhất theo full-test mask mAP50:** CA→SimAM tại Head P4, đạt **0.477**.

## Bảng tổng hợp module LKA→SimAM

| Vị trí | Full box mAP50 | Full box mAP50-95 | Full mask mAP50 | Full mask mAP50-95 | Labeled box mAP50 | Labeled mask mAP50 | Labeled mask mAP50-95 | Healthy-aware score | Train (phút) | Epoch chạy | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Neck concat | 0.403 | 0.141 | 0.355 | 0.091 | 0.459 | 0.409 | 0.105 | 0.338228 | 83.09 | 94 | 64 |
| **Head P3** | **0.529** | 0.211 | **0.474** | **0.167** | **0.569** | **0.508** | **0.180** | **0.444831** | 46.19 | 64 | 52 |
| Head P4 | 0.479 | 0.167 | 0.393 | 0.116 | 0.540 | 0.445 | 0.131 | 0.370672 | **42.16** | **59** | 29 |
| Sau C2PSA | 0.475 | 0.183 | 0.416 | 0.130 | 0.539 | 0.471 | 0.151 | 0.398165 | 46.60 | 64 | 34 |
| Backbone P3 | 0.414 | 0.138 | 0.304 | 0.101 | 0.443 | 0.324 | 0.107 | 0.221613 | 69.62 | 100 | 58 |
| Neck P3/P4 | 0.524 | **0.223** | 0.420 | 0.156 | 0.547 | 0.437 | 0.161 | 0.361539 | 53.20 | 69 | 42 |

**Tốt nhất theo full-test mask mAP50:** LKA→SimAM tại Head P3, đạt **0.474**.

## So sánh trực tiếp theo từng vị trí

| Vị trí | CA→SimAM full mask mAP50 | LKA→SimAM full mask mAP50 | Chênh lệch CA−LKA | Tốt hơn |
|---|---:|---:|---:|---|
| Neck concat | 0.409 | 0.355 | +0.054 | CA→SimAM |
| Head P3 | 0.409 | 0.474 | −0.065 | LKA→SimAM |
| Head P4 | 0.477 | 0.393 | +0.084 | CA→SimAM |
| Sau C2PSA | 0.438 | 0.416 | +0.022 | CA→SimAM |
| Backbone P3 | 0.385 | 0.304 | +0.081 | CA→SimAM |
| Neck P3/P4 | 0.443 | 0.420 | +0.023 | CA→SimAM |

## Metric test đầy đủ

| Kiến trúc | Vị trí | Full box mAP50 | Full box mAP50-95 | Full mask mAP50 | Full mask mAP50-95 | Labeled box mAP50 | Labeled box mAP50-95 | Labeled mask mAP50 | Labeled mask mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CA→SimAM | Neck concat | 0.533 | 0.223 | 0.409 | 0.134 | 0.564 | 0.235 | 0.433 | 0.142 |
| CA→SimAM | Head P3 | 0.504 | 0.198 | 0.409 | 0.137 | 0.536 | 0.211 | 0.434 | 0.146 |
| CA→SimAM | Head P4 | 0.564 | 0.219 | 0.477 | 0.159 | 0.591 | 0.228 | 0.505 | 0.168 |
| CA→SimAM | Sau C2PSA | 0.525 | 0.220 | 0.438 | 0.162 | 0.547 | 0.230 | 0.458 | 0.169 |
| CA→SimAM | Backbone P3 | 0.462 | 0.191 | 0.385 | 0.128 | 0.498 | 0.204 | 0.409 | 0.136 |
| CA→SimAM | Neck P3/P4 | 0.536 | 0.232 | 0.443 | 0.151 | 0.567 | 0.244 | 0.473 | 0.159 |
| LKA→SimAM | Neck concat | 0.403 | 0.141 | 0.355 | 0.091 | 0.459 | 0.160 | 0.409 | 0.105 |
| LKA→SimAM | Head P3 | 0.529 | 0.211 | 0.474 | 0.167 | 0.569 | 0.229 | 0.508 | 0.180 |
| LKA→SimAM | Head P4 | 0.479 | 0.167 | 0.393 | 0.116 | 0.540 | 0.190 | 0.445 | 0.131 |
| LKA→SimAM | Sau C2PSA | 0.475 | 0.183 | 0.416 | 0.130 | 0.539 | 0.212 | 0.471 | 0.151 |
| LKA→SimAM | Backbone P3 | 0.414 | 0.138 | 0.304 | 0.101 | 0.443 | 0.149 | 0.324 | 0.107 |
| LKA→SimAM | Neck P3/P4 | 0.524 | 0.223 | 0.420 | 0.156 | 0.547 | 0.231 | 0.437 | 0.161 |

## Metric training và generalization

| Kiến trúc | Vị trí | Epoch chạy | Best epoch | Best val mask mAP50 | Best val mask mAP50-95 | Last train seg loss | Last val seg loss | Loss gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CA→SimAM | Neck concat | 100 | 87 | 0.57374 | 0.19507 | 2.81518 | 3.65226 | 0.83708 |
| CA→SimAM | Head P3 | 100 | 97 | 0.53227 | 0.16303 | 2.74560 | 3.87499 | 1.12939 |
| CA→SimAM | Head P4 | 100 | 72 | 0.56010 | 0.19309 | 2.80127 | 3.72694 | 0.92567 |
| CA→SimAM | Sau C2PSA | 100 | 98 | 0.54419 | 0.18322 | 2.80642 | 3.46147 | 0.65505 |
| CA→SimAM | Backbone P3 | 100 | 78 | 0.49624 | 0.18209 | 3.00820 | 3.67428 | 0.66608 |
| CA→SimAM | Neck P3/P4 | 100 | 96 | 0.55358 | 0.18589 | 2.82055 | 3.45452 | **0.63397** |
| LKA→SimAM | Neck concat | 94 | 64 | 0.47771 | 0.16569 | 1.47933 | 5.00953 | 3.53020 |
| LKA→SimAM | Head P3 | 64 | 52 | 0.46473 | 0.13718 | 2.05570 | 4.38443 | 2.32873 |
| LKA→SimAM | Head P4 | 59 | 29 | 0.43706 | 0.15796 | 2.19801 | 4.71077 | 2.51276 |
| LKA→SimAM | Sau C2PSA | 64 | 34 | 0.48592 | 0.16703 | 2.18677 | 4.11387 | **1.92710** |
| LKA→SimAM | Backbone P3 | 100 | 58 | 0.51669 | 0.17724 | 1.93950 | 4.71078 | 2.77128 |
| LKA→SimAM | Neck P3/P4 | 69 | 42 | 0.48136 | 0.15809 | 1.74974 | 4.81739 | 3.06765 |

## Kết luận

1. **CA→SimAM tại Head P4** đạt full-test mask mAP50 cao nhất: **0.477**.
2. **LKA→SimAM tại Head P3** gần như ngang bằng về full-test mask mAP50 (**0.474**), đồng thời tốt nhất về labeled mask mAP50 (**0.508**), labeled mask mAP50-95 (**0.180**) và healthy-aware score trong các cấu hình có đủ metric (**0.444831**).
3. CA→SimAM tốt hơn LKA→SimAM tại 5/6 vị trí theo full-test mask mAP50; ngoại lệ là Head P3.
4. Trong nhóm CA→SimAM, Neck P3/P4 có loss gap thấp nhất (**0.63397**). Trong nhóm LKA→SimAM, sau C2PSA có loss gap thấp nhất (**1.92710**).
5. Nếu ưu tiên độ chính xác segmentation tổng thể, chọn CA→SimAM Head P4. Nếu ưu tiên labeled-only accuracy, healthy-aware score và thời gian train thấp hơn, LKA→SimAM Head P3 là lựa chọn cân bằng hơn.
