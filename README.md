# CVio Shrimp Disease Classification Capstone

Repository này lưu mã và tài liệu cho các thử nghiệm YOLO11n-Seg phát hiện
bệnh trên tôm. Các kết quả của **expanded dataset
(ShrimpDisBD-TigerShrimp_MrTuDat v1)** đã được chắt lọc để có thể review và
push lên Git mà không kèm dataset, checkpoint hoặc file chạy thô.

## Cấu trúc push lên Git

```text
reproducible_segmentation/  cấu hình, top-5 module, kết quả CSV/biểu đồ đã chọn và tài liệu tái lập
yolov11n_attention/         mã Ultralytics vendored và notebook thử nghiệm gốc
outputs/                    gói kết quả thô tại máy (bị Git bỏ qua; xem outputs/README.md)
```

`reproducible_segmentation/` được thêm song song, không thay thế notebook
hiện có. Xem [hướng dẫn tái lập](reproducible_segmentation/README.md),
[kết quả expanded dataset](reproducible_segmentation/docs/RESULTS.md) và
[danh sách hạng mục còn thiếu](reproducible_segmentation/docs/MISSING_ITEMS.md).

## Quy ước Git

- Commit mã, notebook cần giữ, cấu hình và artefact gọn trong
  `reproducible_segmentation/`.
- Không commit dữ liệu, weight, cache, kết quả Kaggle thô hoặc ZIP trong
  `outputs/`; chúng được bỏ qua qua `.gitignore`.
- Khi có ZIP kết quả mới, đặt chúng vào `outputs/` rồi chạy
  `python reproducible_segmentation/scripts/03_import_mrtu_outputs.py` để cập
  nhật các CSV/biểu đồ gọn trước khi commit.
