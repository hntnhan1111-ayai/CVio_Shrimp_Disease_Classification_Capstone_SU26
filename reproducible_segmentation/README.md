# YOLO11n-Seg reproducibility layer

Đây là lớp gọn, được thêm song song với `yolov11n_attention/`; không di chuyển
hay thay thế notebook, module và run lịch sử. Nó đóng gói một baseline và năm
ứng viên được xếp hạng trong báo cáo:

1. SimAM + CA strong
2. LKA → SimAM head
3. DPCA strong
4. CoTE + BoundaryLite P4 strong
5. SimAM + CA + WIoU v3

Mục tiêu là chuẩn hoá catalog thí nghiệm, protocol, kết quả lịch sử và các run
manifest cho hai dataset/protocol đang theo dõi. Các implementation trong
notebook cũ vẫn là nguồn chuẩn cho đến khi được port và kiểm thử riêng.

## Layout

```text
configs/       catalog hai dataset, strong augmentation, baseline và top-5
src/           catalog ứng viên, protocol và hàm healthy-aware score
scripts/       kiểm tra môi trường, catalog và tạo run manifest
artifacts/     CSV kết quả lịch sử, trạng thái dataset mới, source inventory
docs/          dữ liệu, tái lập, kết quả, migration và các hạng mục còn thiếu
tests/         unit test không cần dataset hoặc GPU
```

## Quick checks

Từ thư mục này, dùng Python 3.10+:

```powershell
python scripts/01_validate_catalog.py
python -m unittest discover -s tests -v
python scripts/02_create_run_manifest.py --dataset mrtu_v1 --candidate simam_ca_strong --seed 42
```

Lệnh cuối chỉ tạo manifest dưới `artifacts/generated/`; không khởi chạy train.
Xem [docs/REPRODUCE.md](docs/REPRODUCE.md) trước khi kết nối runner YOLO.
