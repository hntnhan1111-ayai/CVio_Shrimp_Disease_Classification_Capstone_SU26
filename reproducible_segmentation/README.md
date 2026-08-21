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

## Benchmark model segmentation dưới 15M params

`config.yaml` là nguồn cấu hình duy nhất cho benchmark mới. File này khóa protocol
theo notebook `aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb`: split
grouped-stratified chống leakage, seed,
image size, optimizer, augmentation, checkpoint selection, confidence threshold và
healthy-aware metrics. Danh sách model từ tài liệu gợi ý được tách thành hai track:

- Primary instance track: YOLO11n/YOLO26n/YOLO26s và RTMDet-Ins tiny/s.
- Secondary semantic track: LR-ASPP, DeepLabV3-MNV3, SegFormer, TopFormer và
  PP-MobileSeg; track này cần class-index masks và không được trộn metric với primary.

Kiểm tra lệnh trước khi train:

```powershell
$env:CVIO_MRTU_DATA_YAML = 'D:\path\to\data.yaml'
python scripts/04_run_ultralytics_benchmark.py --model yolo11n_seg --dry-run
python scripts/04_run_ultralytics_benchmark.py --model yolo11n_seg --smoke
```

Sau smoke test, chạy full baseline rồi mới chạy các model Ultralytics với cùng split:

```powershell
python scripts/04_run_ultralytics_benchmark.py --model yolo11n_seg
python scripts/04_run_ultralytics_benchmark.py --model yolo26n_seg
python scripts/04_run_ultralytics_benchmark.py --model yolo26s_seg
```

Mỗi lần chạy ghi environment JSON và kiểm tra số params thực tế trước khi train.

Runner hiện thực thi ba model Ultralytics trong primary track. Nó xác minh hash notebook
tham chiếu, schema/class của dataset và giới hạn tham số trước khi gọi `YOLO.train()`.
`--dry-run` không import Torch hoặc tải checkpoint; `--smoke` vẫn cần dataset và checkpoint.

RTMDet dùng cùng Roboflow project/version nhưng tải `coco-segmentation`. Hai
notebook `rtmdet_ins_*_fix_leakage_coco.ipynb` tự gom lại toàn bộ export, tái tạo
split grouped-stratified seed 42, kiểm tra đúng 905/115/129 ảnh rồi train bằng
MMDetection. Dùng Kaggle Secret `ROBOFLOW_API_KEY`; không ghi API key vào notebook.

Notebook baseline gốc là read-only về mặt quy trình. SHA-256 của file được khóa
trong `config.yaml`; runner sẽ dừng nếu baseline đã thay đổi.

RTMDet-Ins cần adapter COCO và môi trường MMDetection riêng. Workflow này chưa được
đóng gói trong thư mục hiện tại và không được chạy qua script Ultralytics.

Lệnh cuối chỉ tạo manifest dưới `artifacts/generated/`; không khởi chạy train.
Xem [docs/REPRODUCE.md](docs/REPRODUCE.md) trước khi kết nối runner YOLO.
