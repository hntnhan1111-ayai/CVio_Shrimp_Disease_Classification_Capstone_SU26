# Phan tich huong cai tien Nhom D

Nhom D tap trung vao chien luoc training thay vi tiep tuc tang do phuc tap kien truc. Cac ket qua hien co cho thay baseline sach kha manh, WIoU v3 cua nhom B giam healthy false positive tot, con nhom C chi co LKA head vuot baseline ro rang. Vi vay nhom D nen giu kien truc SimAM+CA head lam backbone thi nghiem on dinh, sau do can thiep vao cach toi uu va cach chon nguong du doan.

## D1 - SWA checkpoint averaging

Muc tieu la lam mo hinh bot bam vao mot checkpoint don le trong dataset nho. Notebook `yolov11n_simam_swa.ipynb` train voi `save_period=5`, trung binh cac checkpoint cuoi, roi danh gia `swa_avg.pt`. Huong nay co rui ro ton them dung luong checkpoint nhung khong doi YAML hay loss.

## D2 - Label smoothing + confidence calibration

Local Ultralytics config khong co `label_smoothing`, nen notebook `yolov11n_simam_label_smoothing_calibration.ipynb` patch truc tiep BCE classification target trong `v8DetectionLoss`. Sau training, notebook sweep nguong confidence tren validation de giam healthy false positive truoc khi danh gia test.

## D3 - Gradient accumulation

Ultralytics build nay khong nhan tham so `accumulate` truc tiep; trainer tinh accumulation bang `round(nbs / batch)`. Notebook `yolov11n_simam_gradient_accumulation.ipynb` dung `batch=4` va `nbs=32`, tuong duong effective batch gan 32. Huong nay uu tien gradient on dinh hon, doi lai training lau hon.

## D4 - Warmup + cosine LR

Notebook `yolov11n_simam_warmup_cosine.ipynb` dung warmup dai hon, `cos_lr=True`, `lrf=0.0001`, `patience=20`, `epochs=150`. Day la bien the duoc Ultralytics ho tro truc tiep cua huong SGDR/warmup-cosine trong tai lieu, tranh patch scheduler phuc tap de giam loi runtime.

## Ghi chu channel va patch module

Tat ca notebook dung YAML SimAM+CA voi channel thuc sau width scaling cua YOLO11n: P3=64, P4=128, P5=256. Runtime patch register `SimAM` va `CoordAtt` vao namespace cua Ultralytics truoc khi ghi YAML va smoke-build model, nen neu module hoac channel sai, loi se xuat hien truoc training.
