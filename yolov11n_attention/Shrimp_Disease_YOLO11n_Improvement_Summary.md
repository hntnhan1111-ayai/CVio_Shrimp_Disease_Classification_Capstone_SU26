# Shrimp Disease Instance Segmentation using YOLO11n-Seg + Attention

## Tổng hợp dự án, các hướng cải tiến và kết quả thực nghiệm

Tài liệu tổng hợp toàn bộ quá trình nghiên cứu, đánh giá attention modules, các hướng cải tiến 1–8 và kết luận hiện tại.

## Kết luận chính

- Dataset: Shrimp Disease Segmentation (BG, WSSV)
- Base model: YOLO11n-Seg
- Best attention: SimAM + CA
- Best result:
  - Full Test Mask mAP50 = 0.495
  - Labeled Test Mask mAP50 = 0.519
  - Mask mAP50-95 = 0.173

## Các hướng đã thử

1. P2 Feature Injection
2. Larger Segmentation Head
3. Img Size & Mask Ratio
4. Dice/Tversky Loss
5. Auxiliary Semantic Branch
6. DFF Module
7. Copy-Paste Augmentation
8. Balanced Sampling

## Kết luận cuối

YOLO11n-Seg + SimAM_CA là mô hình tốt nhất hiện tại và được chọn làm mô hình chính cho luận văn.
