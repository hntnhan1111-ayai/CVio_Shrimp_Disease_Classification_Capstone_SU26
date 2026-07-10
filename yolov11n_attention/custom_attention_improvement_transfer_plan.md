# Kế Hoạch Cải Tiến Custom Attention Modules Cho YOLO11n-Seg

> Tài liệu này viết lại từ hướng phân tích trong `Phan_Tich_Va_De_Xuat_Cai_Tien_YOLO11n_Seg.md`, nhưng áp dụng riêng cho các module mới trong `custom_attention`, `custom_attention_research_modules`, và `custom_attention_multiseed`.

---

## 1. Mục Tiêu

Mục tiêu của kế hoạch này là xác định hướng cải tiến tiếp theo cho các custom attention modules đã chạy, dựa trên hai nguồn:

1. Kết quả baseline clean YOLO11n-seg:
   - Full test mask mAP50: `0.441`
   - Diseased-only mask mAP50: `0.466`
   - Full/Diseased mask mAP50-95: `0.161 / 0.169`
   - Healthy FP rate: `36.6%`
   - Healthy-aware score: `0.393`

2. Kết luận từ phân tích SimAM_CA:
   - Attention chỉ đặt ở head chưa đủ để xử lý mất thông tin sớm trong backbone.
   - Gap giữa mAP50 và mAP50-95 lớn, nghĩa là mask boundary/detail còn yếu.
   - Dataset nhỏ gây overfitting rõ.
   - Healthy false positive là KPI quan trọng.
   - Thêm nhiều attention block không đồng nghĩa tốt hơn.

Thông điệp trung tâm:

> Hướng tiếp theo không phải là thêm nhiều attention hơn, mà là kết hợp attention có kiểm soát với cải tiến mask head, calibration, training protocol và xử lý false positive.

---

## 2. Tóm Tắt Kết Quả Hiện Tại Của Module Mới

### 2.1 Baseline clean

| Metric | Baseline clean |
|---|---:|
| Full test mask mAP50 | 0.441 |
| Diseased-only mask mAP50 | 0.466 |
| Full test mask mAP50-95 | 0.161 |
| Diseased-only mask mAP50-95 | 0.169 |
| Healthy FP rate | 36.6% |
| Disease miss rate | 12.5% |
| Healthy-aware score | 0.393 |

### 2.2 Module đáng giữ lại

| Nhóm | Module | Tín hiệu chính | Vấn đề còn lại |
|---|---|---|---|
| Best clean candidate | Triplet Attention Segment Head | Gần baseline nhất; FP giảm nhẹ | mAP50-95 giảm, miss tăng |
| Best overall signal | CoTE strong | Disease mAP50 và HA score tốt nhất | Chưa công bằng vì dùng strong augmentation |
| Best low-FP direction | LPSC strong | Healthy FP thấp | Disease miss cao |
| Lightweight candidate | SGE-ECA Head Gate | Nhẹ, có giảm FP/miss single-run | Accuracy giảm nhiều |
| Mask-head candidate | CESA-Lite Segment Head | Tương đối cân bằng trong clean run | Miss và mAP50-95 chưa tốt |

### 2.3 Module không nên ưu tiên mở rộng

| Module | Lý do |
|---|---|
| SCSG | Accuracy giảm rất mạnh |
| P3P4 Semantic Attention Gate | Healthy FP tăng rất cao |
| CA-Spatial Low-FP Gate | Accuracy và mask quality giảm mạnh |
| Low-FP Residual Spatial Gate | Không đạt mục tiêu giảm FP |
| Boundary-Aware Lite Attention | FP healthy tăng mạnh do khuếch đại texture/noise |
| CoTE-BL P4 strong | FP thấp nhưng miss-rate quá cao |

---

## 3. Chẩn Đoán Bottleneck

| Bottleneck | Biểu hiện trong kết quả | Nguyên nhân khả dĩ | Hướng xử lý |
|---|---|---|---|
| Mask boundary/detail yếu | mAP50-95 thấp hơn nhiều so với mAP50 | Segment head chỉ có `nm=32`, nearest upsample, loss chưa ưu tiên boundary | `nm=64`, boundary-aware loss, bilinear/learned upsample |
| Attention head-only không đủ | Nhiều module head không vượt baseline | Backbone đã mất detail trước khi attention ở head can thiệp | Thêm attention nhẹ ở backbone P3/P4, ưu tiên SimAM |
| Overfitting dataset nhỏ | Best epoch sớm, loss gap cao | Dataset nhỏ, augmentation conservative, attention tăng capacity | Warmup/cosine, label smoothing, SWA, augmentation matched |
| Healthy FP cao | Nhiều module nhầm texture healthy thành bệnh | Healthy negative chiếm ~35%, texture vỏ tôm gây nhiễu | Threshold calibration, negative mining, low-FP soft gate |
| Low-FP đi kèm miss cao | LPSC/CoTE-BL giảm FP nhưng bỏ sót bệnh | Gate quá mạnh, suppression thiếu rescue path | Soft gate, SimAM rescue, tune gamma, threshold sweep |
| So sánh chưa công bằng | CoTE strong tốt nhưng khác augmentation | Strong augmentation có thể tự cải thiện metric | Rerun baseline strong matched |
| Quá nhiều attention gây xung đột | Combo 3 modules trong SimAM_CA analysis kém hơn 2 modules | Gradient conflict, trùng cơ chế attention | Mỗi experiment chỉ thay 1 yếu tố chính |

---

## 4. Nguyên Tắc Thiết Kế Thí Nghiệm Tiếp Theo

| Nguyên tắc | Cách áp dụng |
|---|---|
| Giữ protocol công bằng | Mỗi module phải có baseline cùng seed, split, augmentation, threshold và evaluation logic |
| Không cộng dồn attention tùy tiện | Không gắn `CA + SimAM + custom attention` cùng lúc ở head nếu chưa có ablation |
| Ưu tiên sửa mask head trước khi thêm attention mới | Vì mAP50-95 là điểm yếu chung |
| Tách mục tiêu `mAP` và `low-FP` | Module low-FP chỉ tốt nếu miss-rate không tăng quá nhiều |
| Dùng healthy-aware score làm metric quyết định phụ | Vì bài toán có nhiều healthy negatives |
| Luôn báo cáo full test và diseased-only test | Tránh kết luận sai do healthy images làm lệch full metric |
| Có threshold sweep sau training | Một model tốt có thể bị đánh giá thấp nếu threshold chưa tối ưu |

---

## 5. Nhóm A - Cải Tiến Evaluation Và Calibration

Đây là nhóm ưu tiên cao nhất vì không cần retrain hoặc ít tốn compute.

### A1. TTA Inference

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Kiểm tra model có tăng mAP nhờ test-time augmentation không |
| Áp dụng cho | Baseline, Triplet, CoTE, LPSC, CESA, SGE-ECA |
| Ưu tiên | Rất cao |
| Effort | Thấp |
| Kỳ vọng | Tăng mAP50 nhẹ, đặc biệt với lesion nhỏ |

Ví dụ:

```python
metrics = model.val(
    data=str(data_yaml),
    split="test",
    imgsz=640,
    augment=True,
    conf=0.25,
    iou=0.7,
)
```

Tiêu chí thành công:

- Full hoặc diseased mAP50 tăng.
- Healthy FP không tăng quá mạnh.
- Ranking module không bị đảo chiều bất lợi.

### A2. Threshold Sweep

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Tìm điểm cân bằng giữa mAP, FP healthy và disease miss |
| Áp dụng cho | Tất cả module, đặc biệt LPSC/Triplet/CoTE |
| Ưu tiên | Rất cao |
| Effort | Thấp |
| Metric chính | Healthy-aware score |

Sweep đề xuất:

| Parameter | Values |
|---|---|
| `conf` | 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50 |
| `iou` | 0.50, 0.60, 0.70 |

Tiêu chí chọn checkpoint/threshold:

```text
healthy_aware =
  diseased_mask_mAP50
  - 0.05 * count_MAE
  - 0.15 * disease_miss_rate
  - 0.10 * healthy_FP_rate
```

### A3. Matched Baseline Cho Strong Augmentation

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Kiểm tra CoTE strong tốt do module hay do augmentation |
| Áp dụng cho | CoTE strong, LPSC strong, các P4 strong variants |
| Ưu tiên | Rất cao |
| Effort | Trung bình |

Cần chạy:

1. Baseline YOLO11n-seg với cùng strong augmentation.
2. CoTE strong cùng seed/protocol.
3. LPSC strong cùng seed/protocol.

Kết luận chỉ hợp lệ nếu:

- CoTE strong vẫn cao hơn baseline strong về disease mAP50 hoặc healthy-aware score.
- FP giảm không đi kèm miss-rate tăng lớn.

---

## 6. Nhóm B - Cải Tiến Training Protocol

Nhóm này nên dùng làm protocol chung cho các retrain tiếp theo.

### B1. Warmup + Cosine LR + Patience Ngắn Hơn

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Ổn định training attention, giảm overfit cuối training |
| Áp dụng cho | Tất cả retrain modules |
| Ưu tiên | Cao |
| Effort | Thấp |

Config đề xuất:

```python
TRAIN_STABLE_ARGS = {
    "epochs": 150,
    "patience": 20,
    "cos_lr": True,
    "warmup_epochs": 8,
    "lr0": 0.01,
    "lrf": 0.001,
    "label_smoothing": 0.05,
}
```

Tiêu chí thành công:

- Best epoch không quá sớm.
- Val-train loss gap giảm.
- mAP50-95 hoặc healthy-aware score tăng.

### B2. Label Smoothing

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Giảm overconfidence và nhiễu annotation |
| Áp dụng cho | Triplet, CoTE, CESA, SGE-ECA, LPSC |
| Ưu tiên | Cao |
| Effort | Thấp |

Đề xuất:

```python
model.train(
    label_smoothing=0.05,
)
```

Phù hợp vì biên BG/WSSV có thể không sắc nét, annotation mask thủ công có noise.

### B3. Strong/Underwater Augmentation Matched

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Tăng robustness và giảm overfit |
| Áp dụng cho | Baseline, Triplet, CoTE, LPSC, CESA, SGE-ECA |
| Ưu tiên | Cao |
| Effort | Trung bình |

Config đề xuất:

```python
UNDERWATER_AUG_ARGS = {
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "fliplr": 0.5,
    "flipud": 0.3,
    "hsv_h": 0.05,
    "hsv_s": 0.50,
    "hsv_v": 0.40,
    "scale": 0.50,
    "translate": 0.10,
    "degrees": 10.0,
    "erasing": 0.10,
}
```

Lưu ý báo cáo:

- Nếu dùng augmentation mới, phải rerun baseline cùng augmentation.
- Không so trực tiếp với baseline clean nếu protocol khác.

### B4. SWA Sau Training

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Cải thiện generalization bằng averaging checkpoint |
| Áp dụng cho | Module có overfit hoặc best epoch dao động |
| Ưu tiên | Trung bình |
| Effort | Trung bình |

Nên dùng sau khi đã có nhiều checkpoint cuối hoặc checkpoint quanh best epoch.

---

## 7. Nhóm C - Cải Tiến Mask Head Và Boundary Quality

Đây là nhóm có tác động cao vì tất cả model đều yếu ở mAP50-95.

### C1. Tăng Prototype `nm=32 -> nm=64`

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Tăng số mask basis vectors cho Segment head |
| Áp dụng cho | Triplet, CESA, CoTE, LPSC, SGE-ECA |
| Ưu tiên | Rất cao |
| Effort | Thấp-Trung bình |

YAML:

```yaml
- [[p3, p4, p5], 1, Segment, [nc, 64, 256]]
```

Nên chạy trước trên:

| Priority | Module |
|---:|---|
| 1 | Triplet Attention Segment Head |
| 2 | CoTE |
| 3 | CESA-Lite Segment Head |
| 4 | SGE-ECA Head Gate |
| 5 | LPSC-soft |

Tiêu chí thành công:

- mAP50-95 tăng.
- Diseased mAP50 không giảm.
- Healthy FP không tăng quá nhiều.

### C2. Boundary-Aware Mask Loss

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Ép model học biên mask tốt hơn |
| Áp dụng cho | Triplet, CoTE, CESA, Prototype-aware |
| Ưu tiên | Cao |
| Effort | Cao hơn vì cần sửa loss |

Nên dùng sau khi `nm64` đã được kiểm chứng.

Lý do:

- Boundary-aware attention hiện tại làm FP tăng vì nó khuếch đại texture.
- Boundary loss an toàn hơn vì tác động vào objective, không trực tiếp khuếch đại feature healthy.

Tiêu chí thành công:

- Diseased mAP50-95 tăng rõ.
- Mask boundary qualitative tốt hơn.
- Healthy FP không tăng mạnh.

### C3. Bilinear Hoặc Learned Upsampling

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Giảm artifact do nearest-neighbor upsampling |
| Áp dụng cho | Baseline, Triplet, CoTE, CESA, SGE-ECA |
| Ưu tiên | Trung-Cao |
| Effort | Trung bình |

YAML thử nghiệm đơn giản:

```yaml
- [-1, 1, nn.Upsample, [None, 2, "bilinear"]]
```

Nên test theo thứ tự:

1. Baseline + bilinear.
2. Triplet + bilinear.
3. CoTE + bilinear.

Không nên kết hợp ngay với nhiều thay đổi khác, để biết riêng upsampling có lợi hay không.

---

## 8. Nhóm D - Cải Tiến Kiến Trúc Attention Có Kiểm Soát

Nhóm này chỉ nên chạy sau A/B/C, vì chi phí retrain cao và dễ gây confound.

### D1. SimAM Ở Backbone P3/P4

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Tập trung vào lesion từ feature sớm, trước khi head xử lý |
| Áp dụng cho | Triplet, CoTE, SGE-ECA, LPSC-soft |
| Ưu tiên | Cao trong nhóm kiến trúc |
| Effort | Trung bình |

Ý tưởng:

```yaml
backbone:
  ...
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, SimAM, []]
  ...
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, SimAM, []]
```

Module nên ghép:

| Module head | Vì sao hợp với SimAM-backbone |
|---|---|
| Triplet | Triplet đã gần baseline, SimAM-backbone có thể phục hồi lesion focus sớm |
| CoTE | CoTE có HA tốt, SimAM-backbone có thể tăng recall/mAP |
| SGE-ECA | Head gate nhẹ, backbone SimAM không tăng params nhiều |
| LPSC-soft | Cần rescue true lesion để giảm miss |

Không nên ghép ngay với:

- CESA-Lite, vì CESA đã có SimAM-like/neuron saliency ở head.
- CA-lite spatial variants, vì dễ trùng spatial gating.

### D2. CA Dual Hoặc CA Backbone

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Bổ sung coordinate cue từ feature thấp |
| Áp dụng cho | Triplet, CoTE |
| Ưu tiên | Trung bình |
| Effort | Trung bình |

Chỉ nên chạy nếu:

- SimAM-backbone không đủ.
- Module không có coordinate gate sẵn.

Không nên ưu tiên cho:

- CESA-Lite
- CA-Lite Spatial Gate
- CA-Spatial Low-FP Gate

Vì các module này đã có cơ chế coordinate/spatial, dễ trùng lặp.

### D3. LKA Head Thay Cho Một Số Spatial Gate

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Capture pattern WSSV phân mảnh với receptive field lớn hơn |
| Áp dụng cho | CoTE, Triplet |
| Ưu tiên | Trung bình-Thấp |
| Effort | Trung bình-Cao |

Nên xem đây là ablation phase sau, không phải hướng chạy đầu tiên.

---

## 9. Nhóm E - Rescue Cho Low-FP Modules

Nhóm này dành riêng cho các module giảm FP nhưng tăng miss-rate.

### E1. LPSC Soft Gate

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Giữ khả năng giảm FP của LPSC nhưng giảm bỏ sót bệnh |
| Áp dụng cho | LPSC strong, LPSC-ER, LPSC-UA |
| Ưu tiên | Cao nếu mục tiêu là low-FP deployment |
| Effort | Trung bình |

Hướng sửa:

- Giảm gate strength/gamma.
- Dùng residual centered gate.
- Thêm SimAM rescue path.
- Tune threshold sau training.

Tiêu chí thành công:

| Metric | Mục tiêu |
|---|---|
| Healthy FP | Thấp hơn baseline |
| Disease miss | Không vượt baseline quá nhiều |
| Diseased mAP50 | Không giảm quá 0.02-0.03 so với baseline |
| HA score | Tăng hoặc ngang baseline |

### E2. CoTE Rescue Thay Vì CoTE-BL

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Giữ cân bằng tốt của CoTE strong nhưng không làm miss tăng |
| Áp dụng cho | CoTE strong |
| Ưu tiên | Cao |

Không nên tiếp tục CoTE-BL làm hướng chính vì FP thấp nhưng miss-rate quá cao. Nếu muốn cải tiến CoTE, nên ưu tiên:

1. CoTE + `nm64`
2. CoTE + boundary loss
3. CoTE + threshold calibration
4. CoTE + SimAM-backbone

### E3. Prototype-Aware Gate Với Boundary Loss

| Thuộc tính | Nội dung |
|---|---|
| Mục tiêu | Biến mask-specific gate thành hướng mask-quality thật sự |
| Áp dụng cho | Prototype-Aware Mask Gate Lite |
| Ưu tiên | Trung bình |

Prototype-aware giảm FP mạnh nhưng mAP50-95 thấp. Vì vậy không nên tăng gate tiếp; nên cải thiện objective:

- `nm64`
- boundary-aware loss
- threshold sweep

---

## 10. Nhóm F - Hướng Nâng Cao

Nhóm này chỉ nên làm sau khi đã xác nhận các hướng A-E.

| Hướng | Áp dụng cho | Mục tiêu | Ưu tiên |
|---|---|---|---:|
| WIoU v3 | Tất cả | Giảm ảnh hưởng annotation outlier ở box | Trung bình |
| Pseudo-labeling | Baseline/CoTE/Triplet teacher | Mở rộng data | Trung bình-Thấp |
| Knowledge Distillation | Student YOLO11n + best module | Học representation từ YOLO11m/l | Thấp trong giai đoạn hiện tại |
| Negative mining | Module FP cao | Thêm healthy hard negatives | Trung bình |
| CLAHE preprocessing | Dataset toàn bộ | Tăng contrast lesion | Trung bình, cần kiểm chứng kỹ |

Lưu ý:

- Pseudo-labeling và KD có thể tạo thêm confound, không nên dùng trước khi có ranking module ổn định.
- CLAHE có thể giúp WSSV sáng hơn nhưng cũng có nguy cơ làm healthy texture giống lesion hơn.

---

## 11. Kế Hoạch Chạy Theo Phase

### Phase 0 - Chuẩn Hóa Báo Cáo Và Evaluation

| Priority | Experiment | Module | Mục tiêu | Output |
|---:|---|---|---|---|
| 1 | Baseline clean threshold sweep | Baseline | Lấy threshold fair nhất | CSV threshold + HA score |
| 2 | TTA baseline | Baseline | Biết lợi ích TTA | Test metrics |
| 3 | TTA + threshold sweep | Triplet, CoTE, LPSC, CESA, SGE-ECA | So lại ranking không retrain | Ranking mới |

Kết quả cần có:

- Full mAP50 / mAP50-95
- Diseased mAP50 / mAP50-95
- Healthy FP rate
- FP masks/image
- Disease miss rate
- Count MAE
- Healthy-aware score

### Phase 1 - Matched Protocol Cho Candidate Chính

| Priority | Experiment | Vì sao |
|---:|---|---|
| 1 | Baseline strong matched | Cần so công bằng với CoTE strong |
| 2 | CoTE strong matched | Xác nhận CoTE có thật sự tốt |
| 3 | LPSC strong matched | Xác nhận low-FP trade-off |
| 4 | Triplet matched 3 seeds | Best clean candidate cần kiểm chứng ổn định |

Tiêu chí quyết định:

- Nếu CoTE strong vẫn vượt baseline strong về HA score, giữ CoTE làm hướng chính.
- Nếu Triplet 3 seeds ổn định và FP thấp, giữ Triplet làm hướng clean-safe.
- Nếu LPSC miss quá cao, chỉ giữ làm low-FP auxiliary.

### Phase 2 - Mask Head Improvement

| Priority | Experiment | Module | Mục tiêu |
|---:|---|---|---|
| 1 | `nm64` | Triplet | Tăng mask detail |
| 2 | `nm64` | CoTE | Tăng mAP50-95 cho best HA candidate |
| 3 | `nm64` | CESA | Kiểm tra segment-head module |
| 4 | Boundary loss | Triplet/CoTE | Tăng boundary quality |
| 5 | Bilinear upsample | Baseline/Triplet | Kiểm tra bottleneck upsampling |

Không nên chạy `nm64 + boundary loss + bilinear + SimAM-backbone` cùng một lúc ngay từ đầu. Cần ablation từng bước.

### Phase 3 - Controlled Backbone Attention

| Priority | Experiment | Module | Mục tiêu |
|---:|---|---|---|
| 1 | SimAM-backbone + Triplet head | Triplet | Cứu feature sớm, giữ head tốt |
| 2 | SimAM-backbone + CoTE head | CoTE | Tăng disease mAP và HA |
| 3 | SimAM-backbone + SGE-ECA head | SGE-ECA | Edge-friendly variant |
| 4 | SimAM-backbone + LPSC-soft | LPSC | Giảm miss của low-FP gate |

### Phase 4 - Low-FP Rescue Và Advanced

| Priority | Experiment | Mục tiêu |
|---:|---|---|
| 1 | LPSC-soft gamma sweep | Tìm gate strength không bỏ sót bệnh |
| 2 | CoTE + boundary loss | Cải thiện mAP50-95 |
| 3 | Negative mining healthy | Giảm FP texture |
| 4 | SWA | Cải thiện generalization |
| 5 | KD hoặc pseudo-labeling | Chỉ làm khi cần tăng trần accuracy |

---

## 12. Ma Trận Ưu Tiên Cuối Cùng

| Rank | Hướng | Experiment đại diện | Lý do ưu tiên |
|---:|---|---|---|
| 1 | Evaluation calibration | TTA + threshold sweep | Nhanh, không retrain, có thể đổi kết luận |
| 2 | Matched strong baseline | Baseline strong vs CoTE strong | Bắt buộc để kết luận công bằng |
| 3 | Mask head `nm64` | Triplet/CoTE + nm64 | Đánh đúng bottleneck mask mAP50-95 |
| 4 | Stable training protocol | warmup/cosine + label smoothing | Giảm overfit cho tất cả retrain |
| 5 | Boundary objective | CoTE/Triplet + boundary loss | Tăng mask detail trực tiếp |
| 6 | SimAM-backbone | Triplet/CoTE + SimAM P3/P4 | Bổ sung attention sớm thay vì chỉ ở head |
| 7 | LPSC rescue | LPSC-soft + threshold/gamma sweep | Giữ low-FP nhưng giảm miss |
| 8 | Bilinear/learned upsample | Baseline/Triplet + upsample change | Kiểm tra nguyên nhân boundary thô |
| 9 | LKA/CA dual | Triplet/CoTE variants | Là kiến trúc phụ, chạy sau |
| 10 | KD/pseudo-labeling | Teacher-student hoặc unlabeled data | Nâng cao, dễ confound |

---

## 13. Kế Hoạch Cụ Thể Theo Module

| Module | Vai trò | Experiment nên chạy | Tiêu chí giữ lại | Tiêu chí loại |
|---|---|---|---|---|
| Triplet Attention Segment Head | Best clean candidate | TTA/threshold, 3 seeds, nm64, SimAM-backbone | HA ngang/tăng baseline, FP giảm, disease mAP không giảm nhiều | mAP50-95 tiếp tục giảm hoặc miss tăng |
| CoTE strong | Best overall signal | Matched baseline strong, nm64, boundary loss | Vượt baseline strong về HA/disease mAP | Chỉ tốt do augmentation |
| LPSC strong | Low-FP candidate | threshold sweep, soft gamma, SimAM rescue | FP giảm nhưng miss không quá cao | Miss-rate cao hơn 20% kéo dài |
| CESA-Lite | Mask-head candidate | nm64, stable training, boundary loss | mAP50-95 tăng và miss giảm | Vẫn kém Triplet/CoTE toàn diện |
| SGE-ECA | Lightweight candidate | SimAM-backbone, nm64, group/gamma tune | Accuracy phục hồi, FP vẫn thấp | mAP tiếp tục thấp |
| Prototype-aware | Auxiliary mask candidate | nm64 + boundary loss | mAP50-95 phục hồi | FP giảm nhưng accuracy quá thấp |
| Context suppression | Auxiliary context candidate | soft gate + threshold | Miss thấp, FP giảm | Suppress disease tiếp tục |

---

## 14. Template Báo Cáo Kết Quả

Khi báo cáo từng experiment, dùng cùng format:

| Field | Nội dung |
|---|---|
| Experiment ID | Ví dụ `triplet_nm64_stable_seed42` |
| Baseline matched | Baseline clean/strong nào được dùng |
| Module change | Chỉ rõ thay đổi duy nhất |
| Training protocol | Seed, epochs, patience, augmentation |
| Full test metrics | Box/mask mAP50, mAP50-95 |
| Diseased-only metrics | Mask mAP50, mAP50-95, miss-rate, count MAE |
| Healthy metrics | FP rate, FP masks/image |
| Healthy-aware score | Score tổng hợp |
| Qualitative | Ảnh lesion nhỏ, ảnh healthy noise, boundary case |
| Decision | Keep / retry / reject |

Kết luận mẫu:

> Experiment này được xem là cải thiện nếu healthy-aware score tăng, diseased-only mAP50 không giảm đáng kể, mAP50-95 tăng hoặc giữ ổn định, và healthy FP không tăng so với matched baseline.

---

## 15. Kết Luận Định Hướng

Các cải tiến từ SimAM_CA có thể chuyển sang custom attention modules, nhưng nên chuyển theo thứ tự:

1. Chuẩn hóa evaluation bằng TTA và threshold sweep.
2. Rerun baseline matched cho các strong-augmentation modules.
3. Cải thiện mask head bằng `nm64`, boundary loss và upsampling.
4. Ổn định training bằng warmup/cosine, label smoothing và augmentation matched.
5. Chỉ sau đó mới thêm attention ở backbone, ưu tiên SimAM P3/P4.
6. Với low-FP modules như LPSC, tập trung vào soft gate và recall rescue thay vì tăng suppression.

Tóm lại:

> Không nên báo cáo rằng hướng tiếp theo là "thêm attention". Nên báo cáo rằng hướng tiếp theo là "controlled attention + better mask head + calibrated inference + matched training protocol".
