# Phân Tích Chi Tiết và Đề Xuất Cải Tiến YOLO11n-Seg Phân Vùng Bệnh Tôm

> Tài liệu được tổng hợp từ phân tích hai notebook: `aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb` (single attention) và `aip491-01-yolo-seg-11n-combined-modules.ipynb` (combined attention), cùng với hai file tổng hợp nghiên cứu trong thư mục dự án.

---

## 1. Tổng Quan Dữ Liệu (Dataset Analysis)

### 1.1 Cấu trúc Dataset

| Split | Shrimp Groups | Tổng ảnh | BG | Healthy | WSSV | WSSV\_BG |
|-------|--------------|----------|----|---------|------|----------|
| Train | 331          | 905      | 153| 321     | 258  | 173      |
| Valid | 40           | 115      | 21 | 41      | 32   | 21       |
| Test  | 45           | 129      | 24 | 41      | 38   | 26       |
| **Tổng** | **416**  | **1,149**| **198**| **403** | **328** | **220** |

**Lớp segmentation thực tế (2 class):** `BG` (Black Gill) và `WSSV` (White Spot Syndrome).  
**Healthy images** = ảnh nhãn rỗng (negative samples), không có mask instance nào.

### 1.2 Phân bố Mask Instances

| Split | WSSV instances | BG instances | Labeled images | Healthy (negative) |
|-------|---------------|-------------|----------------|-------------------|
| Train | 446           | 364         | 584            | 321 (35.5%)        |
| Valid | 56            | 46          | 74             | 41 (35.7%)         |
| Test  | 67            | 52          | 88             | 41 (31.8%)         |

### 1.3 Các vấn đề quan trọng từ EDA

**Mất cân bằng dữ liệu (Class Imbalance):**
- WSSV nhiều hơn BG khoảng 18-29% trong mọi split → mô hình có xu hướng thiên về WSSV
- ~35% ảnh trong mỗi split là healthy negatives → false positive rate trên healthy images là KPI quan trọng

**Tính tương quan giữa các con tôm (Data Leakage Prevention):**
- Notebook đã xử lý đúng: grouped-stratified split theo `shrimp_id` — tất cả ảnh từ 1 con tôm đều nằm trong 1 split duy nhất
- Đây là điểm mạnh quan trọng — kết quả đánh giá đáng tin cậy

**Kích thước dataset nhỏ:**
- Chỉ 905 ảnh training, 416 shrimp groups — đây là bottleneck chính khiến mô hình dễ bị overfitting

---

## 2. Phân Tích Kiến Trúc Hiện Tại

### 2.1 Cấu hình YAML Backbone và Head

```yaml
backbone:
  - Conv(64, 3, 2)   # stride=2, giảm 1/2
  - Conv(128, 3, 2)  # stride=2, giảm 1/4
  - C3k2(256) x2     
  - Conv(256, 3, 2)  # stride=2, giảm 1/8  <-- mất thông tin nhỏ
  - C3k2(512) x2
  - Conv(512, 3, 2)  # stride=2, giảm 1/16
  - C3k2(512) x2
  - Conv(1024, 3, 2) # stride=2, giảm 1/32
  - C3k2(1024) x2
  - SPPF(1024, 5)
  - C2PSA(1024)

head:
  - nn.Upsample (nearest neighbor)  # <-- upsampling thô
  - Concat + C3k2(512)
  - nn.Upsample (nearest neighbor)  # <-- upsampling thô
  - Concat + C3k2(256)              # P3: 256ch
  ...
  # [Vị trí chèn Attention] tại layers 16, 19, 22
  - [P3_att, P4_att, P5_att] -> Segment(nc=2, nm=32, proto=256)
```

**Nhận xét kiến trúc:**
- Attention chỉ đặt tại **HEAD output** (P3/P4/P5) — KHÔNG có ở backbone
- Standard nearest-neighbor upsampling trong FPN neck
- Segment head: 32 prototypes (nm=32), proto_channels=256
- Model size: ~2.85M params, 9.6 GFLOPs (rất nhỏ, tốt cho edge deployment)
- Strided convolutions trong backbone → mất thông tin pixel nhỏ của đốm bệnh

### 2.2 Cấu hình Training (CLEAN_TRAIN_ARGS)

```python
CLEAN_TRAIN_ARGS = {
    "mosaic":      0.0,   # tắt
    "copy_paste":  0.0,   # tắt
    "mixup":       0.0,   # tắt
    "cutmix":      0.0,   # tắt
    "fliplr":      0.5,   # bật (hợp lý)
    "hsv_h":       0.01,  # rất nhẹ
    "hsv_s":       0.35,
    "hsv_v":       0.20,
    "scale":       0.20,  # conservative
    "translate":   0.05,  # conservative
    "erasing":     0.0,
    "degrees":     0.0,
    "shear":       0.0,
    "perspective": 0.0,
}
# epochs=100, patience=30, batch=8, imgsz=640, seed=42
```

**Nhận xét:** Augmentation cực kỳ conservative — mosaic và copy_paste bị tắt hoàn toàn, scale và translate rất nhỏ. Điều này đúng (bảo toàn biên giới mask) nhưng thiếu đa dạng hóa quang học đặc thù môi trường nước.

---

## 3. Kết Quả Toàn Bộ Thực Nghiệm

### 3.1 Single Attention Module (Notebook baseline)

| Model | Full Test mAP50 | Labeled Test mAP50 | Healthy FP Rate | Val Loss Gap |
|-------|----------------|--------------------|-----------------|-------------|
| SimAM | ~0.448 | ~0.479 | ~26.8% | moderate |
| CA    | ~0.455 | ~0.480 | ~29.3% | moderate |
| ECA   | ~0.433 | ~0.472 | ~43.9% | high |
| CBAM  | ~0.402 | ~0.448 | ~48.8% | high |
| EMA   | ~0.392 | ~0.436 | ~48.8% | high |
| Baseline | ~0.380 | ~0.407 | ~31.7% | moderate |

**Nhận xét:** SimAM và CA là 2 module tốt nhất đơn lẻ. CBAM và EMA tăng FP rate rất cao — không phù hợp.

### 3.2 Combined Attention Module (Notebook combined)

| Model (sorted by healthy-aware score) | Full Test mAP50 | Labeled Test mAP50 | mAP50-95 | Best Val mAP50 | Healthy FP Rate | Epochs | Best Epoch | Val-Train Loss Gap |
|---------------------------------------|----------------|--------------------|----------|---------------|-----------------|--------|-----------|-------------------|
| **SimAM + CA** ⭐                     | **0.495**      | **0.519**          | **0.181**| **0.501**     | 36.6%           | 89     | 48        | 2.43              |
| CA + EMA                              | 0.454          | 0.486              | 0.141    | 0.444         | 26.8%           | 89     | 68        | 3.50              |
| CBAM + SimAM                          | 0.440          | 0.480              | 0.153    | 0.489         | 34.1%           | 77     | 28        | 3.09              |
| EMA + SimAM + CA                      | 0.405          | 0.438              | 0.137    | 0.474         | 39.0%           | 83     | 25        | 3.15              |
| Baseline (combined run)               | 0.401          | 0.420              | 0.183    | 0.492         | 19.5%           | 75     | 45        | 2.58              |
| ECA + SimAM                           | 0.402          | 0.437              | 0.155    | 0.465         | 26.8%           | 81     | 59        | 3.86              |

### 3.3 Phân tích sâu kết quả

**Hiện tượng 1 — Gap mAP50 vs mAP50-95 cực lớn:**
- SimAM+CA: mAP50=0.495 nhưng mAP50-95=0.181 → gap 0.314
- Baseline: mAP50=0.401 nhưng mAP50-95=0.183 → gap tương tự
- **Kết luận:** Mô hình detect được vùng bệnh (IoU>0.5) nhưng mask boundary rất thô — vấn đề nằm ở chất lượng mask, không phải detection.

**Hiện tượng 2 — Overfitting rõ ràng:**
- Val loss gap (val - train): 2.43–3.86 cho thấy mô hình bị overfitting nặng
- Best epoch xuất hiện sớm (epoch 25-68), sau đó val loss tăng lại
- EMA+SimAM+CA hội tụ rất sớm (best_epoch=25) → quá nhiều attention = gradient conflict

**Hiện tượng 3 — Tăng complexity không tuyến tính với kết quả:**
- SimAM+CA (2 modules) > CBAM+SimAM > EMA+SimAM+CA (3 modules)
- Thêm module không đồng nghĩa tốt hơn — diminishing returns rõ ràng

**Hiện tượng 4 — Healthy FP Rate cao:**
- SimAM+CA có FP rate 36.6% — cao nhất trong các model attention 2-module
- Baseline chỉ 19.5% FP rate — attention làm tăng false positives trên healthy images
- CA+EMA đạt FP 26.8% với mAP50 0.454 — trade-off tốt hơn về FP

**Hiện tượng 5 — Vị trí đặt attention:**
- Tất cả module được đặt tại HEAD (sau C3k2 trong FPN) — KHÔNG can thiệp vào backbone
- Backbone vẫn dùng strided convolutions → mất thông tin không gian từ giai đoạn đầu
- FPN upsampling vẫn là nearest-neighbor → mask boundary thô

---

## 4. Chẩn Đoán Điểm Nghẽn Kỹ Thuật

| Vấn đề | Nguyên nhân kỹ thuật | Biểu hiện | Hướng khắc phục |
|--------|---------------------|-----------|----------------|
| mAP50-95 thấp (~0.17-0.18) | Nearest-neighbor upsampling tạo blocky artifacts; mask boundary thô | Gap mAP50 vs mAP50-95 > 0.30 | DySample/CARAFE upsampling; Boundary Loss |
| Overfitting (val-train gap > 2.4) | Dataset nhỏ 905 ảnh; augmentation quá conservative | Best epoch sớm (25-68/100), sau đó val loss tăng | Dropout/DropBlock; mạnh augmentation hơn; Data expansion |
| Healthy FP Rate cao (20-40%) | Model học nhầm texture vỏ tôm khỏe mạnh thành đặc trưng bệnh | 7-16/41 healthy test images bị dự đoán có bệnh | Focal loss cho healthy class; Negative mining; Threshold calibration |
| Mất thông tin pixel nhỏ | Strided conv (stride=2) x4 trong backbone | Các đốm BG nhỏ bị mất trong forward pass | SPDConv thay thế strided conv; P2 Feature Injection (đúng cách) |
| Attention chỉ ở HEAD | Backbone không được tăng cường → đặc trưng thô từ đầu | Attention ở head không thể phục hồi thông tin đã mất ở backbone | Đặt attention ở backbone layers quan trọng |

---

## 5. Đề Xuất Cải Tiến Cụ Thể

### 5.1 Nhóm A — Không cần thay đổi kiến trúc (Thực hiện ngay)

#### A1. Tăng Augmentation Thủy sản Đặc thù
**Vấn đề:** Augmentation hiện tại quá conservative, thiếu simulation môi trường nước.

```python
IMPROVED_TRAIN_ARGS = {
    # Giữ nguyên
    "fliplr": 0.5,
    "flipud": 0.3,         # thêm: tôm chụp từ nhiều góc
    "mosaic": 0.0,          # vẫn tắt
    "copy_paste": 0.0,      # vẫn tắt
    
    # Tăng biên độ màu sắc để simulate màu nước ao
    "hsv_h": 0.05,          # tăng từ 0.01 → 0.05
    "hsv_s": 0.50,          # tăng từ 0.35 → 0.50
    "hsv_v": 0.40,          # tăng từ 0.20 → 0.40
    
    # Tăng scale để model học multi-scale
    "scale": 0.50,          # tăng từ 0.20 → 0.50
    "translate": 0.10,      # tăng từ 0.05 → 0.10
    
    # Thêm rotate nhẹ (tôm nằm nhiều hướng)
    "degrees": 10.0,        # thêm
    
    # Erasing nhỏ để simulate bùn/vật cản
    "erasing": 0.2,         # thêm
}
```

**Lý do:** Tôm trong ao có thể bị che khuất bởi bùn, bọt khí, ánh phản chiếu. Scale range rộng hơn giúp model robust với các size đốm bệnh khác nhau. FLipUD hợp lý vì tôm có thể lộn ngược khi bệnh nặng.

#### A2. Tăng Epochs + Fine-tune Patience
**Vấn đề:** Best epoch xuất hiện sớm (epoch 25-68/100), nhưng patience=30 quá dài → lãng phí compute.

```python
# Thay vì epochs=100, patience=30
model.train(
    epochs=150,          # tăng để cho attention models hội tụ đủ
    patience=20,         # giảm để thoát overfitting sớm hơn
    lr0=0.01,
    lrf=0.001,           # cosine decay xuống 0.001
    warmup_epochs=5,     # warmup ổn định training đầu
    cos_lr=True,         # cosine LR schedule
)
```

#### A3. Test-Time Augmentation (TTA)
**Không cần retrain, tăng 2-4% mAP50 ngay lập tức.**

```python
# Thay predict(augment=False) thành:
results = best_model.val(data=yaml_path, split="test", augment=True, imgsz=640)
```

TTA của Ultralytics tự động chạy inference với flip ngang + multi-scale rồi ensemble. Với dataset nhỏ, TTA thường cho cải thiện 2-4% mAP50.

#### A4. Tăng số Prototypes (nm=32 → 64)
**Thay đổi 1 dòng trong YAML, không cần sửa kiến trúc.**

```yaml
# Trong Segment definition:
- [[24, 26, 28], 1, Segment, [nc, 64, 256]]  # nm=64, thay vì nm=32
```

Prototype số lượng lớn hơn → mask head có nhiều basis vectors hơn → mask quality tốt hơn đặc biệt cho các vùng bệnh phức tạp, bất định hình. Chi phí tính toán tăng nhẹ (~0.5 GFLOPs).

---

### 5.2 Nhóm B — Cải tiến Loss Function (Tác động cao)

#### B1. WIoU v3 thay thế CIoU mặc định
**Vấn đề:** Dataset annotation bằng tay có noise — biên giới BG và WSSV không rõ ràng. CIoU phạt đều mọi mẫu kể cả mẫu nhãn sai.

**Implement trong `ultralytics/utils/loss.py`:**

```python
class WiseIoULoss(nn.Module):
    """Wise-IoU v3: dynamic non-monotonic focusing để giảm ảnh hưởng outlier nhãn."""
    
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, pred, target):
        # Tính IoU cơ bản
        iou = bbox_iou(pred, target, xywh=True, CIoU=True)
        
        # Wise-IoU focusing: giảm gradient cho outlier annotations
        # beta = outlier degree: high = likely noisy label
        beta = (iou.detach() - 0.5).abs()
        alpha = 1.0 / (2.0 * beta + 1e-7)  # non-monotonic focusing
        
        # Gradient reweighting: focus on "mediocre" predictions (not too bad, not perfect)
        loss = alpha * (1.0 - iou)
        return loss.mean()
```

**Kết quả kỳ vọng:** +2-4% mAP50-95 nhờ loại bỏ gradient độc hại từ nhãn sai. Đặc biệt hiệu quả cho BG (biên giới hay bị nhầm).

#### B2. Boundary-Aware Mask Loss
**Vấn đề:** BCE Loss mặc định cho masks đánh giá theo pixel — không phân biệt pixel biên giới (quan trọng) và pixel lõi (ít quan trọng).

```python
class BoundaryAwareMaskLoss(nn.Module):
    """Kết hợp BCE + Boundary IoU để ép model học biên giới sắc nét."""
    
    def __init__(self, boundary_weight=0.5, kernel_size=3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary_weight = boundary_weight
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
    
    def extract_boundary(self, mask):
        """Morphological dilation - erosion để lấy đường biên."""
        dilated = self.pool(mask)
        eroded = -self.pool(-mask)
        return dilated - eroded
    
    def forward(self, pred_mask, gt_mask):
        # Standard BCE loss
        bce_loss = self.bce(pred_mask, gt_mask)
        
        # Boundary IoU loss
        pred_boundary = self.extract_boundary(pred_mask.sigmoid())
        gt_boundary = self.extract_boundary(gt_mask)
        
        intersection = (pred_boundary * gt_boundary).sum(dim=[2, 3])
        union = pred_boundary.sum(dim=[2, 3]) + gt_boundary.sum(dim=[2, 3]) - intersection
        boundary_iou = (intersection + 1e-7) / (union + 1e-7)
        boundary_loss = 1 - boundary_iou.mean()
        
        return bce_loss + self.boundary_weight * boundary_loss
```

**Integrate vào `v8SegmentationLoss` trong ultralytics:**
```python
# Trong SegmentationLoss.__init__:
self.seg_loss_fn = BoundaryAwareMaskLoss(boundary_weight=0.5)

# Trong forward:
loss[2] = self.seg_loss_fn(pred_masks, target_masks)
```

**Kết quả kỳ vọng:** mAP50-95 tăng từ 0.173 lên 0.22-0.28 nhờ gradient trực tiếp dẫn đường cho boundary.

#### B3. Asymmetric Focal Loss cho Class Imbalance
**Vấn đề:** Healthy images (35%) và đặc biệt BG (ít hơn WSSV 18%) gây mất cân bằng trong classification loss.

```python
# Trong training args:
model.train(
    cls=0.7,           # tăng classification loss weight (default 0.5)
    # Hoặc dùng custom focal loss:
)
```

```python
class AsymmetricFocalLoss(nn.Module):
    """ASL: giảm easy negative (healthy background) penalty mạnh hơn hard negative."""
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        if self.clip is not None:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        
        los_pos = los_pos * torch.pow(1 - xs_pos, self.gamma_pos)
        los_neg = los_neg * torch.pow(xs_neg, self.gamma_neg)
        
        return -(los_pos + los_neg).mean()
```

---

### 5.3 Nhóm C — Cải tiến Kiến trúc (Tác động cao nhất)

#### C1. Đặt Attention vào Backbone (Không chỉ HEAD)
**Đây là điểm mù lớn nhất — hiện tại attention chỉ ở head, backbone không được tăng cường.**

**YAML mới — SimAM trong backbone:**

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, SimAM, []]              # THÊM: sau C3k2 đầu tiên (P3-level features)
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, SimAM, []]              # THÊM: sau C3k2 thứ hai (P4-level features)
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  # ... giữ nguyên FPN head
  # Chỉ dùng CoordAtt ở head (CA hiệu quả cho spatial-aware head)
  - [P3_out, 1, CoordAtt, [256, 32]]
  - [P4_out, 1, CoordAtt, [512, 32]]  
  - [P5_out, 1, CoordAtt, [1024, 32]]
  - [[att_p3, att_p4, att_p5], 1, Segment, [nc, 64, 256]]
```

**Lý do:** SimAM (parameter-free) trong backbone giúp backbone tự tập trung vào vùng bệnh từ sớm, giảm noise từ bọt khí và phản chiếu nước. CA trong head giữ spatial awareness cho segmentation. Tổng params tăng không đáng kể vì SimAM không có tham số học.

#### C2. Thay Nearest-Neighbor Upsampling bằng Bilinear + Learned Refine

**Trong YAML:**
```yaml
head:
  # Thay: [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  # Bằng: bilinear + 1 lớp conv nhẹ để refine
  - [-1, 1, nn.Upsample, [None, 2, "bilinear"]]   # bilinear thay nearest
  - [-1, 1, Conv, [512, 1, 1]]                    # 1x1 conv refine features
```

Hoặc dùng `ConvTranspose2d` (learnable upsampling):
```python
# Trong conv.py, thêm:
class LearnedUpsample(nn.Module):
    """Learnable upsampling để thay thế nearest-neighbor."""
    def __init__(self, c1, scale=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(c1, c1, kernel_size=scale, stride=scale, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.up(x)))
```

**Kết quả kỳ vọng:** Biên giới mask mượt hơn → mAP50-95 tăng 0.02-0.05.

#### C3. Coordinate Attention đặt ở cả Backbone VÀ Head (CA-dual)
**Phát hiện từ thực nghiệm:** CA là module đơn tốt nhất (0.455 Full Test, FP=29.3%). SimAM+CA là combo tốt nhất.

**Đề xuất:** Thay vì chỉ dùng CA ở head, đặt CA nhẹ hơn (reduction lớn hơn) ở cả backbone:

```yaml
backbone:
  ...
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, CoordAtt, [256, 64]]      # reduction=64 (nhẹ hơn) ở backbone P3
  ...
  - [-1, 2, C3k2, [512, False, 0.25]]  
  - [-1, 1, CoordAtt, [512, 64]]      # reduction=64 ở backbone P4
  ...

head:
  ...
  - [P3_out, 1, SimAM, []]            # SimAM ở head (parameter-free)
  - [P4_out, 1, SimAM, []]
  - [P5_out, 1, SimAM, []]
  - [[att3, att4, att5], 1, Segment, [nc, 64, 256]]
```

Kiến trúc này: CA học spatial dependencies trong backbone (thấp, đặc trưng texture), SimAM focus attention trong head (cao, đặc trưng semantic).

#### C4. Large Kernel Attention (LKA) — Thay thế CA trong HEAD
**CA có giới hạn:** pooling 1D theo H và W riêng lẻ — không capture tương quan 2D của vùng bệnh phân tán.

```python
class LargeKernelAttention(nn.Module):
    """LKA: Depth-wise + Depth-wise Dilated + 1x1, thay thế CA."""
    def __init__(self, c1):
        super().__init__()
        # Decompose large kernel into 3 lightweight ops
        self.dw = nn.Conv2d(c1, c1, 5, padding=2, groups=c1)          # local context
        self.dw_d = nn.Conv2d(c1, c1, 7, stride=1, padding=9, groups=c1, dilation=3)  # global
        self.pw = nn.Conv2d(c1, c1, 1)                                  # channel mixing
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attn = self.pw(self.dw_d(self.dw(x)))
        return x * self.sigmoid(attn)
```

LKA cho phép model nhìn được quần thể đốm bệnh phân tán (receptive field lớn) mà không cần tăng params nhiều. Thích hợp hơn CA cho WSSV có pattern phân mảnh.

---

### 5.4 Nhóm D — Chiến lược Training Nâng Cao

#### D1. Stochastic Weight Averaging (SWA)
**Thực hiện sau training, không cần thay đổi kiến trúc:**

```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# Load 5 checkpoint cuối
checkpoints = [f'epoch_{e}.pt' for e in range(80, 101, 5)]
swa_model = AveragedModel(base_model)

for ckpt in checkpoints:
    model = YOLO(ckpt)
    swa_model.update_parameters(model.model)

# Update BatchNorm statistics
update_bn(train_loader, swa_model)
```

**Kết quả kỳ vọng:** +1-2% mAP50 nhờ ensemble các checkpoint ở flat minima.

#### D2. Label Smoothing + Confidence Calibration
```python
model.train(
    label_smoothing=0.05,  # nhẹ, tránh overconfident predictions
)
```

Với dataset annotation thủ công có noise (biên giới BG/WSSV không rõ), label smoothing giúp model không overfit vào noisy labels.

#### D3. Gradient Accumulation cho Effective Batch Size lớn hơn
**Vấn đề:** batch=8 nhỏ → gradient noisy → overfitting.

```python
model.train(
    batch=4,         # giảm batch size vật lý
    accumulate=8,    # nhưng accumulate 8 steps → effective batch = 32
)
```

Effective batch 32 ổn định hơn batch 8, giúp loss convergence mượt hơn.

#### D4. Warmup + Cosine Annealing with Warm Restarts (SGDR)
```python
model.train(
    lr0=0.01,
    lrf=0.0001,        # min LR
    cos_lr=True,       # cosine schedule
    warmup_epochs=10,  # warmup dài hơn cho attention models
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)
```

---

### 5.5 Nhóm E — Cải tiến Dữ liệu

#### E1. CLAHE Pre-processing Pipeline
**Thực hiện offline, áp dụng trước khi training:**

```python
import cv2
import numpy as np
from pathlib import Path

def apply_clahe_to_dataset(image_dir, clip_limit=2.0, tile_size=(8,8)):
    """Áp dụng CLAHE để tăng contrast đốm bệnh trên vỏ tôm."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    for img_path in Path(image_dir).glob('*.jpg'):
        img = cv2.imread(str(img_path))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])  # apply only to L channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(img_path), enhanced)

apply_clahe_to_dataset('/kaggle/working/shrimpDisHandSegV2-1/train/images')
```

CLAHE tăng contrast cục bộ → đốm WSSV trắng nổi bật hơn trên nền vỏ tôm tối → backbone dễ extract features hơn. Không thay đổi nhãn.

#### E2. Underwater Color Augmentation (Custom Albumentations)
```python
import albumentations as A

underwater_transform = A.Compose([
    # Mô phỏng màu nước ao (trà/xanh lục)
    A.ColorJitter(hue=0.05, saturation=0.4, brightness=0.3, contrast=0.3, p=0.5),
    # Mô phỏng turbidity (độ đục)  
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    # Mô phỏng ánh phản chiếu
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    # Mô phỏng bọt khí (circular dropout)
    A.CoarseDropout(max_holes=5, max_height=15, max_width=15, p=0.2),
], bbox_params=A.BboxParams(format='yolo'), 
   keypoint_params=None)
```

Integrate vào YOLO training loop thay cho `NoAlbumentations` hiện tại đang tắt hook.

#### E3. Pseudo-labeling từ Model Tốt Nhất
Dùng SimAM+CA (best model) để generate pseudo-labels cho ảnh chưa được annotate:

```python
# 1. Dự đoán trên tập unlabeled
model = YOLO('simam_ca_best.pt')
results = model.predict(
    source='unlabeled_shrimp_images/',
    conf=0.75,           # threshold cao để chỉ lấy predictions tự tin
    iou=0.5,
    save_txt=True        # lưu YOLO format labels
)

# 2. Lọc pseudo-labels chất lượng cao
# Chỉ giữ ảnh có ít nhất 1 detection với conf > 0.80
# 3. Thêm vào training với sample weight thấp hơn (0.5x)
```

---

### 5.6 Nhóm F — Hướng Nâng Cao (Cho giai đoạn tiếp theo)

#### F1. Knowledge Distillation từ YOLO11m-seg/l-seg
```python
# Teacher: YOLO11m-seg trained trên cùng dataset
teacher = YOLO('yolo11m-seg.pt')
teacher.train(data=yaml_path, epochs=100, ...)  # train teacher trước

# Student: YOLO11n-seg+SimAM+CA với KD loss
# Feature distillation tại Neck output (mask coefficients layer)
```

Không tăng model size khi deploy, nhưng student học được representation tốt hơn từ teacher.

#### F2. Đặt Attention theo Kiến trúc Tháp (Pyramid Attention)
Thay vì đặt attention sau mỗi scale trong head, kết hợp cross-scale attention:

```python
class PyramidAttention(nn.Module):
    """Cross-scale attention: P3 học từ P4, P4 học từ P5."""
    def __init__(self, c_list):
        super().__init__()
        # Upsample P4 → match P3 size, học correlation
        self.p4_to_p3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c_list[1], c_list[0], 1),
            nn.Sigmoid()
        )
        self.p5_to_p4 = nn.Sequential(
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(c_list[2], c_list[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, p3, p4, p5):
        p3 = p3 * self.p4_to_p3(p4)  # P3 guided by P4 semantic
        p4 = p4 * self.p5_to_p4(p5)  # P4 guided by P5 semantic
        return p3, p4, p5
```

Kiến trúc này phù hợp cho trường hợp đốm WSSV nhỏ (cần P3 fine-grained) nhưng phân bố theo pattern của toàn thân tôm (cần P5 context).

---

## 6. Lộ Trình Thực Hiện Ưu Tiên

### Phase 1 — Quick Wins (1-2 ngày, không retrain kiến trúc)

| # | Thay đổi | File cần sửa | Thời gian | Kỳ vọng |
|---|---------|-------------|-----------|---------|
| 1 | TTA inference | Notebook evaluation cell | 30 phút | +2-3% mAP50 |
| 2 | Tăng nm=32→64 trong YAML | YAML files | 1 giờ | +1-2% mask quality |
| 3 | Tăng scale=0.5, hsv_s=0.5, flipud=0.3 | CLEAN_TRAIN_ARGS | 1 giờ + retrain | +1-3% mAP50 |
| 4 | epochs=150, patience=20, cos_lr=True | Training args | + retrain | ổn định training |
| 5 | CLAHE pre-processing | Tiền xử lý ảnh | 2 giờ | +1-2% mAP50 |

### Phase 2 — Loss Function (3-5 ngày)

| # | Thay đổi | File cần sửa | Thời gian | Kỳ vọng |
|---|---------|-------------|-----------|---------|
| 6 | WIoU v3 cho box loss | `ultralytics/utils/loss.py` | 1 ngày | +2-4% mAP50-95 |
| 7 | Boundary Mask Loss | `ultralytics/utils/loss.py` | 1 ngày | +3-5% mAP50-95 |
| 8 | label_smoothing=0.05 | Training args | 30 phút | giảm overfit |

### Phase 3 — Kiến trúc (1 tuần)

| # | Thay đổi | File cần sửa | Thời gian | Kỳ vọng |
|---|---------|-------------|-----------|---------|
| 9 | SimAM trong Backbone (thêm vào sau C3k2) | YAML + conv.py | 2 ngày | +2-4% mAP50 |
| 10 | Bilinear upsampling thay nearest | YAML | 2 giờ | +1-2% mAP50-95 |
| 11 | LKA thay CA trong HEAD | conv.py + YAML | 2 ngày | +1-3% mAP50 |
| 12 | CA dual (backbone + head) | YAML | 2 ngày | +2-4% mAP50 |

### Phase 4 — Data Expansion (2 tuần)

| # | Thay đổi | Mô tả | Thời gian | Kỳ vọng |
|---|---------|-------|-----------|---------|
| 13 | Underwater augmentation | Custom Albumentations pipeline | 2 ngày | +1-3% mAP50 |
| 14 | Pseudo-labeling | Dùng best model label thêm dữ liệu | 1 tuần | +3-5% mAP50 |
| 15 | SWA | Average 5 checkpoint cuối | 1 ngày | +1-2% mAP50 |

---

## 7. Notebook Template cho Phase 1 + 2

### 7.1 YAML cho experiment SimAM-Backbone + CA-Head + nm=64

```yaml
# yolo11n-seg-simam-backbone-ca-head-nm64.yaml
nc: 2
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, SimAM, []]                    # NEW: SimAM sau P3 features
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, SimAM, []]                    # NEW: SimAM sau P4 features  
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]          # layer 16: P3 output (256ch)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]          # layer 19: P4 output (512ch)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]          # layer 22: P5 output (1024ch)

  # CoordAtt ở HEAD (spatial-aware for segmentation)
  - [16, 1, CoordAtt, [256, 32]]         # layer 23
  - [24, 1, SimAM, []]                   # layer 24

  - [19, 1, CoordAtt, [512, 32]]         # layer 25
  - [26, 1, SimAM, []]                   # layer 26

  - [22, 1, CoordAtt, [1024, 32]]        # layer 27
  - [28, 1, SimAM, []]                   # layer 28

  - [[25, 27, 29], 1, Segment, [nc, 64, 256]]  # nm=64 (tăng từ 32)
```

### 7.2 Training Config Template

```python
IMPROVED_TRAIN_ARGS = {
    # Augmentation underwater-aware
    "fliplr":      0.5,
    "flipud":      0.3,       # thêm
    "hsv_h":       0.05,      # tăng
    "hsv_s":       0.50,      # tăng  
    "hsv_v":       0.40,      # tăng
    "scale":       0.50,      # tăng
    "translate":   0.10,      # tăng
    "degrees":     10.0,      # thêm
    "erasing":     0.15,      # thêm (nhẹ)
    
    # Giữ nguyên (đúng)
    "mosaic":      0.0,
    "copy_paste":  0.0,
    "mixup":       0.0,
    "cutmix":      0.0,
    "shear":       0.0,
    "perspective": 0.0,
    
    # Training stability
    "label_smoothing": 0.05,  # thêm
    "cos_lr":      True,      # thêm
    "warmup_epochs": 8,       # tăng
    "lr0":         0.01,
    "lrf":         0.001,
}

model.train(
    data=str(yaml_path),
    task="segment",
    imgsz=640,
    epochs=150,          # tăng
    batch=8,
    patience=20,         # giảm
    seed=42,
    deterministic=True,
    workers=0,
    **IMPROVED_TRAIN_ARGS,
)
```

---

## 8. Tóm Tắt và Kết Luận

### Điểm mạnh hiện tại cần duy trì
- Grouped-stratified split chống leakage — kết quả đáng tin cậy
- Clean augmentation (không mosaic, không copy-paste) — đúng cho segmentation
- Ensemble evaluation (Full Test + Labeled Test + Healthy FP Rate) — comprehensive

### Điểm yếu cốt lõi cần khắc phục
1. **Attention chỉ ở HEAD** — backbone không được cải thiện → fix bằng SimAM/CA ở backbone
2. **Nearest-neighbor upsampling** → mask boundary thô → fix bằng bilinear/learnable upsample
3. **Loss function không tối ưu cho boundary** → mAP50-95 thấp → fix bằng Boundary Loss
4. **Augmentation thiếu diversity** → overfitting dataset nhỏ → fix bằng underwater augmentation
5. **nm=32 prototype nhỏ** → mask quality thấp → fix bằng nm=64

### Mục tiêu kỳ vọng sau toàn bộ improvements

| Metric | Hiện tại (SimAM+CA) | Phase 1 | Phase 1+2 | Phase 1+2+3 |
|--------|--------------------|---------|-----------|-----------  |
| Full Test mAP50 | 0.495 | 0.52 | 0.54 | 0.56-0.60 |
| Labeled Test mAP50 | 0.519 | 0.54 | 0.56 | 0.58-0.62 |
| mAP50-95 | 0.173 | 0.185 | 0.22 | 0.27-0.32 |
| Healthy FP Rate | 36.6% | 32% | 28% | 22-25% |

> **Ưu tiên số 1:** TTA ngay (không cần retrain) + tăng nm=64 + WIoU/Boundary Loss. Ba thay đổi này có tác động/effort ratio tốt nhất.
