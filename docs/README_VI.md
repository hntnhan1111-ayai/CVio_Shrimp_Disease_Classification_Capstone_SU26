# Hướng dẫn nhanh

Project lưu code, weights, clean test và mobile corruption benchmark của **YOLO11s-RECSRA**.

- Clean mAP50: **16.036%**, tăng **+3.136 điểm phần trăm**.
- Clean mAP50-95: **4.657%**, tăng **+0.857 điểm phần trăm**.

```bash
git lfs install
PYTHONPATH=src python scripts/verify_repository.py
sha256sum -c checkpoints/SHA256SUMS.txt
```

Dataset không nằm trong repository. Sử dụng `configs/data/data.template.yaml`.
