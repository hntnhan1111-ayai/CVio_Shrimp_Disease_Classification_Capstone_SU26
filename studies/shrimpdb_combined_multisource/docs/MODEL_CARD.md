# Model Card

This model card documents the **yolo26m_asl_ldam_simam_dcfr_combined4_best.pt** checkpoint
produced by the **shrimpdb_combined_multisource** study.

## 1. Model Details

| Attribute | Value |
|---|---|
| Model name | YOLO26m-cls + ASL-LDAM + SimAM-DCFR |
| Checkpoint | `artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt` |
| SHA-256 | `9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606` |
| Architecture | YOLO26m-cls |
| Loss | ASL-LDAM (gamma_pos=0.0, gamma_neg=4.0, label_smoothing=0.1, ldam_max_m=0.5, ldam_scale=30.0) |
| Attention | SimAM-DCFR (e_lambda=0.0001) |
| Training data | ShrimpDB-3 (315 images) + ShrimpDiseaseDB (1,149 images) = 1,464 images |
| Test data | 220 images (Combined-4 test split) |
| Classes | Healthy, BG, WSSV, WSSV_BG |
| Input | RGB, 224 x 224 |
| Output | 4-class logits |
| Seed | 42 |
| Framework | PyTorch 2.10.0+cu128, Ultralytics 8.4.75 |

## 2. Intended Use

This model is a **research prototype** intended for:

- Academic studies on class-imbalanced shrimp disease image classification
- Benchmarking of ASL-LDAM loss and SimAM-DCFR attention mechanisms
- Evaluation of multi-source training protocols under image-level splits

**This model is not intended for clinical veterinary diagnosis, aquaculture management
decisions, or any safety-critical application.** The model was trained and evaluated on
research datasets with limited clinical validation.

## 3. Performance Characteristics

The model was evaluated on the Combined-4 test split (220 images) under the following
conditions:

| Metric | Value |
|---|---:|
| Accuracy | 83.18% |
| Balanced accuracy | 83.85% |
| Macro-F1 | 82.18% |
| Weighted-F1 | 83.19% |
| Cohen's kappa | 77.15% |
| MCC | 77.51% |
| Top-2 accuracy | 95.45% |
| ECE (15 bins) | 33.46% |

### 3.1 Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Healthy | 88.00% | 91.67% | 89.80% | 72 |
| BG | 80.85% | 84.44% | 82.61% | 45 |
| WSSV | 90.91% | 71.43% | 80.00% | 70 |
| WSSV_BG | 67.44% | 87.88% | 76.32% | 33 |

### 3.2 Source-Domain Performance

| Source Dataset | Images | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| ShrimpDB | 47 | 85.11% | 87.92% | 64.77% |
| ShrimpDiseaseDB | 173 | 82.66% | 82.34% | 81.95% |

## 4. Limitations

- **Single seed only**: All metrics are from a single fixed-seed (42) run. No mean ± standard
  deviation or statistical significance testing is reported.
- **Small test set**: The Combined-4 test set contains 220 images. Confidence intervals are
  wide (see bootstrap 95% CI in `artifacts/evaluation/combined4/best_pt/metrics_raw.json`).
- **Image-level split**: The split is image-level. Specimen or animal identity was not
  established, so images from the same specimen may appear in both training and test sets.
- **Calibration**: Expected calibration error (ECE) is 33.46%, indicating that the model's
  predicted probabilities are not well calibrated. The model should not be used where
  calibrated probabilities are required.
- **Class space mismatch**: The model was trained on four classes. If applied to data that
  includes only a subset of classes (e.g., the original ShrimpDB three-class space), the
  `WSSV_BG` class has zero support and its presence in the output reduces the standard
  macro-averaged metrics.
- **Domain shift**: Performance on the ShrimpDB test set is lower when evaluated through the
  Combined-4 model compared to the ShrimpDB-only model, suggesting domain shift effects.
- **No external validation**: Performance on datasets beyond ShrimpDB and ShrimpDiseaseDB has
  not been evaluated.

## 5. Ethical Considerations

- This model is a research prototype and is **not a veterinary diagnostic system**.
- Results should not be used to make decisions about animal health or treatment without
  independent clinical validation.
- The model may perform differently on images from devices, lighting conditions, or shrimp
  species not represented in the training data.
- Misclassification could lead to unnecessary treatment or missed disease detection if used
  outside its intended research context.

## 6. Distribution Instructions

The checkpoint file `artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt`
is the designated distribution artifact. To distribute:

1. Verify the SHA-256 hash:
   ```bash
   sha256sum artifacts/final_application_model/yolo26m_asl_ldam_simam_dcfr_combined4_best.pt
   # Expected: 9fdf51f89a531ffe1158cb5208e15640284d63f4413b649b1ccadc02b3967606
   ```
2. Include the class mapping (`artifacts/final_application_model/class_mapping.json`) with
   the checkpoint file.
3. Include this model card and the accompanying `CLAIMS_AND_LIMITATIONS.md` with any
   distributed checkpoint.
4. Do not distribute the checkpoint as a standalone file without the accompanying
   documentation and limitation notices.
