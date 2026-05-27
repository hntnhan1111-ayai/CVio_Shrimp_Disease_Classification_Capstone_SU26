# Objective

Create and run a compact, reproducible ASL-based custom loss experiment for ShrimpDiseaseImageBD V3 using both ConvNeXt-Tiny/ShrimpXNet-style training and native Ultralytics YOLOv26m-cls training.

The new notebook is `final_asl_custom_losses_convnext_yolo_repeat1_compact.ipynb`. It defaults to `RUN_TRAINING=0`; full training must be enabled explicitly on the Linux RTX 4090 24GB target with `RUN_TRAINING=1`.

# Dataset and Class Mapping

Dataset: ShrimpDiseaseImageBD V3, DOI `10.17632/jhrtdj9txm.3`.

Class mapping:

| id | folder | name | expected count |
| ---: | --- | --- | ---: |
| 0 | `1. Healthy` | Healthy | 403 |
| 1 | `2. BG` | BG | 198 |
| 2 | `3. WSSV` | WSSV | 328 |
| 3 | `4. WSSV_BG` | WSSV_BG | 220 |

Fixed split policy: discover from `DATA_DIR`, sort by `rel_path`, use stratified `train_test_split` with `random_state=42`, split 70/15/15, compute MD5 for every source image, save `fixed_split_manifest_seed42_with_md5.csv`, copy the same split to YOLO classification format, compute `source_md5` and `yolo_md5`, and evaluate YOLO only from `yolo_path`.

# Prior Empirical Evidence

Completed same-seed YOLOv26m-cls all-losses reference:

| rank | loss | Macro-F1 | Kappa | Accuracy | BG_WSSV Recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | CE | 0.9051 | 0.8739 | 0.9075 | 1.0000 |
| 2 | SCE | 0.9040 | 0.8736 | 0.9075 | 1.0000 |
| 3 | LDAM | 0.8984 | 0.8651 | 0.9017 | 0.9697 |
| 4 | Co-Infection Margin ASL | 0.8929 | 0.8582 | 0.8960 | 1.0000 |
| 5 | ASLSingleLabel | 0.8854 | 0.8502 | 0.8902 | 0.9394 |

This completed YOLO reference used same-seed reproducibility, not independent-seed robustness.

Partial YOLO diagnostic from `experiment/final_yolo26m_convnext_tiny_14losses_repeat1_audited.ipynb`: YOLOv26m-cls + ASLSingleLabel reached Macro-F1 0.9109, Kappa 0.8818, Accuracy 0.9133, BG_WSSV Recall 0.9697, and WSSV to BG_WSSV count 5. This was partial and not a completed two-model benchmark.

Completed ConvNeXt-Tiny 14-loss context from the local completed final summary and user-provided context: False-CoInfection Cost CE was the best current ConvNeXt-Tiny loss with Macro-F1 0.8095, Kappa 0.7619, Accuracy 0.8266. Poly-DCS-CE was near-best. These are prior empirical signals, not final results for the new ASL custom-loss experiment.

# Why ASLSingleLabel Is Baseline, Not Proposed

ASLSingleLabel is an existing official-style ASL baseline from the Alibaba-MIIL/ASL implementation and ASL paper. It is the single-label softmax version of ASL and uses asymmetric focusing with `gamma_pos=0`, `gamma_neg=4`, and label smoothing `eps=0.1` in this project.

Do not describe ASLSingleLabel itself as a custom proposed method. The proposed variants in the new notebook are ASL-based co-infection suppression variants built on top of this existing baseline.

# Why False-Co-Infection Suppression Is The Target

The important domain error is not only moderate class imbalance. `WSSV_BG` is a co-infection class with visual overlap against both BG and WSSV. Prior YOLO results showed CE can already recall `WSSV_BG` strongly, so custom losses should avoid blindly increasing `WSSV_BG` probability and should instead suppress false `WSSV_BG` predictions for true BG and true WSSV samples.

# Loss List

The compact ASL custom-loss experiment has 16 losses.

Existing baselines and references:

| key | display name | category |
| --- | --- | --- |
| `baseline_ce` | CE | existing baseline |
| `asl_single_label` | ASLSingleLabel | existing ASL baseline |
| `false_coinfection_cost_ce` | False-CoInfection Cost CE | existing custom reference |
| `poly_dcs_ce` | Poly-DCS-CE | existing custom reference |
| `sce` | SCE | existing robust baseline |
| `ldam` | LDAM | existing margin baseline |

ASL-based custom variants:

| key | display name |
| --- | --- |
| `false_coinfection_cost_asl` | False-CoInfection Cost ASL |
| `poly_dcs_asl` | Poly-DCS-ASL |
| `pairwise_coinfection_ranking_asl` | Pairwise Co-Infection Ranking ASL |
| `confidence_gated_dcs_asl` | Confidence-Gated DCS-ASL |
| `attribute_projection_asl` | Attribute-Projection ASL |
| `sce_asl_hybrid` | SCE-ASL Hybrid |
| `ldam_asl_hybrid` | LDAM-ASL Hybrid |
| `robust_gap_asl` | Robust Gap ASL |
| `distribution_balanced_attribute_asl` | Distribution-Balanced Attribute ASL |
| `cost_poly_asl` | Cost-Poly ASL |

# Model Pipelines

ConvNeXt-Tiny follows the real `experiment/shrimpxnet.ipynb` as closely as practical:

- Model source: `torchvision.models.convnext_tiny`
- Weights: `models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
- Head: ConvNeXt features + avgpool + final norm, then `Linear(512)` style custom classifier with Dropout 0.5 and 4-class output
- Freeze backbone for warmup, then unfreeze `features[5:]` and final norm
- Image size 224
- Epochs 30
- Patience 5
- Effective batch 128, micro batch 32, accumulation 4
- Warmup Adam LR `1e-3`
- Fine-tune Adam LR `2e-5` for backbone and `1e-4` for head
- StepLR `step_size=3`, `gamma=0.9`
- Train transforms: Resize 256, RandomResizedCrop 224 scale `(0.82,1.0)` ratio `(0.90,1.10)`, horizontal flip 0.5, rotation 10, ColorJitter 0.10/0.10/0.05, ImageNet normalization
- Eval transform: torchvision pretrained weight transform

YOLOv26m-cls follows the audited native Ultralytics configuration:

- Model: `yolo26m-cls.pt`
- Native Ultralytics classification training
- Custom losses through `ClassificationModel` and `ClassificationTrainer`
- `imgsz=224`, `epochs=30`, `patience=15`, `batch=128`, `workers=8`, `optimizer="AdamW"`, `lr0=1.25e-3`, `lrf=0.01`, `cos_lr=True`, `cache=True`, `amp=True`, `seed=42`
- Checkpoint selection: Ultralytics default `best.pt`

# Output Files

Output directory: `final_asl_custom_losses_convnext_yolo_repeat1_compact_outputs`.

Required compact outputs:

- `final_summary.xlsx`
- `final_summary.csv`
- `loss_run_summary_raw.csv`
- `run_audit.json`
- `environment_versions.json`
- `fixed_split_manifest_seed42_with_md5.csv`
- `yolo_split_manifest_seed42_with_md5.csv`
- `missing_or_failed_runs.csv`
- `memory_bank_asl_custom_losses_experiment_summary.md`

No plots, ZIPs, confusion matrix PNGs, large prediction tables, or classification-report tables are required for this compact experiment.

# Known Bugs Avoided

Attribute-projection losses must not call `torch.nn.functional.binary_cross_entropy` on probability tensors under AMP. The new notebook uses manual float32 BCE on clamped projected probabilities.

Empty strings must not become `Path(".")` when collecting artifacts. The compact notebook avoids large artifact collection and records status rows directly.

Failed runs must not be counted as completed. Use `Status == "completed"` only.

Summaries must be filtered to current output directory, model keys, loss keys, seed 42, and repeat 1.

YOLO evaluation must use `yolo_path` only after MD5 verification.

Random pretrained fallback is forbidden unless `ALLOW_RANDOM_PRETRAINED_FALLBACK=1` is explicitly set and recorded.

# How To Interpret Final Summary

Primary ranking should use `Test Macro F1`, then `Cohen Kappa`, then `WSSV_BG Recall`, then fewer `WSSV to WSSV_BG` and `BG to WSSV_BG` errors.

The columns `Delta Macro F1 vs ASL`, `Delta Kappa vs ASL`, and `Delta WSSV to WSSV_BG vs ASL` are meaningful only when the same model has a completed ASLSingleLabel row. The column `Delta Macro F1 vs CE` is meaningful only when the same model has a completed CE row.

Rows with `Status` other than `completed` are not results. They are pending, disabled, or failed run records for reproducibility.

# Next Steps After Running

Run the notebook on the Linux RTX 4090 24GB target with `RUN_TRAINING=1`. Inspect `final_summary.xlsx` and `missing_or_failed_runs.csv`. If all 32 model-loss rows complete, compare the ASL-based custom variants against ASLSingleLabel, CE, False-CoInfection Cost CE, and Poly-DCS-CE within each model. Update README only with results that appear in `final_summary.xlsx` or `final_summary.csv`.
