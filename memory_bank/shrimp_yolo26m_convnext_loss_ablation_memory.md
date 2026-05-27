# Project objective

Build controlled shrimp disease classification loss-ablation notebooks for ShrimpDiseaseImageBD V3, comparing native Ultralytics YOLOv26m-cls and ConvNeXt-Tiny/ShrimpXNet-style baselines under reproducible fixed splits.

The immediate next benchmark target is a ConvNeXt-Tiny 14-loss run using `experiment/final_convnext_tiny_shrimpxnet_14losses_repeat1_audited.ipynb`. Do not treat the two-model notebook as a completed benchmark until ConvNeXt-Tiny has run all 14 losses and final summary artifacts exist.

# Dataset and class mapping

Dataset: ShrimpDiseaseImageBD V3.

DOI: `10.17632/jhrtdj9txm.3`.

Class mapping:

| id | folder | name | count |
| ---: | --- | --- | ---: |
| 0 | `1. Healthy` | Healthy | 403 |
| 1 | `2. BG` | BG | 198 |
| 2 | `3. WSSV` | WSSV | 328 |
| 3 | `4. WSSV_BG` | WSSV_BG | 220 |

Core research issue: `WSSV_BG` is a co-infection class with visual overlap against BG and WSSV. Future custom co-infection losses should focus on suppressing false BG_WSSV predictions for true WSSV/BG samples, not simply boosting BG_WSSV recall.

# Fixed split policy

Use one fixed split for every controlled run:

- Sort discovered images by `rel_path` before splitting.
- Use `train_test_split` with `random_state=42`, `shuffle=True`, and stratified labels.
- Use a 70/15/15 train/val/test split.
- Expected counts for 1149 images: train 804, val 172, test 173.
- Save `fixed_split_manifest_seed42_with_md5.csv`.
- Include `rel_path`, `path`, `class_dir`, `class_name`, `label`, `split`, and `md5`.
- Assert no `rel_path` overlap across train, val, and test.
- Assert no missing image files before training.

# YOLOv26m-cls native best config

Preserve native Ultralytics classification training for YOLO. Do not rewrite YOLO as a manual PyTorch loop.

Audited YOLOv26m-cls settings:

- Model weights: `yolo26m-cls.pt`
- `imgsz=224`
- `epochs=30`
- `patience=15`
- `batch=128`
- `workers=8`
- `optimizer="AdamW"`
- `lr0=1.25e-3`
- `lrf=0.01`
- `cos_lr=True`
- `cache=True`
- `amp=True` when CUDA is available
- `seed=42`
- Checkpoint selection: Ultralytics default `best.pt`

Custom YOLO losses must be integrated through Ultralytics-compatible `ClassificationModel` and `ClassificationTrainer` subclassing.

# ConvNeXt-Tiny ShrimpXNet reference pipeline

The exact `shrimpxnet.ipynb` file was not present in this checkout during notebook generation. The ConvNeXt-Tiny pipeline was extracted from the available local ConvNeXt/ShrimpXNet references:

- `legacy/final_yolo26m_convnext_tiny_CE_ASL_5asl_losses_best_pipeline_rtx4090.ipynb`
- `best/final_yolo26m_convnext_tiny_CE_ASL_original_yolo_best_repeat3.ipynb`

Extracted ConvNeXt-Tiny settings:

- Model source: `timm`
- Model aliases: `convnext_tiny.fb_in22k`, `convnext_tiny.fb_in1k`, `convnext_tiny`
- Pretrained approach: `pretrained=True` with fallback to `pretrained=False`
- Head: timm backbone with `num_classes=0`, global average pooling when supported, then `Linear(num_features, 512)`, ReLU, Dropout 0.5, `Linear(512, 4)`
- Image size: 224
- Epochs: 30
- Patience: 15
- Paper/effective batch size: 128
- Micro batch size: 32
- Gradient accumulation: 4
- Warmup epochs: 5
- Warmup optimizer: Adam on classifier/head only, LR `1e-3`
- Fine-tune optimizer: Adam with backbone LR `2e-5` and head LR `1e-4`
- Scheduler: StepLR with `step_size=3`, `gamma=0.9`
- AMP: enabled when CUDA is available
- Gradient clipping: max norm 1.0
- Checkpoint rule: validation Macro-F1 first, validation loss as tie-breaker

Transforms:

- Train: Resize 256 bicubic, RandomResizedCrop 224 with `scale=(0.82, 1.0)` and `ratio=(0.90, 1.10)`, horizontal flip 0.5, rotation 10 degrees bicubic, ColorJitter brightness 0.10 contrast 0.10 saturation 0.05, ToTensor, ImageNet normalization.
- Eval: Resize 236 bicubic, CenterCrop 224, ToTensor, ImageNet normalization.

# 14-loss list and loss categories

The current 14-loss list:

| key | display name | category |
| --- | --- | --- |
| `baseline_ce` | CE | baseline existing |
| `sce` | SCE | baseline existing |
| `ldam` | LDAM | baseline existing |
| `asl_single_label` | ASLSingleLabel | baseline existing |
| `coinfection_margin_asl_old` | Old Co-Infection Margin ASL | baseline existing custom |
| `gce` | GCE | baseline existing |
| `dcs_ce` | Directional Co-Infection Suppression CE | new custom |
| `dcs_sce` | Directional Co-Infection Suppression SCE | new custom |
| `false_coinfection_cost_ce` | False-CoInfection Cost CE | new custom |
| `pairwise_coinfection_ranking_ce` | Pairwise Co-Infection Ranking CE | new custom |
| `confidence_gated_dcs_ce` | Confidence-Gated DCS-CE | new custom |
| `dcs_ldam` | DCS-LDAM | new custom |
| `poly_dcs_ce` | Poly-DCS-CE | new custom |
| `attribute_projection_ce` | Attribute-Projection CE | new custom |

All losses must accept logits `[B, 4]` and targets `[B]`, and each must return a finite scalar tensor in the synthetic sanity check.

ASLSingleLabel is not custom; it is an existing official-style ASL baseline.

# Completed same-seed YOLO all-losses result

Completed reference folder: `experiment/final_yolo26m_cls_all_losses_anchor_best_outputs/`.

This is a completed same-seed reproducibility run, not independent-seed robustness.

Top completed YOLOv26m-cls ranking:

| rank | loss | Macro-F1 | Kappa | Accuracy | BG_WSSV Recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | CE | 0.9051 | 0.8739 | 0.9075 | 1.0000 |
| 2 | SCE | 0.9040 | 0.8736 | 0.9075 | 1.0000 |
| 3 | LDAM | 0.8984 | 0.8651 | 0.9017 | 0.9697 |
| 4 | Co-Infection Margin ASL | 0.8929 | 0.8582 | 0.8960 | 1.0000 |
| 5 | ASLSingleLabel | 0.8854 | 0.8502 | 0.8902 | 0.9394 |

Interpretation: CE was best overall in the completed same-seed YOLO all-losses run. SCE nearly tied CE. LDAM was the strongest margin/long-tail alternative. The older co-infection margin ASL did not beat CE.

# Partial YOLOv26m-cls + ASLSingleLabel best result

Source notebook: `experiment/final_yolo26m_convnext_tiny_14losses_repeat1_audited.ipynb`.

Source CSV: `experiment/final_yolo26m_convnext_tiny_14losses_repeat1_audited_outputs/loss_run_summary_raw.csv`.

Scope: partial YOLO-only diagnostic inside the attempted two-model notebook. This is not a completed final two-model experiment.

Observed partial result:

- Model: YOLOv26m-cls
- Loss: ASLSingleLabel
- Test Macro-F1: 0.9109
- Cohen Kappa: 0.8818
- Accuracy: 0.9133
- BG_WSSV Recall: 0.9697
- WSSV to BG_WSSV count: 5

Interpretation: this is the strongest YOLOv26m-cls result observed in that partial notebook, but the notebook did not complete ConvNeXt-Tiny runs or final summary generation.

# Known failure modes

- `experiment/final_yolo26m_convnext_tiny_14losses_repeat1_audited.ipynb` crashed before ConvNeXt-Tiny runs started.
- `final_summary.xlsx`, `final_summary.json`, and `figures_and_reports.zip` were not created in that failed two-model run.
- AttributeProjectionCE failed under AMP because `torch.nn.functional.binary_cross_entropy` on probability tensors is unsafe to autocast. The ConvNeXt-only notebook uses a manual float32 BCE attribute term.
- Partial-state collection failed when empty failed-result paths became `Path(".")` and were read as files. The ConvNeXt-only notebook skips empty paths and directories.
- Do not claim a final two-model 14-loss benchmark exists until ConvNeXt-Tiny completes all 14 losses and final summary artifacts are present.

# Next actions

1. Run `experiment/final_convnext_tiny_shrimpxnet_14losses_repeat1_audited.ipynb` on Linux RTX 4090 24GB.
2. Confirm all 14 ConvNeXt-Tiny runs complete.
3. Verify `final_convnext_tiny_shrimpxnet_14losses_repeat1_audited_outputs/final_summary.xlsx`, `final_summary.json`, and `figures_and_reports.zip` exist.
4. Compare ConvNeXt-Tiny results against the completed same-seed YOLO reference and the partial YOLOv26m-cls + ASLSingleLabel diagnostic without mixing scopes.
5. If building a final two-model notebook later, use the ConvNeXt-only fixes for AttributeProjectionCE and partial-state collection.
