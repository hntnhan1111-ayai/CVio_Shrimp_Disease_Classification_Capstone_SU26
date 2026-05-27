# Project Scope

This repository contains controlled shrimp disease classification experiments for ShrimpDiseaseImageBD V3. The current research thread compares YOLOv26m-cls and ConvNeXt-Tiny/ShrimpXNet-style pipelines under fixed splits and controlled loss ablations.

# Runtime Target

Target full-training runtime is Linux, Python 3.12.2, CUDA, and RTX 4090 24GB. Local review machines must not be treated as the training target.

# Do Not Run Heavy Training Locally

Do not run full notebooks or GPU-heavy training locally. You may inspect notebook JSON, validate syntax, inspect outputs, and run lightweight sanity checks that do not launch YOLO or ConvNeXt training.

# YOLO Native Ultralytics Rules

YOLOv26m-cls must use native Ultralytics classification training. Do not rewrite YOLO as a manual PyTorch loop. Custom losses must be injected through an Ultralytics-compatible `ClassificationModel` and `ClassificationTrainer` pattern. The official Ultralytics custom trainer docs show passing a trainer class through `model.train(..., trainer=MyCustomTrainer)`, and the classification loss reference extracts logits from `preds` or `preds[1]` and labels from `batch["cls"]`.

Preserve the audited YOLOv26m-cls config unless the user explicitly changes it: `yolo26m-cls.pt`, `imgsz=224`, `epochs=30`, `patience=15`, `batch=128`, `workers=8`, `optimizer="AdamW"`, `lr0=1.25e-3`, `lrf=0.01`, `cos_lr=True`, `cache=True`, `amp=True`, `seed=42`, and Ultralytics default `best.pt` checkpoint selection.

# ConvNeXt ShrimpXNet Reference Rules

The real `experiment/shrimpxnet.ipynb` uses torchvision ConvNeXt-Tiny, not timm: `models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)`. It freezes the backbone, uses a custom classifier `Linear(num_features,512)`, ReLU, Dropout 0.5, `Linear(512,4)`, warms up the classifier, then unfreezes `features[5:]` and final norm for fine-tuning.

Reference settings extracted from `experiment/shrimpxnet.ipynb`: image size 224, epochs 30, patience 5, effective batch 128, micro batch 32, accumulation 4, eval batch 128, warmup epochs 5, warmup Adam LR `1e-3`, fine-tune Adam backbone LR `2e-5`, head LR `1e-4`, StepLR `step_size=3`, `gamma=0.9`, train transforms Resize 256, RandomResizedCrop 224 scale `(0.82,1.0)` ratio `(0.90,1.10)`, horizontal flip 0.5, rotation 10, ColorJitter 0.10/0.10/0.05, ImageNet normalization, and eval transforms from the pretrained weights.

# Dataset Split and MD5 Audit Rules

Use class folders `1. Healthy`, `2. BG`, `3. WSSV`, `4. WSSV_BG` mapped to labels 0, 1, 2, 3 and names `Healthy`, `BG`, `WSSV`, `WSSV_BG`. Discover images from `DATA_DIR`, sort by `rel_path`, split with `train_test_split(random_state=42, stratify=labels)` into 70/15/15, compute MD5 for every source image, and save `fixed_split_manifest_seed42_with_md5.csv`.

For YOLO, copy the same split into a YOLO classification folder tree, compute `source_md5` and `yolo_md5`, assert equality, save `yolo_split_manifest_seed42_with_md5.csv`, and evaluate YOLO only from `yolo_path`.

# Loss Implementation Rules

All losses in the controlled experiments must accept logits shaped `[B,4]` and targets shaped `[B]`, return a scalar tensor, and avoid sample indexes, embeddings, masks, multi-view tensors, or alternate heads. ASLSingleLabel is an existing official-style ASL baseline from Alibaba-MIIL/ASL, not a proposed method. The ASL paper and official repository describe ASL as asymmetric focusing for positive/negative terms and include a single-label softmax implementation.

Cost-sensitive learning, PolyLoss-style polynomial corrections, SCE, and LDAM are existing ideas. Any new contribution claims must be limited to dataset-specific ASL-based co-infection suppression variants unless final results and a stronger literature review justify stronger wording.

# Known Bugs To Avoid

Do not call `torch.nn.functional.binary_cross_entropy` on projected probabilities under autocast. PyTorch AMP documentation recommends logits-safe BCE variants or forcing float32 subregions; for attribute-projection losses in this repo, use manual float32 BCE on clamped probabilities.

Do not convert empty output paths into `Path(".")`. Skip empty strings, whitespace, missing files, directories, and zero-byte files when collecting artifacts.

Do not count failed rows as completed. Use `Status == "completed"` only.

Do not contaminate summaries with previous runs. Filter by current output directory, current model keys, current loss keys, seed 42, and repeat 1.

Do not silently use random pretrained weights. Random fallback must require an explicit config flag and must be recorded in the audit.

# Output Minimization Rules

For compact ASL custom-loss experiments, required deliverables are `final_summary.xlsx`, `final_summary.csv`, `loss_run_summary_raw.csv`, `run_audit.json`, `environment_versions.json`, fixed split manifest, YOLO split manifest, `missing_or_failed_runs.csv`, and the output memory summary. Do not generate plots, ZIPs, confusion matrix PNGs, large prediction tables, or excessive reports unless explicitly requested.

# README and Memory-Bank Update Rules

README and memory-bank files must keep completed references separate from partial diagnostics. Do not claim custom-loss improvement without completed `final_summary` evidence. State that ASLSingleLabel is a baseline, not custom. State when a notebook has not yet been executed on the RTX 4090 target.

# Source Review Notes

Reviewed sources before implementing `final_asl_custom_losses_convnext_yolo_repeat1_compact.ipynb`: Ultralytics custom trainer docs, Ultralytics classification trainer and classification loss references, Ultralytics classification task docs, Alibaba-MIIL/ASL official repository and ASL paper, PolyLoss paper, APL paper/code reference, SCE official code/paper, LDAM-DRW official code/paper, representative cost-sensitive learning references, and PyTorch AMP documentation.

Key source findings:

- Ultralytics custom trainer docs: `model.train(..., trainer=MyCustomTrainer)` is the supported custom trainer entry point. Source: `https://docs.ultralytics.com/guides/custom-trainer/`
- Ultralytics classification trainer docs expose `ClassificationTrainer.get_model`, `preprocess_batch`, and `setup_model`; classification batches place labels in `batch["cls"]`. Source: `https://docs.ultralytics.com/reference/models/yolo/classify/train/`
- Ultralytics classification loss reference computes default classification CE from logits in `preds` or `preds[1]` and labels in `batch["cls"]`. Source: `https://docs.ultralytics.com/reference/utils/loss/`
- Alibaba-MIIL/ASL is the official PyTorch implementation and includes `ASLSingleLabel` for single-label softmax classification. Sources: `https://github.com/Alibaba-MIIL/ASL`, `https://arxiv.org/abs/2009.14119`
- PolyLoss is an existing polynomial-expansion view of CE/focal-style losses, so simple ASL plus Poly1 must not be claimed as broadly novel. Source: `https://arxiv.org/abs/2204.12511`
- APL is an existing asymmetric pseudo-labeling idea in a different single-positive multi-label setting, so this project should avoid broad novelty claims around asymmetric variants. Source: `https://arxiv.org/abs/2203.16219`
- SCE and LDAM are existing robust/noisy-label and class-imbalance references. Sources: `https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels`, `https://github.com/kaidic/LDAM-DRW`
- Cost-sensitive classification is a long-standing area; this project's cost matrix is dataset-specific, not a new cost-sensitive learning framework. Representative source: `https://link.springer.com/article/10.1007/s10994-024-06634-8`
- PyTorch AMP docs warn that plain BCE on probabilities is unsafe under autocast and recommend logits-safe BCE variants or float32 handling. Source: `https://docs.pytorch.org/docs/2.9/amp.html`
