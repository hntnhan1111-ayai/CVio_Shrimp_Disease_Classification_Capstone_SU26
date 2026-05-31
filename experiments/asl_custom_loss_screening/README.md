# ASL Custom-Loss Screening

Short screening experiment for ASL-derived losses before a larger final comparison.

## Scope

The screening matrix runs CE, ASLSingleLabel, and 10 ASL-derived candidates on:

- `convnext_tiny_in22k`
- `mobilenet_v3_large`
- `efficientnet_b0`
- YOLO classification models from `config.YOLO_FAMILY_MODELS`

Torch/TIMM screening rows always use `randaugment=False`. YOLO rows use the existing YOLO training path and set the screening condition to `randaugment=True`, which keeps Ultralytics classification RandAugment handling explicit in `train_kwargs.json` and `run_audit.json`.

YOLO screening rows use a module-level `PaperClassificationTrainer` for CE, ASLSingleLabel, and custom losses so the Ultralytics `ImageFolder` labels are remapped to the paper class order: `Healthy`, `BG`, `WSSV`, `WSSV_BG`.

## Losses

- `baseline_ce`
- `asl_single_label`
- `class_weighted_asl`
- `coinfection_weighted_asl`
- `boundary_weighted_asl`
- `confusion_aware_negative_asl`
- `soft_target_coinfection_asl`
- `attribute_projection_asl`
- `adaptive_gamma_asl`
- `asl_ldam_margin`
- `dangerous_confidence_penalty_asl`
- `coinfection_logit_adjusted_asl`

These candidates are screening hypotheses, not proven improvements. Ranking marks a custom loss as a candidate only when it beats both CE and ASL for the same model, or reduces co-infection boundary errors without a major Macro-F1 loss.

## Run

Prepare the dataset manifests in the chosen output directory first:

```bash
python shrimp_scripts/run_01_prepare_dataset.py --output_dir /kaggle/working/shrimp_outputs_asl_custom_screening
```

List the full matrix:

```bash
python experiments/asl_custom_loss_screening/run_asl_custom_screen.py --list_runs
```

Tiny Torch smoke run:

```bash
python experiments/asl_custom_loss_screening/run_asl_custom_screen.py --output_dir /kaggle/working/shrimp_outputs_asl_custom_screening --backend torch --model convnext_tiny_in22k --loss coinfection_weighted_asl --smoke_test --resume --progress
```

Tiny YOLO smoke run:

```bash
python experiments/asl_custom_loss_screening/run_asl_custom_screen.py --output_dir /kaggle/working/shrimp_outputs_asl_custom_screening --backend yolo --model yolo26m-cls --loss coinfection_weighted_asl --smoke_test --resume --progress
```

Verify completed artifacts:

```bash
python experiments/asl_custom_loss_screening/run_asl_custom_screen.py --output_dir /kaggle/working/shrimp_outputs_asl_custom_screening --verify_artifacts --skip_checkpoint_load
```

Collect and rank:

```bash
python experiments/asl_custom_loss_screening/collect_asl_custom_results.py --output_dir /kaggle/working/shrimp_outputs_asl_custom_screening
```

## Outputs

- `asl_custom_screening_plan_full.csv`
- `asl_custom_screening_plan_selected.csv`
- `asl_custom_screening_run_results.csv`
- `asl_custom_screening_run_results.json`
- `asl_custom_screening_artifact_verification.csv`
- `asl_custom_screening_artifact_verification.json`
- `asl_custom_screening_summary.csv`
- `asl_custom_screening_summary.json`
- `asl_custom_screening_failures.csv`
- `asl_custom_screening_ranked.csv`

Each completed run keeps the same per-run artifact layout as the main `shrimp_scripts` pipeline.
