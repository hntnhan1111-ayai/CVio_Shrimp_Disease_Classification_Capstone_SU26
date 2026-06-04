# Paper Experiments

This folder contains the curated reproducible files for the Stage 1 baselines and the YOLOv26m-cls ASL-LDAM Margin improvement run.

## Folder Map

```text
paper_experiments/
  README.md
  stage1_baselines/
    run_stage1_yolo_baseline_match_run_all.sh
    stage1_yolo_baseline_match_run_all.py
    stage1_timm16_shrimpxnet_baseline.ipynb
  yolo26m_asl_ldam/
    final_yolo26m_improvements_stage1_reference_runner.py
    final_yolo26m_improvements_stage1_reference_rtx4090.ipynb
    patch_yolo_attention_injection_v2.py
  review_artifacts/
    stage1_yolo_baseline_match_run_all_for_review.zip
    stage1_timm16_shrimpxnet_for_review.zip
    final_yolo26m_improvements_against_stage1_baseline_for_review.zip
```

## What Reproduces What

Stage 1 YOLO baseline comparison:
- `paper_experiments/stage1_baselines/run_stage1_yolo_baseline_match_run_all.sh`
- `paper_experiments/stage1_baselines/stage1_yolo_baseline_match_run_all.py`

Stage 1 TIMM / ShrimpXNet baseline comparison:
- `paper_experiments/stage1_baselines/stage1_timm16_shrimpxnet_baseline.ipynb`

YOLOv26m-cls ASL-derived custom-loss experiment:
- `paper_experiments/yolo26m_asl_ldam/final_yolo26m_improvements_stage1_reference_runner.py`
- `paper_experiments/yolo26m_asl_ldam/final_yolo26m_improvements_stage1_reference_rtx4090.ipynb`
- `paper_experiments/yolo26m_asl_ldam/patch_yolo_attention_injection_v2.py`

## Verified Metrics

Fixed Stage 1 YOLOv26m-cls CE baseline:
- Macro-F1: 0.8902
- Accuracy: 0.8902
- Cohen's Kappa: 0.8505

Best ASL-derived loss preserved here:
- Loss key: `asl_ldam_margin`
- Label: ASL-LDAM Margin
- Macro-F1: 0.8991317960390125, rounded to 0.8991
- Accuracy: 0.9017
- Cohen's Kappa: 0.8661

The runner includes `asl_ldam_margin` in `LOSS_KEYS`, and the verified ASL-LDAM result is recorded in the local audit log at `legacy/asl_verify_artifacts_after_yolo26m.log` for `run_id` `asl_custom_screening_ultralytics_yolo26m_cls_asl_ldam_margin_randaugment_seed42_repeat1`.

## How to Run

Stage 1 YOLO baseline:

```bash
bash paper_experiments/stage1_baselines/run_stage1_yolo_baseline_match_run_all.sh
```

Stage 1 TIMM / ShrimpXNet baseline:
- Open `paper_experiments/stage1_baselines/stage1_timm16_shrimpxnet_baseline.ipynb` and run all cells.

YOLOv26m-cls ASL-LDAM Margin workflow:

```bash
python paper_experiments/yolo26m_asl_ldam/final_yolo26m_improvements_stage1_reference_runner.py \
  --project_dir "<path-to-Improving Lightweight Shrimp Disease Classification with Co-Infection-Aware Losses and RandAugment>" \
  --output_dir "<path-to-output-dir>" \
  --patch_script paper_experiments/yolo26m_asl_ldam/patch_yolo_attention_injection_v2.py \
  --epochs 30
```

The notebook wrapper `paper_experiments/yolo26m_asl_ldam/final_yolo26m_improvements_stage1_reference_rtx4090.ipynb` runs the same workflow using the original Linux paths from the source environment.

## Review Artifacts

These ZIPs are review-only artifacts:
- `paper_experiments/review_artifacts/stage1_yolo_baseline_match_run_all_for_review.zip`
- `paper_experiments/review_artifacts/stage1_timm16_shrimpxnet_for_review.zip`
- `paper_experiments/review_artifacts/final_yolo26m_improvements_against_stage1_baseline_for_review.zip`

They were checked locally for prohibited payloads such as `.pt`, `.pth`, `.onnx`, `.engine`, `weights/`, `yolo_fixed_dataset`, and raw image extensions. None were found.

## Intentionally Excluded

Not committed here:
- `weights/`
- `runs/`
- `yolo_fixed_dataset/`
- `processed_images/`
- raw image folders and copied datasets
- `.pt`, `.pth`, `.onnx`, `.engine`
- `.env`
- tokens and secrets
- caches such as `__pycache__` and `.ipynb_checkpoints`
- unrelated old experiments and large generated outputs outside the curated review ZIPs
