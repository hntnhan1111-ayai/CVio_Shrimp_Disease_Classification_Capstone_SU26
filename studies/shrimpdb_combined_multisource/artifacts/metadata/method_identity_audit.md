# Method-identity audit

Status: **contradictory**
Audit timestamp (UTC): `2026-07-24T03:26:55.331740+00:00`

This audit does not modify either external result package or any raw metric file.

## Package conclusions

| Package | Verified method | Confidence | Evidence checks | Contradictions |
|---|---|---:|---:|---:|
| `CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS` | `baseline_ce` | high | 56 | 0 |
| `CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS` | `asl_ldam_simam_dcfr` | high | 60 | 0 |

## Independent evidence

### CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS

- **PASS** `training.method` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\study_config.yaml`; observed `baseline_ce`.
- **PASS** `loss.name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\study_config.yaml`; observed `CrossEntropyLoss`.
- **PASS** `attention.name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\study_config.yaml`; observed `None`.
- **PASS** `attention.enabled` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\study_config.yaml`; observed `False`.
- **PASS** `CVIO_YOLO_LOSS semantic` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\study_config.yaml`; observed `baseline_ce in notebook/runtime source`.
- **PASS** `experiment training.method` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\shrimpdb3.yaml`; observed `baseline_ce`.
- **PASS** `experiment loss.name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\shrimpdb3.yaml`; observed `CrossEntropyLoss`.
- **PASS** `experiment attention.name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\shrimpdb3.yaml`; observed `None`.
- **PASS** `experiment training.method` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\combined4.yaml`; observed `baseline_ce`.
- **PASS** `experiment loss.name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\combined4.yaml`; observed `CrossEntropyLoss`.
- **PASS** `experiment attention.name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\configs\combined4.yaml`; observed `None`.
- **PASS** `run args name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\args.yaml`; observed `yolo26m_cls__baseline_ce__combined4__seed42`.
- **PASS** `effective protocol method` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\effective_research_protocol.json`; observed `baseline_ce`.
- **PASS** `effective protocol loss` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\effective_research_protocol.json`; observed `CrossEntropyLoss`.
- **PASS** `effective protocol attention` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\effective_research_protocol.json`; observed `None`.
- **PASS** `run args name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\args.yaml`; observed `yolo26m_cls__baseline_ce__shrimpdb3__seed42`.
- **PASS** `effective protocol method` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\effective_research_protocol.json`; observed `baseline_ce`.
- **PASS** `effective protocol loss` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\effective_research_protocol.json`; observed `CrossEntropyLoss`.
- **PASS** `effective protocol attention` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\effective_research_protocol.json`; observed `None`.
- **PASS** `training attempts log inspected` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\logs\combined4_training.attempts.json`; observed `True`.
- **PASS** `training log method-bearing run path` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\logs\combined4_training.log`; observed `True`.
- **PASS** `training log method-bearing run path` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\logs\combined4_training_attempt1_batch32.log`; observed `True`.
- **PASS** `training attempts log inspected` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\logs\shrimpdb3_training.attempts.json`; observed `True`.
- **PASS** `training log method-bearing run path` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\logs\shrimpdb3_training.log`; observed `True`.
- **PASS** `training log method-bearing run path` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\logs\shrimpdb3_training_attempt1_batch32.log`; observed `True`.
- **PASS** `baseline CE smoke test` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\audit\baseline_ce_smoke_test.json`; observed `shrimpdb3 class_names Healthy BG WSSV train_class_counts 49 78 94 loss_name CrossEntropyLoss loss 1.2026294469833374 loss_items 1.2026294469833374 attention_enabled False forbidden_attention_modules  trainable_parameters_with_gradients 149 output_shape 1 3 combined4 class_names Healthy BG WSSV WSSV_BG train_class_counts 331 217 323 154 loss_name CrossEntropyLoss loss 1.5672121047973633 loss_items 1.5672121047973633 attention_enabled False forbidden_attention_modules  trainable_parameters_with_gradients 149 output_shape 1 4`.
- **PASS** `notebook contains expected method setting` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_YOLO26m_CE_Baseline_E2E_v1_FIXED.ipynb`; observed `True`.
- **PASS** `notebook contains expected loss setting` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_YOLO26m_CE_Baseline_E2E_v1_FIXED.ipynb`; observed `True`.
- **PASS** `notebook embeds runtime method environment controls` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_YOLO26m_CE_Baseline_E2E_v1_FIXED.ipynb`; observed `['CVIO_YOLO_ATTENTION', 'CVIO_YOLO_LOSS']`.
- **PASS** `notebook trainer evidence: PaperClassificationTrainer` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_YOLO26m_CE_Baseline_E2E_v1_FIXED.ipynb`; observed `True`.
- **PASS** `notebook trainer evidence: CVIO_YOLO_ATTENTION` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_YOLO26m_CE_Baseline_E2E_v1_FIXED.ipynb`; observed `True`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\best.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\best.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV', 3: 'WSSV_BG'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\best.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\best.pt`; observed `4`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\best.pt`; observed `yolo26m_cls__baseline_ce__combined4__seed42`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\last.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\last.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV', 3: 'WSSV_BG'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\last.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\last.pt`; observed `4`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__combined4__seed42\weights\last.pt`; observed `yolo26m_cls__baseline_ce__combined4__seed42`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\best.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\best.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\best.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\best.pt`; observed `3`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\best.pt`; observed `yolo26m_cls__baseline_ce__shrimpdb3__seed42`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\last.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\last.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\last.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\last.pt`; observed `3`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\runs\yolo26m_cls__baseline_ce__shrimpdb3__seed42\weights\last.pt`; observed `yolo26m_cls__baseline_ce__shrimpdb3__seed42`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\final_baseline_model\yolo26m_ce_baseline_combined4_best.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\final_baseline_model\yolo26m_ce_baseline_combined4_best.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV', 3: 'WSSV_BG'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\final_baseline_model\yolo26m_ce_baseline_combined4_best.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\final_baseline_model\yolo26m_ce_baseline_combined4_best.pt`; observed `4`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_YOLO26m_CE_Baseline_seed42_RESULTS\final_baseline_model\yolo26m_ce_baseline_combined4_best.pt`; observed `yolo26m_cls__baseline_ce__combined4__seed42`.

### CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS

- **PASS** `training.method` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `asl_ldam_simam_dcfr`.
- **PASS** `loss.name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `ASL_LDAM`.
- **PASS** `attention.name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `SimAM_DCFR`.
- **PASS** `loss.gamma_pos` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `0.0`.
- **PASS** `loss.gamma_neg` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `4.0`.
- **PASS** `loss.label_smoothing` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `0.1`.
- **PASS** `loss.ldam_max_m` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `0.5`.
- **PASS** `loss.ldam_scale` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `30.0`.
- **PASS** `attention.e_lambda` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\study_config.yaml`; observed `0.0001`.
- **PASS** `experiment training.method` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\shrimpdb3.yaml`; observed `asl_ldam_simam_dcfr`.
- **PASS** `experiment loss.name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\shrimpdb3.yaml`; observed `ASL_LDAM`.
- **PASS** `experiment attention.name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\shrimpdb3.yaml`; observed `SimAM_DCFR`.
- **PASS** `experiment training.method` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\combined4.yaml`; observed `asl_ldam_simam_dcfr`.
- **PASS** `experiment loss.name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\combined4.yaml`; observed `ASL_LDAM`.
- **PASS** `experiment attention.name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\configs\combined4.yaml`; observed `SimAM_DCFR`.
- **PASS** `run args name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\args.yaml`; observed `yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42`.
- **PASS** `effective protocol method` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\effective_research_protocol.json`; observed `asl_ldam_simam_dcfr`.
- **PASS** `effective protocol loss` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\effective_research_protocol.json`; observed `ASL_LDAM`.
- **PASS** `effective protocol attention` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\effective_research_protocol.json`; observed `SimAM_DCFR`.
- **PASS** `run args name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\args.yaml`; observed `yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42`.
- **PASS** `effective protocol method` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\effective_research_protocol.json`; observed `asl_ldam_simam_dcfr`.
- **PASS** `effective protocol loss` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\effective_research_protocol.json`; observed `ASL_LDAM`.
- **PASS** `effective protocol attention` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\effective_research_protocol.json`; observed `SimAM_DCFR`.
- **PASS** `training attempts log inspected` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\logs\combined4_training.attempts.json`; observed `True`.
- **PASS** `training log method-bearing run path` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\logs\combined4_training.log`; observed `True`.
- **PASS** `training log method-bearing run path` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\logs\combined4_training_attempt1_batch32.log`; observed `True`.
- **PASS** `training attempts log inspected` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\logs\shrimpdb3_training.attempts.json`; observed `True`.
- **PASS** `training log method-bearing run path` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\logs\shrimpdb3_training.log`; observed `True`.
- **PASS** `training log method-bearing run path` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\logs\shrimpdb3_training_attempt1_batch32.log`; observed `True`.
- **PASS** `custom method smoke test` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\audit\custom_method_smoke_test.json`; observed `shrimpdb3 class_names Healthy BG WSSV class_counts 49 78 94 loss 27.413162231445312 loss_items 27.413162231445312 trainable_parameters_with_gradients 153 output_shape 1 3 attention_audit attention_key simam_gated_residual__dcfr_texture inserted True target_index 10 inferred_channels 512 head_input_shape 1 512 7 7 original_output_shape 1 3 verified_output_shape 1 3 params_added 529408 combined4 class_names Healthy BG WSSV WSSV_BG class_counts 331 217 323 154 loss 20.996074676513672 loss_items 20.996074676513672 trainable_parameters_with_gradients 153 output_shape 1 4 attention_audit attention_key simam_gated_residual__dcfr_texture inserted True target_index 10 inferred_channels 512 head_input_shape 1 512 7 7 original_output_shape 1 4 verified_output_shape 1 4 params_added 529408`.
- **PASS** `notebook contains expected method setting` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_E2E_v7_RUNTIME_FIX.ipynb`; observed `True`.
- **PASS** `notebook contains expected loss setting` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_E2E_v7_RUNTIME_FIX.ipynb`; observed `True`.
- **PASS** `notebook embeds runtime method environment controls` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_E2E_v7_RUNTIME_FIX.ipynb`; observed `['CVIO_YOLO_ATTENTION', 'CVIO_YOLO_LOSS']`.
- **PASS** `notebook trainer evidence: DynamicPaperClassificationTrainer` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_E2E_v7_RUNTIME_FIX.ipynb`; observed `True`.
- **PASS** `notebook trainer evidence: inject_attention_before_classify` — `D:\CVio\CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_E2E_v7_RUNTIME_FIX.ipynb`; observed `True`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\best.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\best.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV', 3: 'WSSV_BG'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\best.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'AttentionBeforeClassify', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU', 'Sigmoid', 'SimAMDCFR']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\best.pt`; observed `4`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\best.pt`; observed `yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\last.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\last.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV', 3: 'WSSV_BG'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\last.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'AttentionBeforeClassify', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU', 'Sigmoid', 'SimAMDCFR']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\last.pt`; observed `4`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42\weights\last.pt`; observed `yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\best.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\best.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\best.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'AttentionBeforeClassify', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU', 'Sigmoid', 'SimAMDCFR']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\best.pt`; observed `3`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\best.pt`; observed `yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\last.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\last.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\last.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'AttentionBeforeClassify', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU', 'Sigmoid', 'SimAMDCFR']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\last.pt`; observed `3`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\runs\yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42\weights\last.pt`; observed `yolo26m_cls__asl_ldam_simam_dcfr__shrimpdb3__seed42`.
- **PASS** `checkpoint model class` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\final_application_model\yolo26m_asl_ldam_simam_dcfr_combined4_best.pt`; observed `PaperClassificationModel`.
- **PASS** `checkpoint class names` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\final_application_model\yolo26m_asl_ldam_simam_dcfr_combined4_best.pt`; observed `{0: 'Healthy', 1: 'BG', 2: 'WSSV', 3: 'WSSV_BG'}`.
- **PASS** `checkpoint custom-module identity` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\final_application_model\yolo26m_asl_ldam_simam_dcfr_combined4_best.pt`; observed `['AdaptiveAvgPool2d', 'Attention', 'AttentionBeforeClassify', 'BatchNorm2d', 'Bottleneck', 'C2PSA', 'C3k', 'C3k2', 'Classify', 'Conv', 'Conv2d', 'Dropout', 'Identity', 'Linear', 'ModuleList', 'PSABlock', 'PaperClassificationModel', 'Sequential', 'SiLU', 'Sigmoid', 'SimAMDCFR']`.
- **PASS** `checkpoint model YAML output classes` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\final_application_model\yolo26m_asl_ldam_simam_dcfr_combined4_best.pt`; observed `4`.
- **PASS** `checkpoint run name` — `D:\CVio\CVio_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_seed42_RESULTS\final_application_model\yolo26m_asl_ldam_simam_dcfr_combined4_best.pt`; observed `yolo26m_cls__asl_ldam_simam_dcfr__combined4__seed42`.

## Blocking findings

- Requested corrected mapping expects baseline_ce combined4 accuracy 0.8318181818181818, but the verified baseline_ce package raw artifact records 0.8772727272727273.
- Requested corrected mapping expects asl_ldam_simam_dcfr combined4 accuracy 0.8772727272727273, but the verified asl_ldam_simam_dcfr package raw artifact records 0.8318181818181818.
- Combined-4 accuracy is cross-assigned relative to the requested presentation: 87.73% is serialized with the CE checkpoint and 83.18% with the ASL checkpoint. Reassigning these raw results would contradict checkpoint, config, smoke-test, and notebook evidence.

## Decision

Repository-facing result labels and tables must not be changed from this audit alone. The available executable evidence verifies the package identities, but it does not support the requested Combined-4 corrected mapping. A rerun or an additional authoritative artifact is required before scientific presentation can be corrected.
