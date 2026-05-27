# CVio_Shrimp_Disease_Classification_Capstone_SU26

Tài liệu này mô tả cấu trúc lưu trữ notebook, môi trường thực thi, và tổng hợp kết quả tốt nhất đã đạt được từ thư mục báo cáo `final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs`.

## Cấu trúc thư mục

- `best`: chứa notebook tốt nhất tính đến hiện tại. Đây là những phiên bản pipeline có hiệu suất cao nhất và được xem là cơ sở chính để tham chiếu.
- `experiment`: chứa các notebook sử dụng trong quá trình thử nghiệm, điều chỉnh và đánh giá nhiều cấu hình trước khi chọn mô hình tối ưu.
- `legacy`: chứa các notebook cũ, các phiên bản trước đó, hoặc các notebook không còn được ưu tiên sử dụng nhưng vẫn được giữ lại để đối chiếu và truy vết.

## Môi trường thực thi

- Python: 3.12
- Chiến lược đánh giá: so sánh giữa `ASL` và `Baseline CE` trên hai backend chính `convnext_tiny` và `yolo26m-cls`
- Tiêu chí chọn mô hình tốt nhất: `val_macro_f1`

## Kết quả tốt nhất

Kết quả dưới đây được tổng hợp từ các file báo cáo trong thư mục `final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs/reports` và các file ảnh trong `final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs/plots`.

| Mô hình | Loss | Val Macro F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | FPS | Latency (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| convnext_tiny | ASL | 0.8810 | 0.7919 | 0.7720 | 0.7615 | 0.7655 | 24.98 | 40.08 |
| convnext_tiny | Baseline CE | 0.8810 | 0.7919 | 0.7720 | 0.7615 | 0.7655 | 22.29 | 45.67 |
| yolo26m-cls | ASL | 0.8244 | 0.8786 | 0.8834 | 0.8850 | 0.8787 | 15.59 | 64.24 |
| yolo26m-cls | Baseline CE | 0.8126 | 0.8786 | 0.8787 | 0.8859 | 0.8770 | 15.04 | 66.71 |

### Nhận xét chuyên môn

- `convnext_tiny + ASL` là cấu hình có chất lượng xác thực tốt nhất theo `val_macro_f1` và đồng thời giữ được hiệu suất suy luận cân bằng.
- Trên tập kiểm tra, kết quả của `convnext_tiny + ASL` đạt `Test F1-Score = 0.7655` và `Test Accuracy = 0.7919`.
- `yolo26m-cls` có `Test Accuracy` cao hơn, nhưng `convnext_tiny + ASL` vẫn là căn cứ chính để xem là pipeline tốt nhất do có kết quả xác thực ổn định và phù hợp với tiêu chí lựa chọn đã định.

## Trực quan hóa kết quả

Hình ảnh dưới đây được lấy trực tiếp từ thư mục báo cáo. Khi đọc README trên GitHub hoặc trong VS Code, các hình này sẽ giúp so sánh nhanh giữa các mô hình và các tiếp cận loss.

### Tổng hợp Test F1-Score

![Tổng hợp Test F1-Score](final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs/plots/bar_mean_Test_F1-Score.png)

### Tổng hợp Test Accuracy

![Tổng hợp Test Accuracy](final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs/plots/bar_mean_Test_Accuracy.png)

### Ma trận nhầm lẫn của cấu hình tốt nhất

![Ma trận nhầm lẫn convnext_tiny + ASL](final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs/plots/confusion_matrix_repeat03_convnext_tiny_asl.png)

## Tài liệu báo cáo

- Báo cáo phân loại cho cấu hình tốt nhất: `final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs/reports/classification_report_repeat03_convnext_tiny_asl.csv`
- Báo cáo tổng hợp 3 lần chạy: `final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs/reports/ce_asl_convnext_tiny_yolo26m_repeat3_summary.csv`
- Báo cáo độ bền vững: `final_yolo26m_convnext_tiny_ce_asl_original_yolo_best_repeat3_outputs/reports/ce_asl_convnext_tiny_yolo26m_repeat3_reproducibility_stats.csv`

## Ghi chú về cách đọc README

- Ưu tiên bảng tổng hợp khi cần đối chiếu nhanh giữa các cấu hình.
- Ưu tiên hình ảnh khi cần đánh giá bức tranh tổng quan về hiệu suất mô hình.
- Ưu tiên các file `reports/` khi cần kiểm tra lại số liệu gốc hoặc lặp lại phân tích.

## Current Best Partial YOLOv26m-cls Result

The strongest newer YOLOv26m-cls result currently observed in the partial two-model 14-loss attempt is:

| Scope | Model | Loss | Test Macro-F1 | Cohen Kappa | Accuracy | BG_WSSV Recall | WSSV to BG_WSSV |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Partial YOLO-only diagnostic | YOLOv26m-cls | ASLSingleLabel | 0.9109 | 0.8818 | 0.9133 | 0.9697 | 5 |

Source: `experiment/final_yolo26m_convnext_tiny_14losses_repeat1_audited_outputs/loss_run_summary_raw.csv`, produced by `experiment/final_yolo26m_convnext_tiny_14losses_repeat1_audited.ipynb`.

Important interpretation:

- This is a partial YOLO-only diagnostic result from the later two-model notebook attempt, not a completed final two-model benchmark.
- ASLSingleLabel is an existing official-style ASL baseline, not a custom proposed loss.
- `experiment/final_yolo26m_convnext_tiny_14losses_repeat1_audited.ipynb` crashed before ConvNeXt-Tiny runs started and before `final_summary.xlsx`, `final_summary.json`, and `figures_and_reports.zip` were created.

## Completed Same-Seed YOLOv26m-cls All-Losses Reference

Keep this completed reference separate from the newer partial YOLO + ASLSingleLabel diagnostic above. The completed same-seed YOLOv26m-cls all-losses run is under `experiment/final_yolo26m_cls_all_losses_anchor_best_outputs/`.

| Rank | Loss | Macro-F1 | Kappa | Accuracy | BG_WSSV Recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | CE | 0.9051 | 0.8739 | 0.9075 | 1.0000 |
| 2 | SCE | 0.9040 | 0.8736 | 0.9075 | 1.0000 |
| 3 | LDAM | 0.8984 | 0.8651 | 0.9017 | 0.9697 |
| 4 | Co-Infection Margin ASL | 0.8929 | 0.8582 | 0.8960 | 1.0000 |
| 5 | ASLSingleLabel | 0.8854 | 0.8502 | 0.8902 | 0.9394 |

These repeats used same-seed reproducibility, not independent-seed robustness. CE remains the best overall loss in that completed reference run.

## ConvNeXt-Tiny 14-Loss Notebook

New notebook:

- `experiment/final_convnext_tiny_shrimpxnet_14losses_repeat1_audited.ipynb`

Purpose:

- ConvNeXt-Tiny/ShrimpXNet baseline-style 14-loss ablation.
- Fixed 70/15/15 split, seed 42, repeat count 1.
- Intended runtime: Linux with RTX 4090 24GB.
- The exact `shrimpxnet.ipynb` file was not present in this checkout during generation, so the ConvNeXt pipeline was extracted from the available local ConvNeXt/ShrimpXNet reference notebooks under `legacy/` and `best/`.

Expected output directory:

- `final_convnext_tiny_shrimpxnet_14losses_repeat1_audited_outputs/`

Expected summary artifacts include:

- `run_audit.json`
- `environment_versions.json`
- `fixed_split_manifest_seed42_with_md5.csv`
- `loss_run_summary_raw.csv`
- `loss_group_stats.csv`
- `loss_deltas_vs_ce.csv`
- `bg_wssv_error_summary.csv`
- `all_predictions.csv`
- `final_summary.xlsx`
- `final_summary.json`
- `output_table_audit.csv`
- `figures_and_reports.zip`

## Known Runtime Issues and Fixes

- The partial two-model notebook failed during final partial-state handling because empty failed-result paths were converted to `Path(".")` and treated as readable files. The ConvNeXt-only notebook skips empty paths and directories.
- AttributeProjectionCE needed an AMP-safe BCE implementation. The ConvNeXt-only notebook avoids `torch.nn.functional.binary_cross_entropy` on probability tensors under autocast and uses a manual float32 attribute term.
- The final two-model 14-loss comparison is not complete until ConvNeXt-Tiny runs all 14 losses and final summary artifacts exist.

## ASL-Based Custom Loss Variants Experiment

New compact notebook:

- `final_asl_custom_losses_convnext_yolo_repeat1_compact.ipynb`

Purpose:

- Compare ConvNeXt-Tiny/ShrimpXNet-style training and native Ultralytics YOLOv26m-cls training on one fixed 70/15/15 split, seed 42, repeat count 1.
- Evaluate 6 existing baseline/reference losses and 10 dataset-specific ASL-based co-infection suppression variants.
- Keep outputs compact and centered on `final_summary.xlsx` and `final_summary.csv`.

Output directory:

- `final_asl_custom_losses_convnext_yolo_repeat1_compact_outputs/`

Primary outputs:

- `final_summary.xlsx`
- `final_summary.csv`
- `loss_run_summary_raw.csv`
- `run_audit.json`
- `environment_versions.json`
- `fixed_split_manifest_seed42_with_md5.csv`
- `yolo_split_manifest_seed42_with_md5.csv`
- `missing_or_failed_runs.csv`
- `memory_bank_asl_custom_losses_experiment_summary.md`

Existing baseline/reference losses:

| Key | Loss | Category |
| --- | --- | --- |
| `baseline_ce` | CE | existing baseline |
| `asl_single_label` | ASLSingleLabel | existing ASL baseline |
| `false_coinfection_cost_ce` | False-CoInfection Cost CE | existing custom reference |
| `poly_dcs_ce` | Poly-DCS-CE | existing custom reference |
| `sce` | SCE | existing robust baseline |
| `ldam` | LDAM | existing margin baseline |

ASL-based custom variants:

| Key | Loss |
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

Important interpretation:

- ASLSingleLabel is an existing official-style ASL baseline, not a custom proposed method.
- The new ASL variants should be described carefully as dataset-specific ASL-based co-infection suppression variants.
- The notebook defaults to `RUN_TRAINING=0`; no final results exist for this new experiment until it is executed on the Linux RTX 4090 24GB target with `RUN_TRAINING=1`.
- This new experiment must remain separate from the completed same-seed YOLO all-losses reference, the partial YOLO ASLSingleLabel diagnostic, and the completed ConvNeXt-Tiny 14-loss context.
