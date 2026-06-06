# YOLOv26m-cls CBAM Loss Trio

Runs exactly three variants for the Windows 6GB smoke/full workflow:

1. CE + CBAM
2. ASL + CBAM
3. ASL-LDAM + CBAM

Fixed references included in reports, not rerun:

- Stage1 YOLOv26m-cls CE: Macro-F1 0.8902, Accuracy 0.8902, Kappa 0.8505
- ASL-LDAM no attention: Macro-F1 0.8991, Accuracy 0.9017, Kappa 0.8661
- Prior ASL-LDAM + sparse_learnable_fusion_cbam: Macro-F1 0.9006, Accuracy 0.9017, Kappa 0.8658

Important: this experiment folder requires the full CVio project code in the repository root, especially:

- shrimp_scripts/
- experiments/asl_custom_loss_screening/

Run from target Windows 6GB repo root:

```powershell
cd C:\Users\ADMIN\Downloads\CVio_Shrimp_Disease_Classification_Capstone_SU26
.\.venv\Scripts\activate
powershell -ExecutionPolicy Bypass -File .\experiments\yolo26m_cbam_loss_trio\run_windows6gb_cbam_loss_trio.ps1 -Mode smoke
```

If smoke passes:

```powershell
powershell -ExecutionPolicy Bypass -File .\experiments\yolo26m_cbam_loss_trio\run_windows6gb_cbam_loss_trio.ps1 -Mode full
```
