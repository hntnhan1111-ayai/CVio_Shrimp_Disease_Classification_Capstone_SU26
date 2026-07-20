# YOLO11n Custom Attention - Nhom E

Group E implements rescue experiments for low-FP modules that may increase disease miss-rate.

- `yolo11_cus_att_lpsc_soft_gate.ipynb`: E1 LPSC Soft Gate rescue for LPSC strong, LPSC-ER, and LPSC-UA.
- `yolo11_cus_att_cote_rescue.ipynb`: E2 CoTE rescue variants with `nm64`, boundary loss, threshold calibration, and SimAM-backbone.
- `yolo11_cus_att_prototype_aware_boundary_loss.ipynb`: E3 Prototype-Aware gate with `nm64`, boundary-aware mask loss, and threshold sweep.

Each notebook is standalone: it embeds helper code, custom modules, model YAMLs, data download/split preparation, training/evaluation, threshold sweep, and test-set prediction preview. No `_shared` file is required when uploading/running a single notebook.
