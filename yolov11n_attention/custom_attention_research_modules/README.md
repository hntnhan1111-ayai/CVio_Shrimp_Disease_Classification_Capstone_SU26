# Custom Attention Research Modules

This folder contains lightweight attention proposals from new_res_attention/deep-research-report.md and the follow-up report deep-research-report (1).md for YOLO11n-seg shrimp disease instance segmentation (BG/WSSV).

The original clean baseline notebook is not modified. The generated notebooks preserve the Roboflow dataset path, grouped shrimp-level no-leakage split, data YAML, nc, class names, seed, training configuration, validation/test logic, healthy false-positive metrics, and report export flow.

The training notebooks follow the self-contained style used in yolov11n_simam_NhomB: each notebook includes the attention class definitions, the Ultralytics namespace/parse_model patch, the YAML text writer, and the sanity-check function inline. Running a notebook does not import custom_attention_research_modules._shared.

## Original Modules

| Folder | Class | Full name | Placement |
|---|---|---|---|
| cote_gate | CoTEGate | Consensus-Regularized Triplet-ECA Gate | P4-only before Segment |
| scsg_gate | SCSGGate | Scale-Aware Coordinate-SGE Gate | P4-only before Segment |
| lpsc_gate | LPSCGate | Lesion-Preserving NAM-SimAM Contrast Gate | P4-only before Segment |

Each original module has one main notebook. Follow-up improvement notebooks are kept in separate folders so the baseline/original experiments remain unchanged.

## Report Improvement Notebooks

These notebooks are Kaggle-standalone: attention patching and YAML creation are inline in the notebook, so uploading a single notebook to Kaggle is enough.

Recommended run order from deep-research-report (1).md:

| Priority | Folder | Notebook | Class | Main idea |
|---|---|---|---|---|
| 1 | cote_sr_gate | cote_sr_p4_strong.ipynb | CoTESimAMRescueGate | CoTE plus SimAM rescue |
| 2 | lpsc_er_gate | lpsc_er_p4_strong.ipynb | LPSCECARescueGate | Soft LPSC plus ECA channel rescue |
| 3 | cote_sd_gate | cote_sd_p4_strong.ipynb | CoTESoftDisagreementGate | tanh-softened CoTE disagreement |
| 4 | lpsc_sg_gate | lpsc_sg_p4_strong.ipynb | LPSCSoftGate | weaker NAM, stronger SimAM rescue |
| 5 | lpsc_ua_gate | lpsc_ua_p4_strong.ipynb | LPSCUncertaintyAttenuatedGate | attenuate NAM when NAM and SimAM disagree |
| 6 | cote_bl_gate | cote_bl_p4_strong.ipynb | CoTEBoundaryLiteGate | CoTE plus light depthwise boundary prior |

All six are P4-only before the Segment head and use the same strong augmentation policy as the current SimAM_CA strong notebook.

Additional controlled hybrid follow-up:

| Folder | Notebook | Class | Augmentation | Main idea |
|---|---|---|---|---|
| colpsc_rescue_lite | colpsc_rescue_lite.ipynb | CoLPSCRescueLiteGate | clean/light baseline augmentation | CoTE as lesion-preserving evidence, LPSC as weak suppression, ECA/SimAM/BoundaryLite as rescue/refinement |
| colpsc_rescue_lite | colpsc_rescue_lite_p4_strong.ipynb | CoLPSCRescueLiteGate | strong augmentation | Same P4-only controlled hybrid with the SimAM_CA strong augmentation policy |

## Structure

    custom_attention_research_modules/
    |-- README.md
    |-- run_all_sanity_checks.py
    |-- _shared/
    |   |-- __init__.py
    |   |-- attention_modules.py
    |   |-- register_attention.py
    |   |-- sanity_check.py
    |   |-- notebook_utils.py
    |   |-- yaml_utils.py
    |-- cote_gate/
    |   |-- README.md
    |   |-- cote_gate.yaml
    |   |-- cote_gate.ipynb
    |-- scsg_gate/
    |   |-- README.md
    |   |-- scsg_gate.yaml
    |   |-- scsg_gate.ipynb
    |-- lpsc_gate/
    |   |-- README.md
    |   |-- lpsc_gate.yaml
    |   |-- lpsc_gate.ipynb
    |-- cote_sr_gate/
    |   |-- cote_sr_p4_strong.ipynb
    |-- lpsc_er_gate/
    |   |-- lpsc_er_p4_strong.ipynb
    |-- cote_sd_gate/
    |   |-- cote_sd_p4_strong.ipynb
    |-- lpsc_sg_gate/
    |   |-- lpsc_sg_p4_strong.ipynb
    |-- lpsc_ua_gate/
    |   |-- lpsc_ua_p4_strong.ipynb
    |-- cote_bl_gate/
    |   |-- cote_bl_p4_strong.ipynb
    |-- colpsc_rescue_lite/
        |-- colpsc_rescue_lite.ipynb
        |-- colpsc_rescue_lite_p4_strong.ipynb

## Sanity Checks

Run one module:

    python custom_attention_research_modules/_shared/sanity_check.py --model custom_attention_research_modules/cote_gate/cote_gate.yaml --imgsz 640

Run all modules:

    python custom_attention_research_modules/run_all_sanity_checks.py

Do not train a notebook if its sanity check fails.

## Training Notebooks

Open one of:

    custom_attention_research_modules/cote_gate/cote_gate.ipynb
    custom_attention_research_modules/scsg_gate/scsg_gate.ipynb
    custom_attention_research_modules/lpsc_gate/lpsc_gate.ipynb

Or one of the report improvement notebooks:

    custom_attention_research_modules/cote_sr_gate/cote_sr_p4_strong.ipynb
    custom_attention_research_modules/lpsc_er_gate/lpsc_er_p4_strong.ipynb
    custom_attention_research_modules/cote_sd_gate/cote_sd_p4_strong.ipynb
    custom_attention_research_modules/lpsc_sg_gate/lpsc_sg_p4_strong.ipynb
    custom_attention_research_modules/lpsc_ua_gate/lpsc_ua_p4_strong.ipynb
    custom_attention_research_modules/cote_bl_gate/cote_bl_p4_strong.ipynb
    custom_attention_research_modules/colpsc_rescue_lite/colpsc_rescue_lite.ipynb
    custom_attention_research_modules/colpsc_rescue_lite/colpsc_rescue_lite_p4_strong.ipynb

Training outputs are routed to:

    runs/custom_attention_research_modules/

Reports are routed under each module folder output directory. At runtime, each notebook writes its own generated YAML under custom_attention_research_modules/<module>/generated_yamls/ before training.

## Notes

- Baseline source: yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb.
- Reference structure: yolov11n_simam_NhomB.
- The baseline notebook and older attention folders are not edited.
- The first pass uses P4-only insertion to reduce shape and overfit risk.
- P3/P4 placement variants are still deferred. The new improvement notebooks are P4-only to match the report's first-pass recommendation.
