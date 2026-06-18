# Specimen-Leakage-Aware Shrimp Disease Segmentation

This repository contains the paper-facing code for leakage-aware shrimp disease instance segmentation experiments. The project builds a YOLO-based segmentation baseline on a hand-labeled BG/WSSV shrimp disease dataset and evaluates it under specimen-grouped splitting, where all images of the same physical shrimp are kept in the same split.

The main goal is to make shrimp disease segmentation evaluation fairer and easier to reproduce. Random image-level splitting can place different views of the same shrimp into train, validation, and test sets, which inflates performance. The notebooks in this repository compare leakage-prone split policies against leakage-free grouped-specimen evaluation, then screen lightweight YOLO models and augmentation/preprocessing variants.

## Repository Layout

```text
shrimp-leakage-aware-segmentation/
├── notebooks/
│   ├── baseline/
│   ├── model-sweep/
│   ├── split-policy/
│   ├── yolo_aug_policy_search/
│   ├── photometric_randaug/
│   ├── copy_paste_focused_sweep/
│   ├── frequency_wavelet_revisit/
│   └── README.md
├── paper/
│   ├── manuscript.tex
│   ├── references.bib
│   └── figures/template files
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Dataset

The experiments use a hand-labeled instance segmentation dataset derived from ShrimpDiseaseImageBD. The labeled segmentation dataset is hosted on Roboflow Universe:

https://universe.roboflow.com/lets-try-this/shrimpdishandsegv2

The notebooks download the dataset through Roboflow. Set the API key as an environment variable or cloud secret named `ROBOFLOW_API_KEY`. Do not hardcode API keys into notebooks before committing.

## Experiment Contract

The paper-facing baseline uses:

- Model: `YOLO11n-seg`
- Dataset: hand-labeled shrimp disease instance segmentation
- Classes: `BG`, `WSSV`
- Split policy: disease-stratified grouped-specimen split
- Primary seed: `42`
- YOLO hidden Albumentations hook: enabled
- Baseline augmentation: clean-light YOLO augmentation
- Main evaluation: held-out specimen test set

The grouped split is the key reproducibility constraint. Images from the same specimen must not appear across train, validation, and test at the same time.

## Notebook Run Order

See [notebooks/README.md](notebooks/README.md) for the detailed notebook guide.

Recommended order:

1. `notebooks/baseline/yolo11n_grouped_clean_baseline.ipynb`
2. `notebooks/split-policy/random_and_stratified_random_seed_variance.ipynb`
3. `notebooks/split-policy/grouped_specimen_seed_stability.ipynb`
4. `notebooks/model-sweep/grouped_model_sweep.ipynb`
5. Optimization notebooks:
   - `notebooks/yolo_aug_policy_search/yolo_aug_policy_search.ipynb`
   - `notebooks/photometric_randaug/photometric_randaug_screening.ipynb`
   - `notebooks/copy_paste_focused_sweep/copy_paste_focused_sweep.ipynb`
   - `notebooks/frequency_wavelet_revisit/frequency_wavelet_revisit.ipynb`

The notebooks are designed for cloud execution on Kaggle or Colab. Use `SMOKE_RUN = True` only for quick runtime validation. Use full training settings for paper metrics.

## Outputs to Preserve

After cloud runs, download the generated reports before deleting the session:

- `*_summary.csv`
- `*_partial.csv`, if the run was interrupted
- `*_paper_table.csv` or `*_paper_row.csv`
- `split_manifests/`
- `results.csv`
- `best.pt`, when checkpoint inspection or future evaluation is needed

Large generated artifacts are intentionally excluded from version control.

## Installation

The notebooks install or check their own dependencies in cloud environments. For local inspection, create an environment with:

```bash
pip install -r requirements.txt
```

Torch/CUDA packages are not pinned in `requirements.txt` because Kaggle and Colab provide different GPU runtimes. Use the platform-provided PyTorch build unless you intentionally manage CUDA yourself.

## Paper

The manuscript source is in `paper/manuscript.tex`. The current paper studies:

- specimen-level leakage in shrimp disease segmentation,
- grouped-specimen split construction,
- split-policy stability across seeds,
- YOLO model selection under leakage-free evaluation,
- healthy-aware diagnostics for false positives on healthy shrimp.

## Citation

This work is still under active paper preparation. A provisional `CITATION.cff` is included so that the code and protocol can be cited consistently. Update it with the final venue, DOI, and repository URL after publication.

## License

The code in this repository is released under the MIT License. Dataset images and annotations may be subject to their original dataset and hosting terms.
