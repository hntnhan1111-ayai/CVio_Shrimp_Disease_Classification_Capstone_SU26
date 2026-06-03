# YOLO11n-seg + SimAM Head Experiment

## Goal

This folder contains an Ultralytics YOLO11n instance segmentation experiment that inserts SimAM attention on the P3, P4, and P5 feature maps immediately before the segmentation head.

## Files Changed

- `ultralytics/ultralytics/nn/modules/conv.py`: added the parameter-free `SimAM` module.
- `ultralytics/ultralytics/nn/modules/__init__.py`: exported `SimAM`.
- `ultralytics/ultralytics/nn/tasks.py`: imported `SimAM` and added parser support so it preserves input channels.

## New Files

- `ultralytics/ultralytics/cfg/models/11/yolo11n-seg-simam-head.yaml`: YOLO11n-seg YAML with SimAM inserted before `Segment`.
- `yolo11n_seg_simam_experiment.ipynb`: notebook for environment checks, model build, training, validation, prediction, visualization, and export.
- `setup_simam_ultralytics.py`: Colab/local setup helper that clones Ultralytics if missing and applies the SimAM patch.
- `README_SIMAM.md`: this guide.

## Run The Notebook

Open `yolo11n_seg_simam_experiment.ipynb` from the `yolov11n_attention` directory. The notebook expects the local Ultralytics source to be available at:

```text
yolov11n_attention/ultralytics
```

If needed, install the local source in editable mode:

```bash
cd yolov11n_attention/ultralytics
pip install -e .
```

## Run On Google Colab

1. Upload, clone, or open the `yolov11n_attention` folder in Colab or Google Drive.
2. Open `yolo11n_seg_simam_experiment.ipynb` in Colab.
3. Add a Colab Secret named `ROBOFLOW_API_KEY` for the Roboflow dataset.
4. Run the `Colab Quick Setup` cell first.
   - If you opened the notebook directly from GitHub and the folder is not present in `/content`, the cell can clone this repo from `GITHUB_REPO_URL`.
   - If `ultralytics/` is missing, the setup helper clones it.
   - The helper applies the `SimAM` source patch.
   - The notebook installs local Ultralytics with `pip install -e ultralytics`.
5. If the folder is in Google Drive, uncomment the Drive mount lines in that cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

6. If your folder path is different, set:

```python
COLAB_WORK_DIR = "/content/drive/MyDrive/yolov11n_attention"
```

7. Run the environment and dataset setup cells.
8. Run the model build and dummy forward cells before training.

The dataset setup cell uses the same dataset as `baseline-sweep-yolov11m.ipynb`:

```text
Roboflow workspace: shirmpdiseasedtection
Roboflow project: ewu_shrimp_disease
Version: 1
Format: yolo26
Seed: 42
```

It downloads to `/content/ewu_shrimp_disease-1` on Colab and updates `data.yaml` to absolute paths:

- `/content/ewu_shrimp_disease-1/train/images`
- `/content/ewu_shrimp_disease-1/valid/images`
- `/content/ewu_shrimp_disease-1/test/images`

The notebook also sets `TEST_SOURCE` to the test image folder automatically.

## Why `ultralytics/` May Not Appear On GitHub

The local `yolov11n_attention/ultralytics` directory is itself a nested Git repository with its own `.git` folder and remote:

```text
https://github.com/ultralytics/ultralytics.git
```

Because of that, the parent shrimp project does not automatically track all files inside it. A normal push of the parent repo may only include files such as the notebook and README, while `ultralytics/` remains untracked or becomes a submodule-style entry if added directly.

Recommended Colab approach: commit `setup_simam_ultralytics.py`, the notebook, and this README. Let Colab clone and patch Ultralytics during setup.

If you really want the full `ultralytics/` source folder visible inside your GitHub repo, remove the nested `ultralytics/.git` folder first, then add and commit the folder from the parent repo. This vendors the full upstream source and is much larger.

## Configure DATA_YAML

Normally the dataset setup cell defines `DATA_YAML` automatically. To use a different dataset manually, replace:

```python
DATA_YAML = "/path/to/your/data.yaml"
```

with the path to your shrimp segmentation dataset YAML. Examples:

```python
DATA_YAML = "/kaggle/input/shrimp-dataset/data.yaml"
DATA_YAML = "/content/datasets/shrimp/data.yaml"
DATA_YAML = r"D:/datasets/shrimp/data.yaml"
```

## Train

After the build and dummy forward cells pass, run the training cell. The main training call is:

```python
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    workers=WORKERS,
    project=PROJECT_DIR,
    name=EXP_NAME,
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    patience=30,
    close_mosaic=10,
    mask_ratio=4,
    overlap_mask=True,
)
```

## Validate

The notebook validates:

```text
runs/shrimp_yolo11n_seg_simam/yolo11n_seg_simam_head/weights/best.pt
```

Update `PROJECT_DIR`, `EXP_NAME`, or `BEST_PT` if your run folder is different.

## Predict

Set:

```python
TEST_SOURCE = "/path/to/test/images"
```

then run the prediction and visualization cells. Predictions are saved under:

```text
runs/predict_shrimp_simam/simam_predict
```

## Model YAML Change

The original YOLO11 segmentation head uses P3, P4, and P5 feature maps directly:

```yaml
- [[16, 19, 22], 1, Segment, [nc, 32, 256]]
```

The SimAM version routes each feature map through attention first:

```yaml
# SimAM attention inserted before Segment head for P3, P4, P5 features
- [16, 1, SimAM, []]
- [19, 1, SimAM, []]
- [22, 1, SimAM, []]
- [[23, 24, 25], 1, Segment, [nc, 32, 256]]
```
