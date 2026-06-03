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

## Configure DATA_YAML

In the notebook config cell, replace:

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
