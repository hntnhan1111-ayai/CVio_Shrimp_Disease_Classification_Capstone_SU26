# Model Deployment And Benchmarking

## Model Assets

Source model files live under root `models/`:

- `models/efficientnet_b0/efficientnet_b0_dynamic_range.tflite`
- `models/efficientnet_b0/efficientnet_b0_float16.tflite`
- `models/efficientnet_b0/efficientnet_b0_float32.tflite`
- `models/efficientnet_b0/efficientnet_b0_full_int8.tflite`
- `models/efficientnet_b0/efficientnet_b0.onnx`
- `models/mobilenet_v3_large/mobilenet_v3_large_dynamic_range.tflite`
- `models/mobilenet_v3_large/mobilenet_v3_large_float32.tflite`
- `models/mobilenet_v3_large/mobilenet_v3_large.onnx`
- `models/yolov26n-cls/yolov26n-cls_float16.tflite`
- `models/yolov26n-cls/yolov26n-cls_float32.tflite`
- `models/yolov26n-cls/yolov26n-cls.onnx`

Gradle packages deployable TFLite files by copying `models/**/*.tflite` into generated
app assets with `syncCvioBenchmarkModels`. The runtime asset paths stay stable, for
example:

```text
models/efficientnet_b0/efficientnet_b0_float32.tflite
```

The app registry is:

```text
app/src/main/assets/cvio_model_registry.json
```

The packaged label file is:

```text
app/src/main/assets/cvio_labels.txt
```

## Registered Variants

- EfficientNet-B0: TFLite FP32, FP16, dynamic range, full INT8, ONNX FP32.
- MobileNetV3-Large: TFLite FP32, dynamic range, ONNX FP32.
- YOLOv26n-cls: TFLite FP32, FP16, ONNX FP32.

TFLite entries are supported. ONNX entries are listed in the selector as not supported
in this commit.

## Tensor Shapes And Preprocessing

Inspected TFLite tensors:

- All TFLite inputs: `[1, 224, 224, 3]`.
- All TFLite outputs: `[1, 4]`.
- EfficientNet-B0 full INT8 input: `INT8`, scale `0.018590096`, zero point `-14`.
- EfficientNet-B0 full INT8 output: `INT8`, scale `0.047374308`, zero point `-25`.

EfficientNet-B0 and MobileNetV3-Large use registry-configured ImageNet normalization.
YOLOv26n-cls uses 0..1 scaling and has Softmax at the output. Treat preprocessing as
part of the experiment configuration and verify it against the final training/export
pipeline before reporting accuracy.

## Phone Workflow

1. Build and install from PowerShell:

```powershell
.\gradlew.bat clean
.\gradlew.bat assembleDebug --stacktrace
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
.\gradlew.bat :app:installDebug
```

2. Open the app.
3. Choose the model from the top selector.
4. Choose a ground-truth label only if the selected image label is known.
5. Tap `Select Image`.
6. Read top-k predictions and benchmark metrics in the bottom sheet.
7. Tap `Save Log` to save the last result manually, or use the automatic upload log.
8. Tap `Camera` to return to live camera inference.

## Metrics

- `preprocess_ms`: image rotation, center crop, resize, normalization, and input tensor fill.
- `inference_ms`: averaged interpreter runtime only. Upload inference uses warmup runs and
  averages measured runs; live camera uses one measured run per frame.
- `postprocess_ms`: output dequantization, Softmax if configured, sorting, and top-k selection.
- `total_ms`: preprocess plus averaged inference plus postprocess.
- `fps`: `1000 / inference_ms`, based on model inference only.
- `model_size_mb`: file size from the registry.
- `flops`: `N/A` unless a trusted offline FLOPs value is added to the registry.
- `accuracy`: `N/A` unless a ground-truth label is selected. Session accuracy is computed
  only from labeled inferences in the current session.

Do not infer accuracy from one unlabeled image. Use a labeled batch evaluation workflow
for final model accuracy.

## Pull Logs To Windows

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
mkdir .\experiment_results -Force
& $adb pull /sdcard/Android/data/org.tensorflow.lite.examples.imageclassification/files/cvio_metrics .\experiment_results\cvio_metrics
```

Live logcat filter:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb logcat -c
& $adb logcat | Select-String -Pattern "CVioMetrics|TFLite|ONNX|Inference|Exception|Error"
```

## Add A New TFLite Model

1. Put the file under `models/<architecture>/`.
2. Inspect tensor shape, dtype, quantization params, and output semantics.
3. Add a registry entry in `app/src/main/assets/cvio_model_registry.json`.
4. Add or update labels in `app/src/main/assets/cvio_labels.txt` if class order changes.
5. Keep `flops` as `null` unless measured offline.
6. Rebuild and test on a physical phone.
