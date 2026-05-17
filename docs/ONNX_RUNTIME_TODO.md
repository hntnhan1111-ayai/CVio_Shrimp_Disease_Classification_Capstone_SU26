# ONNX Runtime TODO

ONNX model files are registered and visible in the Android selector, but ONNX Runtime
Android is not enabled in this commit. Selecting an ONNX entry shows that it is not
supported, and live/upload inference remains disabled for that entry.

## Registered ONNX Files

- `models/efficientnet_b0/efficientnet_b0.onnx`
- `models/mobilenet_v3_large/mobilenet_v3_large.onnx`
- `models/yolov26n-cls/yolov26n-cls.onnx`

## Inspected ONNX Shapes

- EfficientNet-B0 input: `input`, `[batch, 3, 224, 224]`, output `logits`, `[batch, 4]`.
- MobileNetV3-Large input: `input`, `[batch, 3, 224, 224]`, output `logits`, `[batch, 4]`.
- YOLOv26n-cls input: `images`, `[1, 3, 224, 224]`, output `output0`, `[1, 4]`.

## Implementation Work Remaining

1. Add the ONNX Runtime Android Gradle dependency.
2. Introduce a `ClassifierRunner` interface if more runtime implementations are added.
3. Implement `OnnxClassifierRunner` using `OrtEnvironment` and `OrtSession`.
4. Package ONNX files as assets only after confirming APK size and asset loading behavior.
5. Support NCHW input buffers and model-specific preprocessing from the registry.
6. Confirm whether outputs are logits or probabilities for each ONNX model.
7. Add ONNX rows to the same CSV metrics logger with runtime `ONNX Runtime Android`.
8. Rebuild and test on a real Android device before enabling ONNX entries.

FLOPs should remain `N/A` until measured offline with a trusted tool.
