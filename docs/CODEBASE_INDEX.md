# Codebase Index

## Branch Purpose

This repository state is the Android demo/testing branch only.
Use it for real-device camera inference, image classification checks, latency/FPS
benchmarking, and model comparison experiments. It is not the final production
mobile app.

## Root Structure

- `README.md` - demo branch overview and quick-start notes.
- `build.gradle` - root Gradle configuration; AGP `8.5.2`, Kotlin `1.9.24`,
  navigation safe-args, and the Gradle download task plugin.
- `settings.gradle` - includes the `:app` module and dependency resolution rules.
- `gradle/wrapper/gradle-wrapper.properties` - pins Gradle `8.7`.
- `gradle/` - Gradle wrapper files.
- `app/` - Android application module.
- `screenshot1.jpg`, `screenshot2.jpg` - sample UI screenshots used by the demo.

## Android Entry Points

- Package / application id / namespace: `org.tensorflow.lite.examples.imageclassification`.
- `app/src/main/java/org/tensorflow/lite/examples/imageclassification/MainActivity.kt` - host activity for the navigation graph.
- `app/src/main/java/org/tensorflow/lite/examples/imageclassification/fragments/PermissionsFragment.kt` - camera permission gate.
- `app/src/main/java/org/tensorflow/lite/examples/imageclassification/fragments/CameraFragment.kt` - CameraX preview, analyzer, and UI controls.
- `app/src/main/java/org/tensorflow/lite/examples/imageclassification/ImageClassifierHelper.kt` - selected-model orchestration, session accuracy, metrics, and logging.
- `app/src/main/java/org/tensorflow/lite/examples/imageclassification/benchmark/BenchmarkModels.kt` - registry data classes, prediction data, and metric data.
- `app/src/main/java/org/tensorflow/lite/examples/imageclassification/benchmark/TfliteClassifierRunner.kt` - raw TensorFlow Lite interpreter runner with tensor shape/dtype/quantization inspection.
- `app/src/main/java/org/tensorflow/lite/examples/imageclassification/benchmark/MetricsLogger.kt` - CSV metrics writer.
- `app/src/main/java/org/tensorflow/lite/examples/imageclassification/fragments/ClassificationResultsAdapter.kt` - renders model outputs.

## Gradle and Build Files

- `app/build.gradle` - Android application module config, dependencies, `namespace`, `compileSdk`, and the model download hook.
- `app/download_models.gradle` - downloads the default `.tflite` models into `app/src/main/assets/`.
- `app/proguard-rules.pro` - module shrinker rules.
- `app/src/main/AndroidManifest.xml` - camera permission, launcher activity, and app metadata.

## Resources and UI

- `app/src/main/res/navigation/nav_graph.xml` - permission flow into the camera screen.
- `app/src/main/res/layout/activity_main.xml` - activity container.
- `app/src/main/res/layout/fragment_camera.xml` - camera preview, top benchmark controls, upload preview, model selector, and results panel.
- `app/src/main/res/layout/info_bottom_sheet.xml` - delegate, threshold, threads, inference timing, and benchmark metrics.
- `app/src/main/res/layout/item_classification_result.xml` - classification result row.
- `app/src/main/res/values/strings.xml` - UI labels and spinner entries.

## Assets, Models, And Inference Data

- `app/src/main/assets/mobilenetv1.tflite`
- `app/src/main/assets/efficientnet-lite0.tflite`
- `app/src/main/assets/efficientnet-lite1.tflite`
- `app/src/main/assets/efficientnet-lite2.tflite`
- `app/src/main/assets/cvio_model_registry.json`
- `app/src/main/assets/cvio_labels.txt`
- `models/efficientnet_b0/*.tflite`
- `models/mobilenet_v3_large/*.tflite`
- `models/yolov26n-cls/*.tflite`

The Gradle task `syncCvioBenchmarkModels` copies deployable `.tflite` files from
root `models/` into generated app assets at build time:
`app/build/generated/assets/cvioBenchmarkModels/models/`.

The labels currently packaged for Android benchmarking are:

- `1. Healthy`
- `2. BG`
- `3. WSSV`
- `4. WSSV_BG`

These labels were recovered from YOLO ONNX metadata. Confirm that they match the
EfficientNet-B0 and MobileNetV3-Large training exports before reporting final accuracy.

## How The App Runs

1. `MainActivity` hosts the navigation graph.
2. `PermissionsFragment` requests camera permission.
3. `CameraFragment` starts a CameraX preview and frame analyzer.
4. `ImageClassifierHelper` loads the selected registry model from assets and runs inference.
5. `ClassificationResultsAdapter` renders the top classification categories.
6. Upload mode uses the Android photo picker on Android 13+ or `ACTION_OPEN_DOCUMENT`
   fallback, then logs uploaded-image inference metrics automatically.

## Build And Run Notes For Windows

- Use Android Studio with Embedded JDK / JDK 17.
- Sync the project after opening the repo root.
- Build from the project root with `.\gradlew.bat clean` and
  `.\gradlew.bat assembleDebug --stacktrace`.
- Run on a physical Android phone with USB debugging enabled.

## Known Issues And Warnings

- AGP `8.5.2` prints a warning when `compileSdk = 35`; the build still succeeds.
- If Android Studio and command-line tools are out of sync, you may see an SDK XML
  version warning. Update Android SDK command-line tools or Android Studio to match.
- The app depends on network access the first time the Gradle download task fetches
  models.
- Camera testing requires a physical device and accepted USB debugging authorization.
- A `processDebugResources` failure usually points to missing SDK platforms, an old
  Gradle JDK, or stale build artifacts in `build/` or `.gradle/`.

## Registered CVio Model Variants

- EfficientNet-B0: TFLite FP32, FP16, dynamic range, full INT8, ONNX FP32.
- MobileNetV3-Large: TFLite FP32, dynamic range, ONNX FP32.
- YOLOv26n-cls: TFLite FP32, FP16, ONNX FP32.

ONNX entries are visible in the selector as unsupported in this commit. See
`docs/ONNX_RUNTIME_TODO.md`.

## Future Model Swap Notes

To add another shrimp disease `.tflite` file, place it under root `models/`, add a
registry entry in `app/src/main/assets/cvio_model_registry.json`, and rebuild. Do not
invent FLOPs or accuracy values in the registry; use `null` until measured offline or
computed from a labeled evaluation workflow.
