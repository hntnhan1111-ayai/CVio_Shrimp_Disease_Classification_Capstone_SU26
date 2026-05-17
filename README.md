# CVio Shrimp Disease Classification - Android Demo

This branch, `demo/android-tflite-benchmark`, is for Android real-device testing only.
It contains the TensorFlow Lite image classification sample for camera inference,
model swaps, and benchmark experiments. It is not the final production mobile app.

## Quick Start

Open the repo in Android Studio, select Embedded JDK / JDK 17 for Gradle, sync the
project, and run it on a physical Android phone with USB debugging enabled.

From PowerShell in the repo root:

```powershell
.\gradlew.bat clean
.\gradlew.bat assembleDebug --stacktrace
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
.\gradlew.bat :app:installDebug
```

## What Is Included

- CameraX-based live image classification on a physical device.
- CVio model selector for all converted variants registered from `models/`.
- Upload/select-image inference from the Android photo picker or file picker.
- Raw TensorFlow Lite benchmarking for FP32, FP16, dynamic range, and full INT8 models.
- Optional ground-truth selection for correctness and session accuracy.
- CSV metrics logging to the app-specific external files directory.
- ONNX model entries are visible in the UI as not supported in this commit.
- A runbook for Windows setup and phone testing in `docs/runbooks/ANDROID_DEMO_WINDOWS.md`.
- A deployment and benchmarking runbook in `docs/runbooks/MODEL_DEPLOYMENT_AND_BENCHMARKING.md`.

## Repository Notes

- Codebase map: `docs/CODEBASE_INDEX.md`
- Windows runbook: `docs/runbooks/ANDROID_DEMO_WINDOWS.md`
- Model benchmarking runbook: `docs/runbooks/MODEL_DEPLOYMENT_AND_BENCHMARKING.md`
- ONNX follow-up: `docs/ONNX_RUNTIME_TODO.md`

## Pull Metrics Logs To Windows

The Android package is `org.tensorflow.lite.examples.imageclassification`.

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
mkdir .\experiment_results -Force
& $adb pull /sdcard/Android/data/org.tensorflow.lite.examples.imageclassification/files/cvio_metrics .\experiment_results\cvio_metrics
```
