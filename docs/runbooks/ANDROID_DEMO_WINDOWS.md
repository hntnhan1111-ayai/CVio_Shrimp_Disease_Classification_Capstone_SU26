# Android Demo Runbook For Windows

## Purpose

This repo branch is for Android demo and benchmark work only.
Use it to validate camera inference, model swaps, and latency/FPS behavior on a real phone.
Do not use it as the final production mobile app branch.

## Prerequisites

- Windows with Git installed.
- Android Studio with Android SDK installed.
- Embedded JDK / JDK 17 selected in Android Studio under Gradle settings.
- A physical Android phone with USB debugging enabled.
- A USB cable that supports data transfer.

## Clone And Checkout The Demo Branch

```powershell
cd C:\Users\nhant\CVio
git clone https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26.git
cd CVio_Shrimp_Disease_Classification_Capstone_SU26
git checkout demo/android-tflite-benchmark
```

If the repo is already cloned locally:

```powershell
cd C:\Users\nhant\CVio\CVio_Shrimp_Disease_Classification_Capstone_SU26
git fetch origin --prune
git checkout demo/android-tflite-benchmark
```

## Open In Android Studio

1. File > Open.
2. Select the repository root folder.
3. Wait for Gradle sync.
4. If prompted, choose Embedded JDK / JDK 17 for Gradle.

## Build And Run

From the project root in PowerShell:

```powershell
.\gradlew.bat clean
.\gradlew.bat assembleDebug --stacktrace
```

Then run the app from Android Studio, or install the debug APK manually:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
.\gradlew.bat :app:installDebug
```

The Android package is:

```text
org.tensorflow.lite.examples.imageclassification
```

## Enable USB Debugging

1. Open Settings on the phone.
2. Tap About phone.
3. Tap Build number seven times to enable Developer options.
4. Open Developer options.
5. Enable USB debugging.
6. Reconnect the phone and accept the RSA prompt.

## Troubleshooting

### Unauthorized Device

- Disconnect and reconnect the cable.
- Accept the RSA fingerprint prompt on the phone.
- Run `adb kill-server` followed by `adb start-server`.
- If the prompt does not reappear, revoke USB debugging authorizations on the phone and try again.

### No Device Found

- Confirm the phone is in file transfer / data mode, not charge-only mode.
- Try a different USB cable or port.
- Re-run `adb devices`.
- Verify the platform tools are installed and visible in Android Studio SDK Manager.

### Gradle Sync Failure

- Confirm Gradle JDK is set to JDK 17.
- Confirm Android SDK Platform 35 is installed.
- Run `.\gradlew.bat --version` to confirm the wrapper is 8.7.
- Clear stale build outputs with `.\gradlew.bat clean`.

### `processDebugResources` Failure

- Delete `build/` and `.gradle/` in the repo if they become stale.
- Re-sync the project in Android Studio.
- Make sure the required Android SDK platform and build tools are installed.
- Check for XML errors in `app/src/main/res/`.

## Benchmarking CVio Models

1. Open the app on the phone.
2. Use the model selector near the top of the screen. It is above the bottom sheet and
   protected from gesture/navigation-bar overlap.
3. Select a TFLite model variant.
4. Tap `Select Image` to choose a shrimp image from the phone.
5. Optionally choose the ground-truth label before running the image.
6. Review top-k predictions and metrics in the bottom sheet.
7. Tap `Camera` to return to live camera inference.

Uploaded-image inference automatically appends a CSV row to:

```text
/sdcard/Android/data/org.tensorflow.lite.examples.imageclassification/files/cvio_metrics/inference_metrics.csv
```

Use PowerShell to pull logs:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
mkdir .\experiment_results -Force
& $adb pull /sdcard/Android/data/org.tensorflow.lite.examples.imageclassification/files/cvio_metrics .\experiment_results\cvio_metrics
```

For live logs:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb logcat -c
& $adb logcat | Select-String -Pattern "CVioMetrics|TFLite|ONNX|Inference|Exception|Error"
```

## Swapping In A Future Shrimp Model

1. Copy the new `.tflite` file under root `models/`.
2. Add an entry to `app/src/main/assets/cvio_model_registry.json`.
3. Confirm input tensor shape, dtype, quantization, preprocessing, and output semantics.
4. Rebuild. Gradle copies `.tflite` files into generated app assets through
   `syncCvioBenchmarkModels`.
5. Retest on a physical phone.

## Notes

- The first Gradle build downloads the demo models automatically through `app/download_models.gradle`.
- CVio `.tflite` model files are copied from root `models/` during Gradle build.
- ONNX files are registered in the UI as unsupported in this commit; see `docs/ONNX_RUNTIME_TODO.md`.
- The app is designed for a real Android device, not the emulator.
- A compileSdk 35 warning from AGP 8.5.2 is expected and does not block the build.
