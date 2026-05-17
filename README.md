# CVio Shrimp Disease Classification - Android Demo

This branch, `demo/android-tflite-benchmark`, is for Android real-device testing only.
It packages the TensorFlow Lite / LiteRT image classification demo for camera inference,
model swaps, and benchmark experiments. It is not the final production mobile app.

## Quick Start

Open the repo in Android Studio, then set the Gradle JDK to Embedded JDK / JDK 17 and sync.
Use a real Android phone. Android 11+ devices can also use wireless debugging; the full steps
are in [docs/runbooks/ANDROID_DEMO_WINDOWS.md](docs/runbooks/ANDROID_DEMO_WINDOWS.md).

From PowerShell in the repo root:

```powershell
.\gradlew.bat clean
.\gradlew.bat assembleDebug --stacktrace
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
.\gradlew.bat :app:installDebug --stacktrace
& $adb shell monkey -p org.tensorflow.lite.examples.imageclassification -c android.intent.category.LAUNCHER 1
```

## What Is Included

- CameraX-based live image classification on a physical device.
- Default demo models in `app/src/main/assets/`.
- TensorFlow Lite Task Vision inference helpers and UI controls for delegate/model benchmarking.
- A runbook for Windows setup and phone testing in `docs/runbooks/ANDROID_DEMO_WINDOWS.md`.

## Repository Notes

- Codebase map: [docs/CODEBASE_INDEX.md](docs/CODEBASE_INDEX.md)
- Windows runbook: [docs/runbooks/ANDROID_DEMO_WINDOWS.md](docs/runbooks/ANDROID_DEMO_WINDOWS.md)
- Future shrimp model swap notes are documented in both files.
