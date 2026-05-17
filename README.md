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
```

## What Is Included

- CameraX-based live image classification on a physical device.
- Default demo models in `app/src/main/assets/`.
- TensorFlow Lite Task Vision inference helpers and UI controls for delegate/model benchmarking.
- A runbook for Windows setup and phone testing in `docs/runbooks/ANDROID_DEMO_WINDOWS.md`.

## Repository Notes

- Codebase map: `docs/CODEBASE_INDEX.md`
- Windows runbook: `docs/runbooks/ANDROID_DEMO_WINDOWS.md`
- Future shrimp model swap notes are documented in both files.
