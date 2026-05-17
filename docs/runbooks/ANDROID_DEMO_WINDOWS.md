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
adb devices
adb install -r app\build\outputs\apk\debug\app-debug.apk
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
- Run `.
gradlew.bat --version` to confirm the wrapper is 8.7.
- Clear stale build outputs with `.
gradlew.bat clean`.

### `processDebugResources` Failure

- Delete `build/` and `.gradle/` in the repo if they become stale.
- Re-sync the project in Android Studio.
- Make sure the required Android SDK platform and build tools are installed.
- Check for XML errors in `app/src/main/res/`.

## Swapping In A Future Shrimp Model

1. Copy the new `.tflite` file into `app/src/main/assets/`.
2. Update `ImageClassifierHelper.kt` so the selected model name points to the new file.
3. Update `app/src/main/res/values/strings.xml` if the spinner should show the new model name.
4. If you only want one shrimp model, simplify the model spinner in `info_bottom_sheet.xml` and `CameraFragment.kt`.
5. Rebuild and retest on a physical phone.

## Notes

- The first Gradle build downloads the demo models automatically through `app/download_models.gradle`.
- The app is designed for a real Android device, not the emulator.
- A compileSdk 35 warning from AGP 8.5.2 is expected and does not block the build.