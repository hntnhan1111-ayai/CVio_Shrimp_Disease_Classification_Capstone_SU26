# Android Demo Runbook For Windows

## Purpose

This repo branch is for Android demo and benchmark work only.
Use it to validate camera inference, model swaps, and latency/FPS behavior on a real phone.
Do not use it as the final production mobile app branch.

## 1. Prerequisites

- Git for Windows
- Android Studio
- Android SDK Platform-Tools
- JDK 17 / Android Studio Embedded JDK
- A real Android phone
- USB cable for wired debugging or Android 11+ for wireless debugging

Verify the basic tools:

```powershell
git --version
Get-ChildItem -Force
Test-Path .\gradlew.bat
```

## 2. Clone And Checkout The Demo Branch

```powershell
cd D:\CVio
git clone https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26.git
cd .\CVio_Shrimp_Disease_Classification_Capstone_SU26
git fetch origin --prune
git checkout demo/android-tflite-benchmark
git pull origin demo/android-tflite-benchmark
git status --short
```

If the repo is already cloned locally, use the same folder and repeat the checkout command.

## 3. Open In Android Studio

1. File > Open.
2. Select the repository root folder.
3. Wait for Gradle sync.
4. In Settings > Build, Execution, Deployment > Build Tools > Gradle, set Gradle JDK to Embedded JDK / JDK 17.
5. Confirm Android SDK Platform-Tools are installed in SDK Manager.

## 4. Build The App

From the repo root in PowerShell:

```powershell
.\gradlew.bat clean
.\gradlew.bat assembleDebug --stacktrace
```

If you only want to install directly:

```powershell
.\gradlew.bat :app:installDebug --stacktrace
```

## 5. USB Debugging Setup

1. On the phone, enable Developer options by tapping Build number 7 times.
2. Turn on USB debugging.
3. Connect the phone with a data-capable cable.
4. Unlock the phone and accept the RSA prompt.

For Xiaomi, Redmi, and POCO devices, also enable USB debugging under Security settings and Install via USB if the option exists.

For Samsung, Pixel, OPPO, Vivo, and Realme devices, install the OEM USB driver if Windows does not detect the phone.

## 6. Wireless Debugging For Android 11+

Wireless debugging requires Android 11 or newer, the phone and laptop on the same Wi-Fi network, and no VPN.
If campus or company Wi-Fi blocks local device-to-device traffic, use phone hotspot or home Wi-Fi instead.

Phone side:

1. Settings > Developer options > Wireless debugging > On.
2. Tap Pair device with pairing code.
3. Keep the pairing popup open while you connect.

PowerShell:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"

& $adb version
& $adb kill-server
& $adb start-server
& $adb pair PHONE_IP:PAIRING_PORT PAIRING_CODE
& $adb connect PHONE_IP:DEBUG_PORT
& $adb devices -l
```

If pairing fails, toggle Wireless debugging off/on, use the fresh pairing code, and try again.

## 7. Verify The Device Is Connected

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"

& $adb devices -l
```

Expected state is `device`, not `unauthorized` or `offline`.

## 8. Install And Verify The App

```powershell
.\gradlew.bat :app:installDebug --stacktrace
& $adb shell pm list packages | Select-String -Pattern "org.tensorflow.lite.examples.imageclassification"
& $adb shell dumpsys package org.tensorflow.lite.examples.imageclassification | Select-String -Pattern "versionName|versionCode|firstInstallTime|lastUpdateTime"
```

If you need the APK directly, the expected debug output is:

```powershell
& $adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

## 9. Launch The App

From the phone UI, open the app icon after install.

From ADB:

```powershell
& $adb shell monkey -p org.tensorflow.lite.examples.imageclassification -c android.intent.category.LAUNCHER 1
& $adb shell am start -n org.tensorflow.lite.examples.imageclassification/.MainActivity
```

## 10. Inspect Logs

```powershell
& $adb logcat -c
& $adb logcat -v time | Select-String -Pattern "tflite|tensorflow|classification|inference|latency|fps|camera|exception|error|fatal"
```

## 11. Model Swap Notes

The sample currently uses ImageNet demo models from `app/src/main/assets/` and does not ship a standalone `labels.txt` file.
To add a shrimp disease model later, place the `.tflite` file in `app/src/main/assets/`, update `ImageClassifierHelper.kt`, and add matching labels metadata or a `labels.txt` asset if your model requires it.

## 12. Common Issues And Fixes

### Unauthorized Device

- Unlock the phone and accept the RSA popup.
- Choose Always allow if you trust the machine.
- If the popup does not appear, go to Developer options and revoke USB debugging authorizations, toggle USB debugging off/on, then reconnect.
- On Windows, remove adb keys only if needed:

```powershell
Remove-Item "$env:USERPROFILE\.android\adbkey*" -Force -ErrorAction SilentlyContinue
```

### No Device Found

- Use a cable that supports data, not charge-only.
- Set the phone USB mode to File Transfer/MTP.
- Run `adb kill-server` and `adb start-server` again.
- Install the OEM USB driver if Windows still does not see the device.

### Wireless Pairing Failure

- Make sure the phone pairing popup is still open.
- Use the current pairing port and code from the phone screen.
- Turn Wireless debugging off/on to get a fresh pairing code.
- Disable VPN.
- Switch to hotspot or home Wi-Fi if the current network blocks local traffic.

### Gradle Warning About compileSdk 35

- AGP 8.5.2 warns that it was tested up to compileSdk 34.
- Treat this as a warning unless the build actually fails.
- If the team intentionally accepts the warning, document `android.suppressUnsupportedCompileSdk=35` in `gradle.properties`.

### installDebug Failure

- Check `adb devices -l` first.
- If the device is offline or unauthorized, fix ADB before retrying.
- If you get an update mismatch, uninstall the old app and retry.
- Re-run `.
gradlew.bat :app:installDebug --stacktrace` after the device is ready.

### Camera Permission

- Open App info > Permissions > Camera > Allow.
- If the app still shows a permission issue, clear the app and reinstall.
- Use `adb logcat` to confirm the permission request flow.

### Logcat Debugging

- Use `adb logcat -v time` and filter on `tflite`, `camera`, `error`, `fatal`, or `exception`.
- If logs are noisy, clear them with `adb logcat -c` before reproducing the issue.

### Package Launch

- Verify the package name is `org.tensorflow.lite.examples.imageclassification`.
- Launch it from the phone icon after install, or use `adb shell monkey` if you want an ADB launch.

## 13. Replacing The Demo With A Future Shrimp Model

1. Copy the shrimp `.tflite` model into `app/src/main/assets/`.
2. Update the model selection mapping in `ImageClassifierHelper.kt`.
3. Add or update labels metadata or `labels.txt` for the new model if required.
4. Adjust the UI spinner labels in `app/src/main/res/values/strings.xml` if you want the new model name exposed in the app.
5. Rebuild and retest on a physical phone.

## 14. Notes

- The first Gradle build downloads the demo models automatically through `app/download_models.gradle`.
- The app is designed for a real Android device, not the emulator.
- A compileSdk 35 warning from AGP 8.5.2 is expected and does not block the build.