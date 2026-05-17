# Codebase Index

## Branch Purpose

This repository state is the Android demo/testing branch only.
Use it for real-device camera inference, image classification checks, latency/FPS
benchmarking, and model comparison experiments. It is not the final production
mobile app.

## Repository Purpose

- Android TensorFlow Lite / LiteRT image classification demo for real-device testing.
- Benchmarking branch for camera inference, image swaps, latency, FPS, and future shrimp disease model trials.

## Root Structure

- `README.md` - quick-start summary and links to the Windows runbook.
- `build.gradle` - root Gradle configuration; AGP `8.5.2`, Kotlin `1.9.24`, navigation safe-args, and the Gradle download task plugin.
- `settings.gradle` - includes the `:app` module and dependency resolution rules.
- `gradle/wrapper/gradle-wrapper.properties` - pins Gradle `8.7`.
- `gradle/` - Gradle wrapper files.
- `app/` - Android application module.
- `docs/` - runbooks and codebase index for future debugging.
- `screenshot1.jpg`, `screenshot2.jpg` - sample UI screenshots used by the demo.

## Android Module Map

- Module path: `app/`
- Android manifest: `app/src/main/AndroidManifest.xml`
- Application ID and namespace: `org.tensorflow.lite.examples.imageclassification`
- Main activity: `app/src/main/java/org/tensorflow/lite/examples/imageclassification/MainActivity.kt`
- Permission gate: `app/src/main/java/org/tensorflow/lite/examples/imageclassification/fragments/PermissionsFragment.kt`
- Camera pipeline: `app/src/main/java/org/tensorflow/lite/examples/imageclassification/fragments/CameraFragment.kt`
- Inference helper: `app/src/main/java/org/tensorflow/lite/examples/imageclassification/ImageClassifierHelper.kt`
- Results adapter: `app/src/main/java/org/tensorflow/lite/examples/imageclassification/fragments/ClassificationResultsAdapter.kt`

## Gradle And Build Files

- `app/build.gradle` - Android application module config, dependencies, `namespace`, `compileSdk`, and the model download hook.
- `app/download_models.gradle` - downloads the default `.tflite` models into `app/src/main/assets/`.
- `app/proguard-rules.pro` - module shrinker rules.
- `app/src/main/AndroidManifest.xml` - camera permission, launcher activity, and app metadata.

## Resources, Assets, And Labels

- UI layouts: `app/src/main/res/layout/activity_main.xml`, `fragment_camera.xml`, `info_bottom_sheet.xml`, `item_classification_result.xml`
- Navigation: `app/src/main/res/navigation/nav_graph.xml`
- Strings and UI labels: `app/src/main/res/values/strings.xml`
- Demo model assets: `app/src/main/assets/mobilenetv1.tflite`, `efficientnet-lite0.tflite`, `efficientnet-lite1.tflite`, `efficientnet-lite2.tflite`
- There is no standalone `labels.txt` file in this sample. Category labels are surfaced from TensorFlow Lite Task Vision result objects in `ClassificationResultsAdapter.kt`.
- Test assets exist under `app/src/androidTest/assets/`, including `coffee.jpg` and `mobilenetv1.tflite`.

## How The App Runs

1. `MainActivity` hosts the navigation graph.
2. `PermissionsFragment` requests camera permission.
3. `CameraFragment` starts a CameraX preview and frame analyzer.
4. `ImageClassifierHelper` loads the selected model from assets and runs inference.
5. `ClassificationResultsAdapter` renders the top classification categories.

## Useful Windows Debug Commands

```powershell
git status --short
git branch --show-current
.\gradlew.bat assembleDebug --stacktrace
.\gradlew.bat :app:installDebug --stacktrace
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
& $adb logcat -v time
& $adb shell pm list packages | Select-String -Pattern "org.tensorflow.lite.examples.imageclassification"
```

## Known Stacktrace Areas

- Gradle sync and dependency resolution
- `adb` authorization or wireless pairing
- Model loading from `app/src/main/assets/`
- Camera permission and CameraX startup
- TensorFlow Lite inference, delegate selection, and `processDebugResources`

## Known Issues And Warnings

- AGP `8.5.2` prints a warning when `compileSdk = 35`; the build still succeeds.
- If Android Studio and command-line tools are out of sync, you may see an SDK XML version warning.
- The app depends on network access the first time the Gradle download task fetches models.
- Camera testing requires a physical device and accepted USB debugging authorization.
- Future shrimp disease support requires replacing the demo `.tflite` model and, if needed, adding matching labels metadata or a `labels.txt` asset.

## Future Model Swap Notes

To replace the demo model with a shrimp disease `.tflite` file, place the new model in
`app/src/main/assets/`, update the model selection logic in `ImageClassifierHelper.kt`,
and adjust the spinner labels in `app/src/main/res/values/strings.xml` if you want the UI
to expose the new model name.