# CVio Shrimp Disease Classification

Dự án xây dựng pipeline huấn luyện, tối ưu model và ứng dụng Android chạy offline để phân loại tình trạng bệnh tôm bằng LiteRT/TensorFlow Lite.

## Cấu Trúc Thư Mục

- `LiteRT-for-Android/`: mã nguồn ứng dụng Android Kotlin + Jetpack Compose.
- `mobile_accuracy_optimization/`: model đã tối ưu, báo cáo đánh giá và dataset kiểm thử.
- `onnx_models/`: model ONNX và các bản convert sang TFLite/LiteRT.
- `*.ipynb`: notebook huấn luyện, convert và kiểm thử inference.

## APK Android

Sau khi build, APK debug nằm tại:

```text
LiteRT-for-Android/app/build/outputs/apk/debug/app-debug.apk
```

Ứng dụng yêu cầu điện thoại Android 12 trở lên vì `minSdk = 31`.

## Build Lại APK Trên Windows

1. Mở terminal tại thư mục repo.
2. Chuyển vào project Android:

```powershell
cd LiteRT-for-Android
```

3. Build sạch APK debug:

```powershell
.\gradlew.bat clean assembleDebug
```

Nếu máy chưa nhận JDK/Android SDK, cài Android Studio hoặc thiết lập `JAVA_HOME`, `ANDROID_HOME`, `ANDROID_SDK_ROOT` trước khi chạy Gradle.

## Tải APK Về Điện Thoại Và Cài Đặt

1. Build APK theo bước trên.
2. Copy file `LiteRT-for-Android/app/build/outputs/apk/debug/app-debug.apk` sang điện thoại bằng một trong các cách:
   - Kết nối cáp USB, chọn `File transfer`, copy APK vào thư mục `Download`.
   - Upload APK lên Google Drive/Zalo/Telegram/email, sau đó tải về trên điện thoại.
3. Trên điện thoại, mở ứng dụng `Files` hoặc `My Files`, vào thư mục chứa APK.
4. Chạm vào `app-debug.apk`.
5. Nếu Android hỏi quyền cài ứng dụng từ nguồn không xác định, chọn `Settings` và bật `Allow from this source` cho Files/Chrome/Drive.
6. Quay lại file APK, chọn `Install`, sau đó chọn `Open`.
7. Khi app hỏi quyền camera, chọn `Allow` để dùng tính năng chụp ảnh.

## Cài Bằng ADB

Nếu đã bật Developer options và USB debugging:

```powershell
adb install -r LiteRT-for-Android/app/build/outputs/apk/debug/app-debug.apk
```

Nếu bị lỗi trùng package hoặc khác chữ ký debug, gỡ bản cũ rồi cài lại:

```powershell
adb uninstall rs.smobile.shrimpdisease
adb install LiteRT-for-Android/app/build/outputs/apk/debug/app-debug.apk
```

## Lỗi Thường Gặp

- `App not installed`: gỡ bản app cũ trên điện thoại rồi cài lại APK mới.
- `There was a problem parsing the package`: điện thoại có thể đang dưới Android 12 hoặc file APK copy bị lỗi.
- Camera không mở: vào `Settings > Apps > CVio > Permissions > Camera` và cấp quyền camera.
- Model không load: đảm bảo các file `.tflite` và `labels.txt` vẫn nằm trong `LiteRT-for-Android/app/src/main/assets/`.
