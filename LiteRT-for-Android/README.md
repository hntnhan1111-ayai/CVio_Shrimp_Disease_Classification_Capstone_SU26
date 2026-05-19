# CVio Android App

Ứng dụng Android dùng LiteRT/TensorFlow Lite để phân loại ảnh tôm theo 4 nhãn trong `assets/labels.txt`:

- `Healthy`
- `BG`
- `WSSV`
- `WSSV_BG`

App chạy inference trực tiếp trên điện thoại, không cần internet sau khi cài đặt.

## Tính Năng

- Chọn ảnh từ thư viện hoặc chụp snapshot bằng camera.
- Chọn model `.tflite` đã đóng gói trong `app/src/main/assets/`.
- Chọn nhãn ground-truth để tính accuracy trong phiên chạy.
- Hiển thị confidence, thời gian inference, FPS camera, threshold, số lần accept/reject.
- Export inference logs ra file `.txt`.

## Yêu Cầu

- Android Studio hoặc Android SDK + JDK 17.
- Điện thoại Android 12 trở lên (`minSdk = 31`).
- Camera permission nếu muốn dùng tính năng chụp ảnh.

## Build APK

Tại thư mục này, chạy:

```powershell
.\gradlew.bat clean assembleDebug
```

APK sau khi build:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Debug APK đã được Gradle ký bằng debug key nên có thể cài trực tiếp lên điện thoại để kiểm thử.

## Cài APK Trên Điện Thoại

1. Copy `app/build/outputs/apk/debug/app-debug.apk` sang điện thoại.
2. Mở file APK bằng `Files`, `My Files`, Chrome hoặc Google Drive.
3. Bật quyền `Install unknown apps` cho ứng dụng đang mở APK nếu Android yêu cầu.
4. Chọn `Install`.
5. Mở app `CVio`.
6. Cấp quyền camera khi app hỏi.

Nếu Play Protect cảnh báo vì đây là APK tự build, chọn phần chi tiết và tiếp tục cài đặt nếu bạn tin tưởng file APK được build từ repo này.

## Cài Bằng ADB

```powershell
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Gỡ bản cũ nếu Android báo lỗi chữ ký/package:

```powershell
adb uninstall rs.smobile.shrimpdisease
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Cấu Hình Model

Các file chính:

- `app/src/main/assets/*.tflite`: model LiteRT/TensorFlow Lite.
- `app/src/main/assets/labels.txt`: thứ tự nhãn phải khớp output của model.
- `app/src/main/java/rs/smobile/shrimpdisease/classifier/ModelConfig.kt`: model mặc định, input size, threshold, số thread.

Khi thay model:

1. Copy model `.tflite` vào `app/src/main/assets/`.
2. Cập nhật `MODEL_FILE` trong `ModelConfig.kt` nếu muốn model đó là mặc định.
3. Kiểm tra `labels.txt` có đúng số lớp và đúng thứ tự class khi train.
4. Build lại APK.

## Lỗi Thường Gặp

- `App not installed`: gỡ app cũ rồi cài lại APK mới.
- `There was a problem parsing the package`: thiết bị đang dưới Android 12 hoặc file APK copy bị lỗi.
- `Failed to load model`: kiểm tra model `.tflite` và `labels.txt` trong `assets/`.
- Camera không hoạt động: vào `Settings > Apps > CVio > Permissions` và cấp quyền camera.
