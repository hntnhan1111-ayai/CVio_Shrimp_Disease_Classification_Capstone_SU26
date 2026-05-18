# Chạy Shrimp Disease LiteRT Android

File này là checklist nhanh cho project Android trong thư mục `LiteRT-for-Android`.

## 1. Mở Project

Mở thư mục `LiteRT-for-Android` bằng Android Studio hoặc terminal. Dùng JDK 17 và Android SDK.

Nếu cần khai báo SDK thủ công, tạo/cập nhật `local.properties`:

```properties
sdk.dir=D\:/path/to/Android/Sdk
```

## 2. Model Và Label

Model và label được đóng gói trong:

```text
app/src/main/assets/
```

File cấu hình mặc định:

```text
app/src/main/java/rs/smobile/shrimpdisease/classifier/ModelConfig.kt
```

`labels.txt` phải khớp đúng số lớp và thứ tự class của model:

```text
Healthy
BG
WSSV
WSSV_BG
```

## 3. Build APK

Windows:

```powershell
.\gradlew.bat clean assembleDebug
```

Linux/macOS:

```bash
./gradlew clean assembleDebug
```

APK output:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 4. Cài APK

Cài bằng ADB:

```powershell
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Hoặc copy `app-debug.apk` sang điện thoại, mở bằng Files/Chrome/Drive, bật `Install unknown apps` nếu Android yêu cầu, rồi chọn `Install`.

## 5. Chạy App

Mở app **Shrimp Disease** trên điện thoại.

- `Choose image`: chọn ảnh từ thư viện.
- `Load camera`: mở camera preview.
- `Take snapshot`: chụp khung hình hiện tại để phân loại.
- `Load model`: đổi model `.tflite` đã đóng gói trong assets.
- `Export logs`: xuất log inference ra file `.txt`.

Tất cả inference chạy offline bằng LiteRT/TensorFlow Lite, không gọi server.

## 6. Lỗi Thường Gặp

- `SDK not found`: kiểm tra `ANDROID_HOME`, `ANDROID_SDK_ROOT` hoặc `sdk.dir`.
- `adb unauthorized`: mở khóa điện thoại và bấm `Allow USB debugging`.
- Camera không mở: vào `Settings > Apps > Shrimp Disease > Permissions > Camera`.
- `labels count mismatch`: sửa `labels.txt` để số dòng và thứ tự class khớp model.
- `model input shape mismatch`: app chỉ hỗ trợ tensor ảnh RGB 4D `[1, H, W, 3]` hoặc `[1, 3, H, W]`.
