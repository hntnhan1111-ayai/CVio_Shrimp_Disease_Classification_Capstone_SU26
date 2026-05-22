# CVio Shrimp Disease Classification

CVio là ứng dụng Android chạy offline để phân loại tình trạng bệnh tôm bằng LiteRT/TensorFlow Lite. Ứng dụng hỗ trợ chọn ảnh từ thư viện, chụp ảnh bằng camera, loại nền bằng U2Net/rembg TFLite, chạy mô hình phân loại bệnh và lưu lịch sử dự đoán cho farmer/admin.

## Tính Năng Chính

- Phân loại 4 nhãn trong `assets/labels.txt`: `Healthy`, `BG`, `WSSV`, `WSSV_BG`.
- Chạy inference trực tiếp trên thiết bị Android bằng LiteRT/TensorFlow Lite.
- Tiền xử lý ảnh có remove background bằng model `u2net_dynamic_range_int8.tflite`.
- Hỗ trợ nhiều model `.tflite` đóng gói trong assets và cho phép đổi model trong app.
- Lưu lịch sử dự đoán, confidence, top-3 predictions, thời gian inference, FPS và threshold.
- Hỗ trợ giao diện farmer và admin.
- Đồng bộ user, farmer profile và metadata dự đoán bằng Firebase Auth + Cloud Firestore.
- Không dùng Firebase Storage/Cloudinary. Firestore chỉ lưu chuỗi `imageUri`/URL nếu có, không upload file ảnh.

## Cấu Trúc Thư Mục

```text
.
+-- LiteRT-for-Android/                  Android app Kotlin + Jetpack Compose
|   +-- app/src/main/assets/             TFLite models và labels.txt
|   +-- app/src/main/java/...            Source code Android
|   +-- firebase/                        Firestore rules và firebase.json
+-- mobile_accuracy_optimization/        Dataset, báo cáo và model tối ưu
+-- onnx_models/                         ONNX models và bản convert sang TFLite
+-- *.ipynb                              Notebook train/convert/evaluate
+-- README.md
```

## Yêu Cầu

- Windows + PowerShell hoặc terminal tương đương.
- JDK 17.
- Android SDK hoặc Android Studio.
- Thiết bị Android 12 trở lên vì `minSdk = 31`.
- Bật Developer options + USB debugging nếu cài bằng ADB.
- Firebase project nếu muốn dùng đăng nhập/đồng bộ cloud.

## Model Đang Dùng

File cấu hình chính:

```text
LiteRT-for-Android/app/src/main/java/rs/smobile/shrimpdisease/classifier/ModelConfig.kt
```

Mặc định:

```text
Classifier: efficientnet_b0_float16.tflite
Background remover: u2net_dynamic_range_int8.tflite
Labels: labels.txt
```

Các model phải nằm trong:

```text
LiteRT-for-Android/app/src/main/assets/
```

## Firebase

Ứng dụng hiện dùng:

```text
Firebase Authentication: Email/Password
Cloud Firestore: users, farmer_profiles, prediction_logs
```

Ứng dụng không dùng Firebase Storage, nên không cần nâng cấp Blaze chỉ để lưu ảnh. Trường `imageUri` trong Firestore chỉ là metadata URL/URI. Nếu ảnh được chọn từ thư viện Android và URI có dạng `content://...`, URI đó thường chỉ dùng được trên chính thiết bị farmer.

### Cấu Hình Firebase Cho Android

1. Tạo Firebase Android app với package:

```text
rs.smobile.shrimpdisease
```

2. Tải file `google-services.json` từ Firebase Console và đặt tại:

```text
LiteRT-for-Android/app/google-services.json
```

3. Bật Email/Password:

```text
Firebase Console > Authentication > Sign-in method > Email/Password > Enable
```

4. Tạo Firestore database ở Native mode. Project hiện tại đang dùng region:

```text
asia-southeast1
```

### Deploy Firestore Rules

Firebase rules nằm ở:

```text
LiteRT-for-Android/firebase/firestore.rules
```

Deploy bằng Firebase CLI:

```powershell
cd LiteRT-for-Android/firebase
npx.cmd --yes firebase-tools deploy --only firestore:rules --project mobile-shrimpsidease
```

## Build APK Debug

Từ thư mục gốc repo:

```powershell
cd LiteRT-for-Android
.\gradlew.bat assembleDebug
```

APK sau khi build nằm tại:

```text
LiteRT-for-Android/app/build/outputs/apk/debug/app-debug.apk
```

Build sạch nếu cần:

```powershell
.\gradlew.bat clean assembleDebug
```

## Cài APK Lên Android

Dùng `adb` nếu đã bật USB debugging:

```powershell
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

Nếu máy chưa nhận `adb`, dùng `adb.exe` trong Android SDK của project:

```powershell
.\android-sdk\platform-tools\adb.exe install -r .\app\build\outputs\apk\debug\app-debug.apk
```

Nếu lỗi chữ ký hoặc package cũ:

```powershell
adb uninstall rs.smobile.shrimpdisease
adb install "LiteRT-for-Android/app/build/outputs/apk/debug/app-debug.apk"
```

## Tài Khoản Và Role

App có flow farmer/admin. User và role được lưu trong Firestore collection:

```text
users
```

Ví dụ document admin:

```text
users/{adminUid}
account: "admin@example.com"
role: "Admin"
displayName: "CVio Admin"
createdAt: <milliseconds>
```

Farmer đăng ký bằng app sẽ được tạo với role `Farmer`.

## Dữ Liệu Firestore

Các collection chính:

```text
users
farmer_profiles
prediction_logs
```

`prediction_logs` lưu metadata như:

```text
ownerId
imageUri
predictedClass
confidence
top3Predictions
modelName
threshold
preprocessing/inference metrics
timestamp
```

## Lưu Ý Khi Push Git

- Không commit `local.properties`.
- Không commit thư mục build: `.gradle/`, `build/`, `app/build/`.
- Nếu repo public, cân nhắc không commit `google-services.json`; người clone repo có thể tự tải từ Firebase Console. File này không phải API secret, nhưng vẫn là cấu hình project Firebase của bạn.
- Các file `.tflite` có dung lượng lớn; nếu repo public hoặc có giới hạn dung lượng, cân nhắc dùng Git LFS hoặc release assets.

## Lỗi Thường Gặp

- `App not installed`: gỡ app cũ rồi cài lại APK mới.
- `There was a problem parsing the package`: thiết bị có thể dưới Android 12 hoặc file APK copy bị lỗi.
- Camera không mở: cấp quyền camera trong `Settings > Apps > CVio > Permissions`.
- Model không load: kiểm tra `.tflite` và `labels.txt` trong `app/src/main/assets/`.
- Firebase không đồng bộ: kiểm tra `google-services.json`, đã bật Email/Password, Firestore đã tạo database và rules đã deploy.
