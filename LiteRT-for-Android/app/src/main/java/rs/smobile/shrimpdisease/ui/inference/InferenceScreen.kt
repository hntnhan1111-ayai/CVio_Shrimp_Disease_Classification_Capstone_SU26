package rs.smobile.shrimpdisease.ui.inference

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.core.tween
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import rs.smobile.shrimpdisease.CameraCaptureCard
import rs.smobile.shrimpdisease.ClassificationUiState
import rs.smobile.shrimpdisease.InferenceInputUiState
import rs.smobile.shrimpdisease.classifier.ClassificationResult
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.ui.components.CameraScanFrame
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CompactInfoRow
import rs.smobile.shrimpdisease.ui.components.EmptyState
import rs.smobile.shrimpdisease.ui.components.MetricsCard
import rs.smobile.shrimpdisease.ui.components.PermissionCard
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.components.ResultStatusBadge
import rs.smobile.shrimpdisease.ui.components.ResultStatusKind
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.components.ShrimpIconButton
import rs.smobile.shrimpdisease.ui.components.ShrimpNavIcon
import rs.smobile.shrimpdisease.ui.components.TopPredictionList
import rs.smobile.shrimpdisease.ui.theme.CVioPrimaryFixed
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
import rs.smobile.shrimpdisease.ui.theme.DarkNavy
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.ui.theme.WarningOrange
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import rs.smobile.shrimpdisease.utils.DiseaseTextUtils
import java.util.Locale

@Composable
fun InferenceScreen(
    inputState: InferenceInputUiState,
    classificationState: ClassificationUiState,
    labels: List<String>,
    selectedGroundTruthLabel: String?,
    modelInfo: ModelInfo,
    benchmarkMetrics: BenchmarkMetrics,
    runtimeDelegateName: String,
    debugInfoEnabled: Boolean,
    cameraFps: Double?,
    cameraPermissionGranted: Boolean,
    onRequestCameraPermission: () -> Unit,
    onOpenCamera: () -> Unit,
    onSnapshot: (Bitmap) -> Unit,
    onFrameObserved: () -> Unit,
    onCameraError: (String) -> Unit,
    onGroundTruthSelected: (String?) -> Unit,
    onSelectAnotherImage: () -> Unit,
    onRunInference: () -> Unit,
    onSaveResult: () -> Unit,
    onHome: () -> Unit,
    onHistory: () -> Unit,
    showBackButton: Boolean = false,
    onBack: (() -> Unit)? = null,
    enableGroundTruthSelection: Boolean = false,
    showResultMetrics: Boolean = false,
    historyActionText: String = "Lịch sử",
    modifier: Modifier = Modifier,
) {
    val result = classificationState.result
    AnimatedContent(
        targetState = result,
        modifier = modifier.fillMaxSize(),
        transitionSpec = {
            val showingResult = targetState != null
            val enterOffset: (Int) -> Int = { width ->
                if (showingResult) width / 4 else -width / 4
            }
            val exitOffset: (Int) -> Int = { width ->
                if (showingResult) -width / 6 else width / 6
            }

            (
                fadeIn(animationSpec = tween(durationMillis = 220)) +
                    slideInHorizontally(
                        animationSpec = tween(durationMillis = 260),
                        initialOffsetX = enterOffset,
                    )
                ).togetherWith(
                    fadeOut(animationSpec = tween(durationMillis = 160)) +
                        slideOutHorizontally(
                            animationSpec = tween(durationMillis = 220),
                            targetOffsetX = exitOffset,
                        )
                )
        },
        label = "inference-content",
    ) { animatedResult ->
        if (animatedResult == null) {
            CaptureUploadContent(
                inputState = inputState,
                classificationState = classificationState,
                labels = labels,
                selectedGroundTruthLabel = selectedGroundTruthLabel,
                cameraPermissionGranted = cameraPermissionGranted,
                onRequestCameraPermission = onRequestCameraPermission,
                onOpenCamera = onOpenCamera,
                onSnapshot = onSnapshot,
                onFrameObserved = onFrameObserved,
                onCameraError = onCameraError,
                onGroundTruthSelected = onGroundTruthSelected,
                onSelectAnotherImage = onSelectAnotherImage,
                onRunInference = onRunInference,
                showBackButton = showBackButton,
                onBack = onBack,
                enableGroundTruthSelection = enableGroundTruthSelection,
            )
        } else {
            DiagnosisResultContent(
                inputState = inputState,
                result = animatedResult,
                modelInfo = modelInfo,
                benchmarkMetrics = benchmarkMetrics,
                runtimeDelegateName = runtimeDelegateName,
                debugInfoEnabled = debugInfoEnabled,
                cameraFps = cameraFps,
                onSaveResult = onSaveResult,
                onScanAnother = onOpenCamera,
                onSelectAnotherImage = onSelectAnotherImage,
                onHome = onHome,
                onHistory = onHistory,
                historyActionText = historyActionText,
                showBackButton = showBackButton,
                onBack = onBack,
                showResultMetrics = showResultMetrics,
            )
        }
    }
}

private fun ClassificationResult.localizedPredictionText(): String {
    return if (isAboveThreshold) {
        DiseaseTextUtils.displayLabel(predictedClass)
    } else {
        "${DiseaseTextUtils.displayLabel(rawTop1Label)}"
    }
}

private fun ClassificationResult.localizedStatusKind(): ResultStatusKind {
    return when {
        !isAboveThreshold -> ResultStatusKind.Warning
        DiseaseTextUtils.isHealthy(predictedClass) -> ResultStatusKind.Healthy
        DiseaseTextUtils.isUnclear(predictedClass) -> ResultStatusKind.Warning
        else -> ResultStatusKind.Disease
    }
}

private fun ClassificationResult.localizedStatusColor() = when (localizedStatusKind()) {
    ResultStatusKind.Healthy -> HealthyGreen
    ResultStatusKind.Disease -> DiseaseRed
    ResultStatusKind.Warning -> WarningOrange
    ResultStatusKind.Neutral -> DarkNavy
}

private fun ClassificationResult.localizedStatusText(): String {
    return when (localizedStatusKind()) {
        ResultStatusKind.Healthy -> "Tình trạng: Khỏe"
        ResultStatusKind.Disease -> "Phát hiện dấu hiệu bệnh"
        ResultStatusKind.Warning -> "Ảnh chưa đủ rõ"
        ResultStatusKind.Neutral -> "Chưa có"
    }
}

private fun localizedInspectionReport(result: ClassificationResult): String {
    val prediction = result.localizedPredictionText()
    val confidence = BenchmarkUtils.confidenceText(result.confidence)
    return when (result.localizedStatusKind()) {
        ResultStatusKind.Healthy -> "Kết quả nghiêng về $prediction với độ tin cậy $confidence. Chưa thấy dấu hiệu bệnh rõ trong ảnh, tiếp tục theo dõi ao định kỳ."
        ResultStatusKind.Disease -> "Kết quả phát hiện $prediction với độ tin cậy $confidence. Nên kiểm tra thêm vài mẫu tôm cùng ao và đối chiếu vùng bệnh trên ảnh trước khi xử lý."
        ResultStatusKind.Warning -> "Kết quả chưa đủ chắc chắn ($prediction, $confidence). Hãy chụp lại ảnh rõ hơn, đủ sáng và lấy trọn vùng đầu, mang, vỏ tôm."
        ResultStatusKind.Neutral -> "Chưa có báo cáo kiểm tra khả dụng cho mẫu này."
    }
}

private fun localizedDiseaseDescription(result: ClassificationResult): String {
    val prediction = result.localizedPredictionText()
    return when (result.localizedStatusKind()) {
        ResultStatusKind.Healthy -> DiseaseTextUtils.diseaseDescription(result.predictedClass)
            ?: "Mẫu hiện chưa cho thấy dấu hiệu bệnh rõ trong nhóm nhãn mà mô hình đang hỗ trợ."
        ResultStatusKind.Disease -> DiseaseTextUtils.diseaseDescription(result.predictedClass)
            ?: "$prediction là nhóm bệnh cần được kiểm tra thêm trên mẫu thực tế. Quan sát kỹ vùng mang, vỏ và mức độ bất thường của tôm để xác nhận."
        ResultStatusKind.Warning -> "Ảnh hoặc độ tin cậy chưa đủ để mô tả bệnh đáng tin cậy. Cần ảnh rõ vùng đầu, mang và vỏ để phân biệt dấu hiệu bệnh."
        ResultStatusKind.Neutral -> "Chưa có mô tả bệnh khả dụng cho kết quả này."
    }
}

private fun ClassificationResult.shouldShowDiseaseMaskCard(): Boolean {
    return isAboveThreshold && localizedStatusKind() == ResultStatusKind.Disease
}

@Composable
private fun CaptureUploadContent(
    inputState: InferenceInputUiState,
    classificationState: ClassificationUiState,
    labels: List<String>,
    selectedGroundTruthLabel: String?,
    cameraPermissionGranted: Boolean,
    onRequestCameraPermission: () -> Unit,
    onOpenCamera: () -> Unit,
    onSnapshot: (Bitmap) -> Unit,
    onFrameObserved: () -> Unit,
    onCameraError: (String) -> Unit,
    onGroundTruthSelected: (String?) -> Unit,
    onSelectAnotherImage: () -> Unit,
    onRunInference: () -> Unit,
    showBackButton: Boolean,
    onBack: (() -> Unit)?,
    enableGroundTruthSelection: Boolean,
    modifier: Modifier = Modifier,
) {
    var retakeRequested by remember { mutableStateOf(false) }
    val hasImage = inputState.bitmap != null || inputState.imageUri != null
    val showCamera = retakeRequested || (inputState.isCameraActive && inputState.bitmap == null)

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        InlineBackButton(
            visible = showBackButton,
            onBack = onBack,
        )

        CapturePreview(
            inputState = inputState,
            showCamera = showCamera,
            cameraPermissionGranted = cameraPermissionGranted,
            onRequestCameraPermission = onRequestCameraPermission,
            onSnapshot = { bitmap ->
                retakeRequested = false
                onSnapshot(bitmap)
            },
            onFrameObserved = onFrameObserved,
            onCameraError = onCameraError,
        )

        if (!hasImage && !showCamera) {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                PrimaryActionButton(
                    text = "Mở máy ảnh",
                    onClick = onOpenCamera,
                    modifier = Modifier.fillMaxWidth(),
                )
                SecondaryActionButton(
                    text = "Chọn ảnh có sẵn",
                    onClick = onSelectAnotherImage,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        if (hasImage) {
            if (enableGroundTruthSelection) {
                GroundTruthSelectorCard(
                    labels = labels,
                    selectedGroundTruthLabel = selectedGroundTruthLabel,
                    onGroundTruthSelected = onGroundTruthSelected,
                )
            }

            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    SecondaryActionButton(
                        text = "Chụp lại",
                        onClick = {
                            retakeRequested = true
                            onOpenCamera()
                        },
                        modifier = Modifier.weight(1f),
                    )
                    SecondaryActionButton(
                        text = "Chọn ảnh khác",
                        onClick = {
                            retakeRequested = false
                            onSelectAnotherImage()
                        },
                        modifier = Modifier.weight(1f),
                    )
                }
                PrimaryActionButton(
                    text = if (classificationState.isLoading) "Đang phân tích..." else "Dùng ảnh này",
                    enabled = !classificationState.isLoading,
                    onClick = onRunInference,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        if (classificationState.errorMessage != null) {
            CVioCard(containerColor = MaterialTheme.colorScheme.errorContainer) {
                Text(
                    text = "Phân tích chưa thành công",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = classificationState.errorMessage,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                )
            }
        }

        TipsCard()
    }
}

@Composable
private fun CapturePreview(
    inputState: InferenceInputUiState,
    showCamera: Boolean,
    cameraPermissionGranted: Boolean,
    onRequestCameraPermission: () -> Unit,
    onSnapshot: (Bitmap) -> Unit,
    onFrameObserved: () -> Unit,
    onCameraError: (String) -> Unit,
) {
    if (showCamera) {
        if (!cameraPermissionGranted) {
            PermissionCard(
                title = "Cần quyền máy ảnh",
                message = "Cho phép dùng máy ảnh để chụp ảnh tôm.",
                actionLabel = "Cho phép máy ảnh",
                onAction = onRequestCameraPermission,
            )
        } else {
            CameraCaptureCard(
                enabled = true,
                capturedBitmap = null,
                onSnapshot = onSnapshot,
                onRetake = {},
                onFrameObserved = onFrameObserved,
                onError = onCameraError,
            )
        }
        return
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(3f / 4f),
        shape = MaterialTheme.shapes.large,
        colors = CardDefaults.cardColors(containerColor = DarkNavy),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            when {
                inputState.bitmap != null -> {
                    Image(
                        bitmap = inputState.bitmap.asImageBitmap(),
                        contentDescription = "Ảnh tôm đã chọn",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                inputState.imageUri != null -> {
                    AsyncImage(
                        model = inputState.imageUri,
                        contentDescription = "Ảnh tôm đã chọn",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                else -> {
                    CameraScanFrame(
                        modifier = Modifier.fillMaxSize(),
                        label = "Đặt tôm vào giữa khung để AI dễ kiểm tra",
                    )
                }
            }

            if (inputState.bitmap != null || inputState.imageUri != null) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(18.dp)
                        .border(
                            width = 2.dp,
                            color = CVioPrimaryFixed.copy(alpha = 0.72f),
                            shape = RoundedCornerShape(24.dp),
                        ),
                )
            }
        }
    }
}

@Composable
private fun GroundTruthSelectorCard(
    labels: List<String>,
    selectedGroundTruthLabel: String?,
    onGroundTruthSelected: (String?) -> Unit,
) {
    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 2.dp) {
        CVioSectionHeader(
            title = "Nhãn thực tế",
            subtitle = "Admin có thể chọn nhãn đúng sau khi chụp hoặc chọn ảnh để tính độ chính xác.",
        )
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            FilterChip(
                selected = selectedGroundTruthLabel == null,
                onClick = { onGroundTruthSelected(null) },
                label = { Text("Không chọn") },
            )
            labels.forEach { label ->
                FilterChip(
                    selected = selectedGroundTruthLabel == label,
                    onClick = { onGroundTruthSelected(label) },
                    label = {
                        Text(
                            text = DiseaseTextUtils.displayLabel(label),
                            maxLines = 1,
                        )
                    },
                )
            }
        }
    }
}

@Composable
private fun TipsCard() {
    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 3.dp) {
        Text(
            text = "Mẹo để AI kiểm tra tốt hơn",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.Bold,
        )
        TipRow("Đặt tôm rõ ở giữa khung hình")
        TipRow("Chụp nơi đủ sáng, tránh bóng đổ mạnh")
        TipRow("Giữ điện thoại chắc tay để ảnh không bị mờ")
    }
}

@Composable
private fun TipRow(text: String) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = "*",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun DiagnosisResultContent(
    inputState: InferenceInputUiState,
    result: ClassificationResult,
    modelInfo: ModelInfo,
    benchmarkMetrics: BenchmarkMetrics,
    runtimeDelegateName: String,
    debugInfoEnabled: Boolean,
    cameraFps: Double?,
    onSaveResult: () -> Unit,
    onScanAnother: () -> Unit,
    onSelectAnotherImage: () -> Unit,
    onHome: () -> Unit,
    onHistory: () -> Unit,
    historyActionText: String,
    showBackButton: Boolean,
    onBack: (() -> Unit)?,
    showResultMetrics: Boolean,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        InlineBackButton(
            visible = showBackButton,
            onBack = onBack,
        )

        ResultOverviewCard(inputState = inputState, result = result)

        ResultActions(
            onSaveResult = onSaveResult,
            onScanAnother = onScanAnother,
            onSelectAnotherImage = onSelectAnotherImage,
            onHome = onHome,
            onHistory = onHistory,
            historyActionText = historyActionText,
        )

        if (showResultMetrics) {
            MetricsCard(
                result = result,
                benchmarkMetrics = benchmarkMetrics,
            )
        }

        DiagnosisDetailsCard(
            result = result,
        )

        if (result.shouldShowDiseaseMaskCard()) {
            AttentionMapCard(inputState = inputState, result = result)
        }
        TopPredictionList(predictions = result.top3Predictions)

        if (debugInfoEnabled) {
            DebugInfoCard(
                inputState = inputState,
                result = result,
                modelInfo = modelInfo,
                runtimeDelegateName = runtimeDelegateName,
                cameraFps = cameraFps,
            )
        }
    }
}

@Composable
private fun InlineBackButton(
    visible: Boolean,
    onBack: (() -> Unit)?,
) {
    if (!visible || onBack == null) return

    ShrimpIconButton(
        icon = ShrimpNavIcon.Back,
        contentDescription = "Quay lại",
        onClick = onBack,
        modifier = Modifier
            .size(44.dp)
            .clip(RoundedCornerShape(999.dp))
            .background(CVioSurfaceContainerLowest),
    )
}

@Composable
private fun ResultOverviewCard(
    inputState: InferenceInputUiState,
    result: ClassificationResult,
) {
    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 4.dp) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(260.dp)
                .clip(RoundedCornerShape(28.dp))
                .background(CVioSurfaceContainerLow),
        ) {
            when {
                inputState.bitmap != null -> {
                    Image(
                        bitmap = inputState.bitmap.asImageBitmap(),
                        contentDescription = "Ảnh tôm đã kiểm tra",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                inputState.imageUri != null -> {
                    AsyncImage(
                        model = inputState.imageUri,
                        contentDescription = "Ảnh tôm đã kiểm tra",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                else -> {
                    EmptyState(
                        title = "Không hiển thị được ảnh",
                        message = "Kết quả đã có nhưng không hiển thị được ảnh xem trước.",
                    )
                }
            }
            if (result.segmentation != null) {
                ResultStatusBadge(
                    text = "Vùng bệnh cần chú ý",
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(14.dp),
                    kind = result.localizedStatusKind(),
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Top,
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                ResultStatusBadge(
                    text = result.localizedStatusText(),
                    kind = result.localizedStatusKind(),
                )
                Text(
                    text = result.localizedPredictionText(),
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                )
            }
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = BenchmarkUtils.confidenceText(result.confidence),
                    style = MaterialTheme.typography.displayLarge,
                    color = result.localizedStatusColor(),
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Độ tin cậy",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        LinearProgressIndicator(
            progress = { result.confidence.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(),
            color = result.localizedStatusColor(),
            trackColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.42f),
        )

        CVioCard(containerColor = CVioSurfaceContainerLow, tonalElevation = 0.dp) {
            Text(
                text = "Báo cáo kiểm tra",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = localizedInspectionReport(result),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun ResultActions(
    onSaveResult: () -> Unit,
    onScanAnother: () -> Unit,
    onSelectAnotherImage: () -> Unit,
    onHome: () -> Unit,
    onHistory: () -> Unit,
    historyActionText: String,
) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            PrimaryActionButton(
                text = "Lưu vào lịch sử",
                onClick = onSaveResult,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = "Chụp ảnh khác",
                onClick = onScanAnother,
                modifier = Modifier.weight(1f),
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SecondaryActionButton(
                text = "Chọn ảnh khác",
                onClick = onSelectAnotherImage,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = historyActionText,
                onClick = onHistory,
                modifier = Modifier.weight(1f),
            )
        }
        SecondaryActionButton(
            text = "Trang chủ",
            onClick = onHome,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun DiagnosisDetailsCard(
    result: ClassificationResult,
) {
    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 3.dp) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                Text(
                    text = "Phân tích bệnh dự đoán",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontWeight = FontWeight.Bold,
                )
            }
            ResultStatusBadge(
                text = when (result.localizedStatusKind()) {
                    ResultStatusKind.Healthy -> "Nguy cơ thấp"
                    ResultStatusKind.Disease -> "Nguy cơ cao"
                    ResultStatusKind.Warning -> "Cần xem lại"
                    ResultStatusKind.Neutral -> "Chưa có"
                },
                kind = result.localizedStatusKind(),
            )
        }

        Text(
            text = "Kết quả chính",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = result.localizedPredictionText(),
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = localizedDiseaseDescription(result),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun AttentionMapCard(
    inputState: InferenceInputUiState,
    result: ClassificationResult,
) {
    val segmentation = result.segmentation
    val hasImage = inputState.bitmap != null || inputState.imageUri != null

    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 3.dp) {
        CVioSectionHeader(
            title = if (segmentation != null) "Mask vùng bệnh" else "Chưa tạo được mask vùng bệnh",
            subtitle = when {
                !hasImage -> "Chưa có ảnh để tạo mask."
                segmentation != null -> "YOLOv11n-seg khoanh vùng tổn thương sau khi classifier phát hiện bệnh."
                else -> "Classifier đã phát hiện bệnh, nhưng YOLOv11n-seg chưa tìm thấy vùng tổn thương đủ rõ trên ảnh này."
            },
        )
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1.65f)
                .clip(RoundedCornerShape(24.dp))
                .background(DarkNavy),
        ) {
            when {
                inputState.bitmap != null -> {
                    Image(
                        bitmap = inputState.bitmap.asImageBitmap(),
                        contentDescription = "Ảnh gốc của mask vùng bệnh",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                inputState.imageUri != null -> {
                    AsyncImage(
                        model = inputState.imageUri,
                        contentDescription = "Ảnh gốc của mask vùng bệnh",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                else -> {
                    CameraScanFrame(
                        modifier = Modifier.fillMaxSize(),
                        label = "Mask vùng bệnh",
                    )
                }
            }

            if (hasImage) {
                if (segmentation != null) {
                    Image(
                        bitmap = segmentation.maskBitmap.asImageBitmap(),
                        contentDescription = "Disease segmentation mask",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )

                    if (segmentation.classCounts.isNotEmpty()) {
                        Row(
                            modifier = Modifier
                                .align(Alignment.TopStart)
                                .padding(12.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            segmentation.classCounts.toSortedMap().forEach { (label, count) ->
                                MaskLegendBadge(label = label, count = count)
                            }
                        }
                    }

                    ResultStatusBadge(
                        text = "Mask ${segmentation.label} ${segmentation.detectionCount} vùng",
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .padding(12.dp),
                        kind = ResultStatusKind.Disease,
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.38f)),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = "Chưa có vùng mask đủ rõ",
                            modifier = Modifier
                                .clip(RoundedCornerShape(999.dp))
                                .background(Color.Black.copy(alpha = 0.56f))
                                .padding(horizontal = 14.dp, vertical = 8.dp),
                            style = MaterialTheme.typography.labelMedium,
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun MaskLegendBadge(
    label: String,
    count: Int,
) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(Color.Black.copy(alpha = 0.48f))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(9.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(maskLegendColor(label)),
        )
        Text(
            text = "$label $count",
            style = MaterialTheme.typography.labelSmall,
            color = Color.White,
            fontWeight = FontWeight.Bold,
        )
    }
}

private fun maskLegendColor(label: String): Color {
    return if ("wssv" in label.lowercase()) {
        Color(red = 124, green = 77, blue = 255)
    } else {
        Color(red = 0, green = 200, blue = 83)
    }
}

@Composable
private fun DebugInfoCard(
    inputState: InferenceInputUiState,
    result: ClassificationResult,
    modelInfo: ModelInfo,
    runtimeDelegateName: String,
    cameraFps: Double?,
) {
    CVioCard(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.75f)) {
        CVioSectionHeader(
            title = "Thông tin kỹ thuật",
            subtitle = "Chỉ dùng khi cần kiểm tra kỹ thuật.",
        )
        CompactInfoRow("Nguồn ảnh", inputState.source?.displayName ?: "Không có")
        CompactInfoRow("Đường dẫn ảnh", inputState.imageUri?.toString() ?: "Ảnh chụp máy ảnh / Không có")
        CompactInfoRow("Mô hình", modelInfo.modelFile)
        CompactInfoRow("Cách chạy", runtimeDelegateName)
        CompactInfoRow("Cỡ ảnh vào", BenchmarkUtils.inputSizeText(modelInfo))
        CompactInfoRow("Dạng dữ liệu vào", BenchmarkUtils.shapeText(modelInfo.input.shape))
        CompactInfoRow("Dạng kết quả", BenchmarkUtils.shapeText(modelInfo.output.shape))
        CompactInfoRow("Ngưỡng tin cậy", BenchmarkUtils.confidenceText(result.threshold))
        CompactInfoRow(
            "FPS máy ảnh",
            cameraFps?.let { String.format(Locale.US, "%.1f", it) } ?: "Không có",
        )
        CompactInfoRow("Chuẩn bị ảnh", BenchmarkUtils.latencyText(result.preprocessingTimeMs))
        CompactInfoRow("AI xử lý", BenchmarkUtils.latencyText(result.modelInferenceTimeMs))
        CompactInfoRow("Xử lý kết quả", BenchmarkUtils.latencyText(result.postprocessingTimeMs))
        CompactInfoRow("Tổng thời gian", BenchmarkUtils.latencyText(result.totalTimeMs))
    }
}

