package rs.smobile.shrimpdisease.ui.inference

import android.graphics.Bitmap
import android.graphics.Color as AndroidColor
import androidx.compose.foundation.Canvas
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import rs.smobile.shrimpdisease.CameraCaptureCard
import rs.smobile.shrimpdisease.ClassificationUiState
import rs.smobile.shrimpdisease.InferenceInputUiState
import rs.smobile.shrimpdisease.InferenceSource
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
    modifier: Modifier = Modifier,
) {
    val result = classificationState.result
    if (result == null) {
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
            modifier = modifier,
        )
    } else {
        DiagnosisResultContent(
            inputState = inputState,
            result = result,
            modelInfo = modelInfo,
            benchmarkMetrics = benchmarkMetrics,
            runtimeDelegateName = runtimeDelegateName,
            debugInfoEnabled = debugInfoEnabled,
            cameraFps = cameraFps,
            onSaveResult = onSaveResult,
            onScanAnother = onOpenCamera,
            onHome = onHome,
            onHistory = onHistory,
            showBackButton = showBackButton,
            onBack = onBack,
            modifier = modifier,
        )
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

private fun localizedDiseaseAnalysis(result: ClassificationResult): String {
    val prediction = result.localizedPredictionText()
    val confidence = BenchmarkUtils.confidenceText(result.confidence)
    return when (result.localizedStatusKind()) {
        ResultStatusKind.Healthy -> "AI dự đoán mẫu là $prediction với độ tin cậy $confidence. Ảnh hiện tại không cho thấy dấu hiệu bệnh rõ ràng trong nhóm nhãn mà mô hình đang hỗ trợ."
        ResultStatusKind.Disease -> {
            val diseaseInfo = DiseaseTextUtils.diseaseInfo(result.predictedClass)
                ?: "Bà con nên xem kết quả này là cảnh báo sớm, kiểm tra thêm tôm trong ao và theo dõi môi trường nước trước khi xử lý."
            "AI dự đoán mẫu có dấu hiệu $prediction với độ tin cậy $confidence. $diseaseInfo"
        }
        ResultStatusKind.Warning -> "AI dự đoán $prediction nhưng ảnh hoặc độ tin cậy chưa đủ chắc chắn. Nên chụp lại ảnh rõ hơn, đủ sáng và đặt tôm ở giữa khung trước khi dùng kết quả để đánh giá ao."
        ResultStatusKind.Neutral -> "Chưa có phân tích bệnh khả dụng cho kết quả này."
    }
}

private data class HeatmapHotspot(
    val xRatio: Float,
    val yRatio: Float,
    val strength: Float,
)

private fun buildImageHeatmapHotspots(
    bitmap: Bitmap?,
    result: ClassificationResult,
): List<HeatmapHotspot> {
    if (bitmap == null || bitmap.width <= 0 || bitmap.height <= 0) {
        return fallbackHotspots(result.localizedStatusKind())
    }

    val statusKind = result.localizedStatusKind()
    val columns = 12
    val rows = 12
    val samples = mutableListOf<HeatmapSample>()
    var totalBrightness = 0f

    for (row in 0 until rows) {
        for (column in 0 until columns) {
            val x = ((column + 0.5f) * bitmap.width / columns)
                .toInt()
                .coerceIn(0, bitmap.width - 1)
            val y = ((row + 0.5f) * bitmap.height / rows)
                .toInt()
                .coerceIn(0, bitmap.height - 1)
            val pixel = bitmap.getPixel(x, y)
            val red = AndroidColor.red(pixel) / 255f
            val green = AndroidColor.green(pixel) / 255f
            val blue = AndroidColor.blue(pixel) / 255f
            val brightness = (red + green + blue) / 3f
            val saturation = maxOf(red, green, blue) - minOf(red, green, blue)
            val xRatio = (x + 0.5f) / bitmap.width
            val yRatio = (y + 0.5f) / bitmap.height
            val centrality = centralityScore(xRatio, yRatio)
            samples += HeatmapSample(
                xRatio = xRatio,
                yRatio = yRatio,
                brightness = brightness,
                saturation = saturation,
                centrality = centrality,
            )
            totalBrightness += brightness
        }
    }

    val averageBrightness = totalBrightness / samples.size.coerceAtLeast(1)
    val scored = samples.map { sample ->
        val contrast = kotlin.math.abs(sample.brightness - averageBrightness)
        val whiteSpotCue = sample.brightness * (1f - sample.saturation * 0.55f)
        val score = when (statusKind) {
            ResultStatusKind.Disease -> {
                whiteSpotCue * 0.42f + contrast * 0.38f + sample.centrality * 0.2f
            }

            ResultStatusKind.Healthy -> {
                sample.centrality * 0.46f +
                    (1f - contrast).coerceIn(0f, 1f) * 0.28f +
                    sample.saturation * 0.26f
            }

            ResultStatusKind.Warning, ResultStatusKind.Neutral -> {
                sample.centrality * 0.5f + contrast * 0.3f + sample.brightness * 0.2f
            }
        }
        sample to score
    }.sortedByDescending { it.second }

    val selected = mutableListOf<Pair<HeatmapSample, Float>>()
    for (candidate in scored) {
        val farEnough = selected.all { selectedSample ->
            val dx = selectedSample.first.xRatio - candidate.first.xRatio
            val dy = selectedSample.first.yRatio - candidate.first.yRatio
            dx * dx + dy * dy > 0.0225f
        }
        if (farEnough) selected += candidate
        if (selected.size == 5) break
    }

    val maxScore = selected.maxOfOrNull { it.second }?.takeIf { it > 0f } ?: 1f
    return selected.ifEmpty { fallbackHotspots(statusKind).map { hotspot -> HeatmapSample(hotspot.xRatio, hotspot.yRatio, 0f, 0f, 0f) to hotspot.strength } }
        .map { (sample, score) ->
            HeatmapHotspot(
                xRatio = sample.xRatio,
                yRatio = sample.yRatio,
                strength = (score / maxScore).coerceIn(0.35f, 1f),
            )
        }
}

private data class HeatmapSample(
    val xRatio: Float,
    val yRatio: Float,
    val brightness: Float,
    val saturation: Float,
    val centrality: Float,
)

private fun centralityScore(
    xRatio: Float,
    yRatio: Float,
): Float {
    val dx = kotlin.math.abs(xRatio - 0.5f) / 0.5f
    val dy = kotlin.math.abs(yRatio - 0.5f) / 0.5f
    return (1f - (dx * dx + dy * dy) / 2f).coerceIn(0f, 1f)
}

private fun HeatmapHotspot.toCanvasOffset(
    canvasWidth: Float,
    canvasHeight: Float,
    bitmap: Bitmap?,
): Offset {
    if (bitmap == null || bitmap.width <= 0 || bitmap.height <= 0) {
        return Offset(canvasWidth * xRatio, canvasHeight * yRatio)
    }

    val scale = maxOf(
        canvasWidth / bitmap.width.toFloat(),
        canvasHeight / bitmap.height.toFloat(),
    )
    val displayedWidth = bitmap.width * scale
    val displayedHeight = bitmap.height * scale
    val offsetX = (canvasWidth - displayedWidth) / 2f
    val offsetY = (canvasHeight - displayedHeight) / 2f

    return Offset(
        x = offsetX + xRatio * displayedWidth,
        y = offsetY + yRatio * displayedHeight,
    )
}

private fun fallbackHotspots(statusKind: ResultStatusKind): List<HeatmapHotspot> {
    return when (statusKind) {
        ResultStatusKind.Healthy -> listOf(
            HeatmapHotspot(0.52f, 0.52f, 0.92f),
            HeatmapHotspot(0.34f, 0.38f, 0.62f),
        )

        ResultStatusKind.Disease -> listOf(
            HeatmapHotspot(0.46f, 0.46f, 1f),
            HeatmapHotspot(0.66f, 0.62f, 0.72f),
            HeatmapHotspot(0.32f, 0.58f, 0.58f),
        )

        ResultStatusKind.Warning -> listOf(
            HeatmapHotspot(0.5f, 0.5f, 0.86f),
            HeatmapHotspot(0.68f, 0.38f, 0.56f),
        )

        ResultStatusKind.Neutral -> listOf(HeatmapHotspot(0.5f, 0.5f, 0.75f))
    }
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

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                SecondaryActionButton(
                    text = if (inputState.source == InferenceSource.SNAPSHOT) "Chụp lại" else "Chọn ảnh khác",
                    onClick = {
                        if (inputState.source == InferenceSource.SNAPSHOT) {
                            retakeRequested = true
                            onOpenCamera()
                        } else {
                            onSelectAnotherImage()
                        }
                    },
                    modifier = Modifier.weight(1f),
                )
                PrimaryActionButton(
                    text = if (classificationState.isLoading) "Đang phân tích..." else "Dùng ảnh này",
                    enabled = !classificationState.isLoading,
                    onClick = onRunInference,
                    modifier = Modifier.weight(1.65f),
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
    onHome: () -> Unit,
    onHistory: () -> Unit,
    showBackButton: Boolean,
    onBack: (() -> Unit)?,
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
            onHome = onHome,
            onHistory = onHistory,
        )

        MetricsCard(
            result = result,
            benchmarkMetrics = benchmarkMetrics,
        )

        DiagnosisDetailsCard(
            result = result,
        )

        AttentionMapCard(inputState = inputState, result = result)
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
            ResultStatusBadge(
                text = "Xem vùng AI chú ý",
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(14.dp),
                kind = ResultStatusKind.Neutral,
            )
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
                    text = result.statusText(),
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
                text = "Phân tích bệnh dự đoán",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = localizedDiseaseAnalysis(result),
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
    onHome: () -> Unit,
    onHistory: () -> Unit,
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
                text = "Kiểm tra ảnh khác",
                onClick = onScanAnother,
                modifier = Modifier.weight(1f),
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SecondaryActionButton(
                text = "Trang chủ",
                onClick = onHome,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = "Lịch sử",
                onClick = onHistory,
                modifier = Modifier.weight(1f),
            )
        }
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
                    text = "Báo cáo kiểm tra",
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
            text = localizedDiseaseAnalysis(result),
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
    val hasImage = inputState.bitmap != null || inputState.imageUri != null
    val statusKind = result.localizedStatusKind()
    val heatmapHotspots = remember(inputState.bitmap, result.predictedClass, result.confidence) {
        buildImageHeatmapHotspots(inputState.bitmap, result)
    }
    val hotspotColor = when (statusKind) {
        ResultStatusKind.Healthy -> HealthyGreen
        ResultStatusKind.Disease -> DiseaseRed
        ResultStatusKind.Warning -> WarningOrange
        ResultStatusKind.Neutral -> MaterialTheme.colorScheme.primary
    }

    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 3.dp) {
        CVioSectionHeader(
            title = "Vùng AI chú ý",
            subtitle = if (hasImage) {
                "Đã tạo bản đồ nhiệt từ ảnh hiện tại và độ tin cậy của AI."
            } else {
                "Chưa có ảnh để tạo bản đồ nhiệt."
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
                        contentDescription = "Ảnh gốc của bản đồ nhiệt",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                inputState.imageUri != null -> {
                    AsyncImage(
                        model = inputState.imageUri,
                        contentDescription = "Ảnh gốc của bản đồ nhiệt",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                else -> {
                    CameraScanFrame(
                        modifier = Modifier.fillMaxSize(),
                        label = "Bản đồ vùng AI chú ý",
                    )
                }
            }

            if (hasImage) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawRect(color = DarkNavy.copy(alpha = 0.12f))

                    heatmapHotspots.forEachIndexed { index, hotspot ->
                        val center = hotspot.toCanvasOffset(
                            canvasWidth = size.width,
                            canvasHeight = size.height,
                            bitmap = inputState.bitmap,
                        )
                        val radius = size.minDimension * (0.18f + hotspot.strength * 0.22f)
                        val alpha = (0.2f + hotspot.strength * 0.48f)
                            .coerceIn(0.18f, 0.68f) * result.confidence.coerceIn(0.45f, 1f)
                        val innerAlpha = if (index == 0) alpha else alpha * 0.82f
                        drawCircle(
                            brush = Brush.radialGradient(
                                colors = listOf(
                                    hotspotColor.copy(alpha = innerAlpha),
                                    hotspotColor.copy(alpha = innerAlpha * 0.42f),
                                    Color.Transparent,
                                ),
                                center = center,
                                radius = radius,
                            ),
                            radius = radius,
                            center = center,
                        )
                    }
                }

                ResultStatusBadge(
                    text = "Đã tải bản đồ nhiệt",
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(12.dp),
                    kind = statusKind,
                )
            }
        }
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

private fun ClassificationResult.displayPredictionText(): String {
    return if (isAboveThreshold) {
        predictedClass
    } else {
        "$rawTop1Label"
    }
}

private fun ClassificationResult.statusKind(): ResultStatusKind {
    val label = displayPredictionText().lowercase()
    return when {
        !isAboveThreshold -> ResultStatusKind.Warning
        "healthy" in label -> ResultStatusKind.Healthy
        else -> ResultStatusKind.Disease
    }
}

private fun ClassificationResult.statusText(): String {
    return when (statusKind()) {
        ResultStatusKind.Healthy -> "Tình trạng: Khỏe"
        ResultStatusKind.Disease -> "Phát hiện dấu hiệu bệnh"
        ResultStatusKind.Warning -> "Ảnh chưa đủ rõ"
        ResultStatusKind.Neutral -> "Chưa có"
    }
}

private fun ClassificationResult.statusColor() = when (statusKind()) {
    ResultStatusKind.Healthy -> HealthyGreen
    ResultStatusKind.Disease -> DiseaseRed
    ResultStatusKind.Warning -> WarningOrange
    ResultStatusKind.Neutral -> DarkNavy
}

private fun diseaseAnalysis(result: ClassificationResult): String {
    val prediction = result.displayPredictionText()
    val confidence = BenchmarkUtils.confidenceText(result.confidence)
    return when (result.statusKind()) {
        ResultStatusKind.Healthy -> "AI dự đoán mẫu là $prediction với độ tin cậy $confidence. Ảnh hiện tại không cho thấy dấu hiệu bệnh rõ ràng trong nhóm nhãn mà mô hình đang hỗ trợ."
        ResultStatusKind.Disease -> "AI dự đoán mẫu có dấu hiệu $prediction với độ tin cậy $confidence. Kết quả này phản ánh các vùng ảnh có đặc trưng bất thường mà mô hình liên hệ với bệnh được dự đoán."
        ResultStatusKind.Warning -> "AI dự đoán $prediction nhưng độ tin cậy chỉ đạt $confidence, thấp hơn ngưỡng cấu hình. Nên chụp lại ảnh rõ hơn trước khi dùng kết quả để đánh giá tình trạng ao."
        ResultStatusKind.Neutral -> "Chưa có phân tích bệnh khả dụng cho kết quả này."
    }
}
