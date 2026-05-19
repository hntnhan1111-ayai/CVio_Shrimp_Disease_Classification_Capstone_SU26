package rs.smobile.shrimpdisease.ui.inference

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
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
import rs.smobile.shrimpdisease.ui.components.CVioMetricTile
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CompactInfoRow
import rs.smobile.shrimpdisease.ui.components.EmptyState
import rs.smobile.shrimpdisease.ui.components.MetricsCard
import rs.smobile.shrimpdisease.ui.components.PermissionCard
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.components.ResultStatusBadge
import rs.smobile.shrimpdisease.ui.components.ResultStatusKind
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.components.TopPredictionList
import rs.smobile.shrimpdisease.ui.theme.CVioPrimaryFixed
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
import rs.smobile.shrimpdisease.ui.theme.DarkNavy
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.ui.theme.WarningOrange
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
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
            modifier = modifier,
        )
    } else {
        DiagnosisResultContent(
            inputState = inputState,
            result = result,
            labels = labels,
            selectedGroundTruthLabel = selectedGroundTruthLabel,
            modelInfo = modelInfo,
            benchmarkMetrics = benchmarkMetrics,
            runtimeDelegateName = runtimeDelegateName,
            debugInfoEnabled = debugInfoEnabled,
            cameraFps = cameraFps,
            onGroundTruthSelected = onGroundTruthSelected,
            onSaveResult = onSaveResult,
            onScanAnother = onOpenCamera,
            onHome = onHome,
            onHistory = onHistory,
            modifier = modifier,
        )
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
                    text = "Open Camera",
                    onClick = onOpenCamera,
                    modifier = Modifier.fillMaxWidth(),
                )
                SecondaryActionButton(
                    text = "Select from Gallery",
                    onClick = onSelectAnotherImage,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        if (hasImage) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                SecondaryActionButton(
                    text = if (inputState.source == InferenceSource.SNAPSHOT) "Retake" else "Choose Again",
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
                    text = if (classificationState.isLoading) "Analyzing..." else "Use This Image",
                    enabled = !classificationState.isLoading,
                    onClick = onRunInference,
                    modifier = Modifier.weight(1.65f),
                )
            }
        }

        if (classificationState.errorMessage != null) {
            CVioCard(containerColor = MaterialTheme.colorScheme.errorContainer) {
                Text(
                    text = "Analysis failed",
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

        GroundTruthCard(
            labels = labels,
            selectedGroundTruthLabel = selectedGroundTruthLabel,
            resultCorrectness = null,
            onGroundTruthSelected = onGroundTruthSelected,
        )
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
                title = "Camera permission needed",
                message = "Grant camera access to take a shrimp image snapshot.",
                actionLabel = "Grant permission",
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
                        contentDescription = "Selected shrimp image",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                inputState.imageUri != null -> {
                    AsyncImage(
                        model = inputState.imageUri,
                        contentDescription = "Selected shrimp image",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                else -> {
                    CameraScanFrame(
                        modifier = Modifier.fillMaxSize(),
                        label = "Position shrimp within the frame for best results",
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
private fun TipsCard() {
    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 3.dp) {
        Text(
            text = "Tips for better analysis",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.Bold,
        )
        TipRow("Place shrimp clearly in the center frame")
        TipRow("Ensure good lighting without harsh shadows")
        TipRow("Hold device steady to avoid blur")
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
    labels: List<String>,
    selectedGroundTruthLabel: String?,
    modelInfo: ModelInfo,
    benchmarkMetrics: BenchmarkMetrics,
    runtimeDelegateName: String,
    debugInfoEnabled: Boolean,
    cameraFps: Double?,
    onGroundTruthSelected: (String?) -> Unit,
    onSaveResult: () -> Unit,
    onScanAnother: () -> Unit,
    onHome: () -> Unit,
    onHistory: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        ResultOverviewCard(inputState = inputState, result = result)

        Text(
            text = "Next Steps",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.Bold,
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            NextStepCard(
                title = if (result.statusKind() == ResultStatusKind.Disease) {
                    "Isolate Sample"
                } else {
                    "Routine Water Check"
                },
                message = if (result.statusKind() == ResultStatusKind.Disease) {
                    "Separate affected pond batch and confirm with field observations."
                } else {
                    "Continue daily DO, pH, and ammonia monitoring."
                },
                modifier = Modifier.weight(1f),
            )
            NextStepCard(
                title = "Monitor Others",
                message = "Observe feeding behavior and visible shell changes in nearby shrimp.",
                modifier = Modifier.weight(1f),
            )
        }

        ResultActions(
            onSaveResult = onSaveResult,
            onScanAnother = onScanAnother,
            onHome = onHome,
            onHistory = onHistory,
        )

        GroundTruthCard(
            labels = labels,
            selectedGroundTruthLabel = selectedGroundTruthLabel,
            resultCorrectness = result.isCorrect,
            onGroundTruthSelected = onGroundTruthSelected,
        )

        DiagnosisDetailsCard(
            result = result,
            modelInfo = modelInfo,
        )

        AttentionMapCard()
        TopPredictionList(predictions = result.top3Predictions)
        MetricsCard(result = result, benchmarkMetrics = benchmarkMetrics)

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
                        contentDescription = "Scanned shrimp",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                inputState.imageUri != null -> {
                    AsyncImage(
                        model = inputState.imageUri,
                        contentDescription = "Scanned shrimp",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }

                else -> {
                    EmptyState(
                        title = "Image unavailable",
                        message = "The scan result was generated without a displayable preview.",
                    )
                }
            }
            ResultStatusBadge(
                text = "View Heatmap",
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
                    kind = result.statusKind(),
                )
                Text(
                    text = result.displayPredictionText(),
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                )
            }
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = BenchmarkUtils.confidenceText(result.confidence),
                    style = MaterialTheme.typography.displayLarge,
                    color = result.statusColor(),
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Confidence",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        LinearProgressIndicator(
            progress = { result.confidence.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(),
            color = result.statusColor(),
            trackColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.42f),
        )

        CVioCard(containerColor = CVioSurfaceContainerLow, tonalElevation = 0.dp) {
            Text(
                text = "What this means",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = resultExplanation(result),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun NextStepCard(
    title: String,
    message: String,
    modifier: Modifier = Modifier,
) {
    CVioCard(
        modifier = modifier,
        containerColor = CVioSurfaceContainerLowest,
        tonalElevation = 3.dp,
    ) {
        CVioIconBubbleForStep(title)
        Text(
            text = title,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun CVioIconBubbleForStep(title: String) {
    val disease = title.contains("isolate", ignoreCase = true)
    Box(
        modifier = Modifier
            .height(44.dp)
            .fillMaxWidth(),
        contentAlignment = Alignment.CenterStart,
    ) {
        ResultStatusBadge(
            text = if (disease) "!" else "OK",
            kind = if (disease) ResultStatusKind.Disease else ResultStatusKind.Healthy,
        )
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
                text = "Save to History",
                onClick = onSaveResult,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = "Scan Another",
                onClick = onScanAnother,
                modifier = Modifier.weight(1f),
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SecondaryActionButton(
                text = "Home",
                onClick = onHome,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = "History",
                onClick = onHistory,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@Composable
private fun DiagnosisDetailsCard(
    result: ClassificationResult,
    modelInfo: ModelInfo,
) {
    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 3.dp) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                Text(
                    text = "Diagnostic Report",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Sample #${result.timestamp.toString().takeLast(4)}",
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
            }
            ResultStatusBadge(
                text = when (result.statusKind()) {
                    ResultStatusKind.Healthy -> "Low Risk"
                    ResultStatusKind.Disease -> "High Risk"
                    ResultStatusKind.Warning -> "Review"
                    ResultStatusKind.Neutral -> "N/A"
                },
                kind = result.statusKind(),
            )
        }

        Text(
            text = "Primary Detection",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = result.displayPredictionText(),
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = resultExplanation(result),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            CVioMetricTile(
                label = "Model",
                value = modelInfo.modelFamily,
                modifier = Modifier.weight(1f),
            )
            CVioMetricTile(
                label = "Time",
                value = BenchmarkUtils.latencyText(result.inferenceTimeMs),
                modifier = Modifier.weight(1f),
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            CVioMetricTile(
                label = "Threshold",
                value = BenchmarkUtils.confidenceText(result.threshold),
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.tertiary,
            )
            CVioMetricTile(
                label = "Input",
                value = BenchmarkUtils.inputSizeText(modelInfo),
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.secondary,
            )
        }
    }
}

@Composable
private fun AttentionMapCard() {
    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 3.dp) {
        CVioSectionHeader(
            title = "Attention Map",
            subtitle = "Heatmap is not available for this model.",
        )
        CameraScanFrame(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1.65f),
            label = "Attention Map Overlay",
        )
    }
}

@Composable
private fun GroundTruthCard(
    labels: List<String>,
    selectedGroundTruthLabel: String?,
    resultCorrectness: Boolean?,
    onGroundTruthSelected: (String?) -> Unit,
) {
    CVioCard(containerColor = CVioSurfaceContainerLow, tonalElevation = 2.dp) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = "Ground truth",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Optional label for testing accuracy.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            ResultStatusBadge(
                text = correctnessText(resultCorrectness),
                kind = when (resultCorrectness) {
                    true -> ResultStatusKind.Healthy
                    false -> ResultStatusKind.Disease
                    null -> ResultStatusKind.Neutral
                },
            )
        }

        if (labels.isEmpty()) {
            Text(
                text = "No labels loaded. Accuracy is N/A.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        } else {
            var expanded by remember { mutableStateOf(false) }
            Box(modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(
                    onClick = { expanded = true },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(18.dp),
                ) {
                    Text(
                        text = selectedGroundTruthLabel ?: "No ground truth",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                DropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false },
                ) {
                    DropdownMenuItem(
                        text = { Text(text = "No ground truth") },
                        onClick = {
                            expanded = false
                            onGroundTruthSelected(null)
                        },
                    )
                    labels.forEach { label ->
                        DropdownMenuItem(
                            text = { Text(text = label) },
                            onClick = {
                                expanded = false
                                onGroundTruthSelected(label)
                            },
                        )
                    }
                }
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
            title = "Debug Info",
            subtitle = "Compact technical diagnostics for research and testing.",
        )
        CompactInfoRow("Source", inputState.source?.displayName ?: "N/A")
        CompactInfoRow("Image URI", inputState.imageUri?.toString() ?: "Camera snapshot / N/A")
        CompactInfoRow("Model", modelInfo.modelFile)
        CompactInfoRow("Delegate", runtimeDelegateName)
        CompactInfoRow("Input size", BenchmarkUtils.inputSizeText(modelInfo))
        CompactInfoRow("Input shape", BenchmarkUtils.shapeText(modelInfo.input.shape))
        CompactInfoRow("Output shape", BenchmarkUtils.shapeText(modelInfo.output.shape))
        CompactInfoRow("Threshold", BenchmarkUtils.confidenceText(result.threshold))
        CompactInfoRow(
            "Camera FPS",
            cameraFps?.let { String.format(Locale.US, "%.1f", it) } ?: "N/A",
        )
        CompactInfoRow("Preprocess", BenchmarkUtils.latencyText(result.preprocessingTimeMs))
        CompactInfoRow("Model inference", BenchmarkUtils.latencyText(result.modelInferenceTimeMs))
        CompactInfoRow("Postprocess", BenchmarkUtils.latencyText(result.postprocessingTimeMs))
        CompactInfoRow("Total pipeline", BenchmarkUtils.latencyText(result.totalTimeMs))
    }
}

private fun ClassificationResult.displayPredictionText(): String {
    return if (isAboveThreshold) {
        predictedClass
    } else {
        "$rawTop1Label / Unknown"
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
        ResultStatusKind.Healthy -> "Status: Healthy"
        ResultStatusKind.Disease -> "Disease detected"
        ResultStatusKind.Warning -> "Low confidence"
        ResultStatusKind.Neutral -> "N/A"
    }
}

private fun ClassificationResult.statusColor() = when (statusKind()) {
    ResultStatusKind.Healthy -> HealthyGreen
    ResultStatusKind.Disease -> DiseaseRed
    ResultStatusKind.Warning -> WarningOrange
    ResultStatusKind.Neutral -> DarkNavy
}

private fun resultExplanation(result: ClassificationResult): String {
    return when (result.statusKind()) {
        ResultStatusKind.Healthy -> "No visible disease signs detected in this shrimp image. Continue routine water quality monitoring and observe feeding behavior."
        ResultStatusKind.Disease -> "The AI detected visual disease indicators. Treat this as field triage, isolate suspicious samples, and confirm with pond conditions before treatment decisions."
        ResultStatusKind.Warning -> "The top prediction is below the confidence threshold. Retake the image with better lighting or inspect the top-3 confidence breakdown."
        ResultStatusKind.Neutral -> "No diagnostic interpretation is available."
    }
}

private fun correctnessText(isCorrect: Boolean?): String {
    return when (isCorrect) {
        true -> "Correct"
        false -> "Incorrect"
        null -> "Accuracy: N/A"
    }
}
