package rs.smobile.shrimpdisease.ui.inference

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
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
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import rs.smobile.shrimpdisease.CameraCaptureCard
import rs.smobile.shrimpdisease.ClassificationUiState
import rs.smobile.shrimpdisease.InferenceInputUiState
import rs.smobile.shrimpdisease.InferenceSource
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.ui.components.EmptyState
import rs.smobile.shrimpdisease.ui.components.InferenceActionPanel
import rs.smobile.shrimpdisease.ui.components.MetricsCard
import rs.smobile.shrimpdisease.ui.components.PredictionCard
import rs.smobile.shrimpdisease.ui.components.TopPredictionList
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.theme.CVioPrimaryFixedDim
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
    var isRetakingSnapshot by remember { mutableStateOf(false) }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        CVioSectionHeader(
            title = "Capture",
            subtitle = "Use a clear image with good lighting for more reliable classification.",
        )

        InputPreviewCard(
            inputState = inputState,
            isRetakingSnapshot = isRetakingSnapshot,
            cameraPermissionGranted = cameraPermissionGranted,
            onRequestCameraPermission = onRequestCameraPermission,
            onSnapshot = { bitmap ->
                isRetakingSnapshot = false
                onSnapshot(bitmap)
            },
            onFrameObserved = onFrameObserved,
            onCameraError = onCameraError,
        )

        RetakeOrSelectButton(
            inputState = inputState,
            isRetakingSnapshot = isRetakingSnapshot,
            onRetakeSnapshot = {
                isRetakingSnapshot = true
            },
            onSelectAnotherImage = onSelectAnotherImage,
        )

        GroundTruthCard(
            labels = labels,
            selectedGroundTruthLabel = selectedGroundTruthLabel,
            resultCorrectness = classificationState.result?.isCorrect,
            onGroundTruthSelected = onGroundTruthSelected,
        )

        InferenceActionPanel(
            hasImage = inputState.bitmap != null,
            hasResult = classificationState.result != null,
            isLoading = classificationState.isLoading,
            onRunInference = onRunInference,
            onSaveResult = onSaveResult,
            onGoHome = onHome,
            onHistory = onHistory,
        )

        if (classificationState.result != null || classificationState.errorMessage != null) {
            PredictionCard(
                result = classificationState.result,
                isLoading = false,
                errorMessage = classificationState.errorMessage,
            )
        }

        classificationState.result?.let { result ->
            TopPredictionList(predictions = result.top3Predictions)

            MetricsCard(
                result = result,
                benchmarkMetrics = benchmarkMetrics,
            )
        }

        if (debugInfoEnabled) {
            DebugInfoCard(
                inputState = inputState,
                result = classificationState.result,
                modelInfo = modelInfo,
                runtimeDelegateName = runtimeDelegateName,
                cameraFps = cameraFps,
            )
        }
    }
}

@Composable
private fun RetakeOrSelectButton(
    inputState: InferenceInputUiState,
    isRetakingSnapshot: Boolean,
    onRetakeSnapshot: () -> Unit,
    onSelectAnotherImage: () -> Unit,
) {
    when (inputState.source) {
        InferenceSource.SNAPSHOT -> {
            OutlinedButton(
                onClick = onRetakeSnapshot,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(text = if (isRetakingSnapshot || inputState.bitmap == null) "Camera ready" else "Retake Snapshot")
            }
        }

        InferenceSource.GALLERY -> {
            OutlinedButton(
                onClick = onSelectAnotherImage,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(text = "Select Another Image")
            }
        }

        null -> {
            OutlinedButton(
                onClick = onSelectAnotherImage,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(text = "Select Image")
            }
        }
    }
}

@Composable
private fun GroundTruthCard(
    labels: List<String>,
    selectedGroundTruthLabel: String?,
    resultCorrectness: Boolean?,
    onGroundTruthSelected: (String?) -> Unit,
) {
    CVioCard {
            Text(
                text = "Ground truth label",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
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
                    ) {
                        Text(text = selectedGroundTruthLabel ?: "Accuracy: N/A")
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
            CVioStatusChip(
                text = "Accuracy: ${correctnessText(resultCorrectness)}",
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
            )
    }
}

@Composable
private fun InputPreviewCard(
    inputState: InferenceInputUiState,
    isRetakingSnapshot: Boolean,
    cameraPermissionGranted: Boolean,
    onRequestCameraPermission: () -> Unit,
    onSnapshot: (Bitmap) -> Unit,
    onFrameObserved: () -> Unit,
    onCameraError: (String) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        CVioSectionHeader(
            title = "Image preview",
            subtitle = "Position the shrimp in the center frame.",
        )

        val showCameraPreview = isRetakingSnapshot || (inputState.isCameraActive && inputState.bitmap == null)
        if (showCameraPreview) {
            if (!cameraPermissionGranted) {
                EmptyState(
                    title = "Camera permission needed",
                    message = "Grant camera access to take a shrimp image snapshot.",
                )
                Button(
                    onClick = onRequestCameraPermission,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(text = "Grant permission")
                }
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
        } else {
            ImagePreview(inputState)
        }
    }
}

@Composable
private fun ImagePreview(inputState: InferenceInputUiState) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(3f / 4f),
        shape = MaterialTheme.shapes.large,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.inverseSurface),
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
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center,
                    ) {
                        EmptyState(
                            title = "No image selected",
                            message = "Go back Home and select an image or open the camera.",
                        )
                    }
                }
            }
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .fillMaxSize()
                    .padding(24.dp)
                    .clip(RoundedCornerShape(28.dp))
                    .background(CVioPrimaryFixedDim.copy(alpha = 0.10f)),
            )
        }
    }
}

@Composable
private fun DebugInfoCard(
    inputState: InferenceInputUiState,
    result: rs.smobile.shrimpdisease.classifier.ClassificationResult?,
    modelInfo: ModelInfo,
    runtimeDelegateName: String,
    cameraFps: Double?,
) {
    CVioCard(containerColor = MaterialTheme.colorScheme.surfaceVariant) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                text = "Debug info",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            DebugRow("Source", inputState.source?.displayName ?: "N/A")
            DebugRow("Image URI", inputState.imageUri?.toString() ?: "Camera snapshot / N/A")
            DebugRow("Model", modelInfo.modelFile)
            DebugRow("Delegate", runtimeDelegateName)
            DebugRow("Input size", BenchmarkUtils.inputSizeText(modelInfo))
            DebugRow("Input shape", BenchmarkUtils.shapeText(modelInfo.input.shape))
            DebugRow("Output shape", BenchmarkUtils.shapeText(modelInfo.output.shape))
            DebugRow(
                "Threshold",
                result?.threshold?.let(BenchmarkUtils::confidenceText) ?: "N/A",
            )
            DebugRow(
                "Camera FPS",
                cameraFps?.let { String.format(Locale.US, "%.1f", it) } ?: "N/A",
            )
            if (result != null) {
                DebugRow("Preprocess", BenchmarkUtils.latencyText(result.preprocessingTimeMs))
                DebugRow("Model inference", BenchmarkUtils.latencyText(result.modelInferenceTimeMs))
                DebugRow("Postprocess", BenchmarkUtils.latencyText(result.postprocessingTimeMs))
                DebugRow("Total pipeline", BenchmarkUtils.latencyText(result.totalTimeMs))
            }
        }
    }
}

@Composable
private fun DebugRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(0.8f),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            modifier = Modifier
                .weight(1.2f)
                .padding(start = 12.dp),
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
        )
    }
}

private fun correctnessText(isCorrect: Boolean?): String {
    return when (isCorrect) {
        true -> "Correct"
        false -> "Incorrect"
        null -> "N/A"
    }
}
