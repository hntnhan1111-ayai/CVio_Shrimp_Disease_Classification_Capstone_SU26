package rs.smobile.shrimpdisease

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import dagger.hilt.android.AndroidEntryPoint
import rs.smobile.shrimpdisease.classifier.ClassificationResult
import rs.smobile.shrimpdisease.classifier.InputLayout
import rs.smobile.shrimpdisease.classifier.ModelConfig
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.classifier.Prediction
import rs.smobile.shrimpdisease.classifier.TensorInfo
import rs.smobile.shrimpdisease.ui.theme.ShrimpDiseaseTheme
import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToInt

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContent {
            val viewModel: MainViewModel = hiltViewModel()
            val classificationState by viewModel.classificationState.collectAsStateWithLifecycle()
            val metricsState by viewModel.metricsState.collectAsStateWithLifecycle()
            val inferenceLogs by viewModel.inferenceLogs.collectAsStateWithLifecycle()
            val expectedLabel by viewModel.expectedLabel.collectAsStateWithLifecycle()
            val modelInfo by viewModel.modelInfo.collectAsStateWithLifecycle()
            val labels by viewModel.labels.collectAsStateWithLifecycle()
            val availableModels by viewModel.availableModels.collectAsStateWithLifecycle()
            var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
            var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
            var selectedSource by remember { mutableStateOf<InferenceSource?>(null) }
            var isCameraActive by remember { mutableStateOf(false) }
            var hasCameraPermission by remember {
                mutableStateOf(
                    ContextCompat.checkSelfPermission(
                        this,
                        Manifest.permission.CAMERA,
                    ) == PackageManager.PERMISSION_GRANTED,
                )
            }
            val context = LocalContext.current
            val cameraPermissionLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.RequestPermission()
            ) { granted ->
                hasCameraPermission = granted
                if (!granted) {
                    isCameraActive = false
                }
            }
            val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
                selectedImageUri = uri
                isCameraActive = false
                if (uri == null) return@rememberLauncherForActivityResult

                runCatching { decodeBitmapFromUri(context, uri) }
                    .onSuccess { bitmap ->
                        selectedBitmap = bitmap
                        selectedSource = InferenceSource.GALLERY
                        viewModel.clearClassification()
                    }
                    .onFailure { error ->
                        viewModel.showError(error.message ?: "Could not decode selected image.")
                    }
            }

            val exportLogsLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
                if (uri == null) return@rememberLauncherForActivityResult
                val saved = saveInferenceLogsToUri(context, uri, inferenceLogs)
                if (saved) {
                    Toast.makeText(context, "Inference logs saved", Toast.LENGTH_SHORT).show()
                } else {
                    viewModel.showError("Failed to save inference logs.")
                }
            }

            ShrimpDiseaseTheme {
                MainScreen(
                    selectedImageUri = selectedImageUri,
                    selectedBitmap = selectedBitmap,
                    selectedSource = selectedSource,
                    isCameraActive = isCameraActive,
                    cameraPermissionGranted = hasCameraPermission,
                    onRequestCameraPermission = { cameraPermissionLauncher.launch(Manifest.permission.CAMERA) },
                    onToggleCamera = {
                        if (!hasCameraPermission) {
                            isCameraActive = true
                            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        } else {
                            isCameraActive = !isCameraActive
                        }
                    },
                    classificationState = classificationState,
                    metricsState = metricsState,
                    logs = inferenceLogs,
                    modelInfo = modelInfo,
                    labels = labels,
                    expectedLabel = expectedLabel,
                    onExpectedLabelChange = viewModel::setExpectedLabel,
                    onResetMetrics = viewModel::resetMetrics,
                    onPickImage = {
                        isCameraActive = false
                        selectedImageUri = null
                        selectedBitmap = null
                        selectedSource = null
                        imagePicker.launch("image/*")
                    },
                    onTakeSnapshot = { bitmap ->
                        selectedImageUri = null
                        selectedBitmap = bitmap
                        selectedSource = InferenceSource.SNAPSHOT
                        viewModel.clearClassification()
                    },
                    onClearInput = {
                        selectedImageUri = null
                        selectedBitmap = null
                        selectedSource = null
                        viewModel.clearClassification()
                    },
                    onPredict = {
                        val bitmap = selectedBitmap
                        when {
                            bitmap == null -> {
                                viewModel.showError("Choose an image or take a snapshot before predicting.")
                            }

                            expectedLabel == null -> {
                                viewModel.showError("Select a ground-truth label before predicting.")
                            }

                            else -> {
                                viewModel.classifyBitmap(bitmap, selectedSource ?: InferenceSource.GALLERY)
                            }
                        }
                    },
                    onFrameObserved = viewModel::recordCameraFrame,
                    onCameraError = viewModel::showError,
                    availableModels = availableModels,
                    onLoadModel = viewModel::loadModel,
                    onSetThreshold = viewModel::setConfidenceThreshold,
                    onExportLogs = { exportLogsLauncher.launch("inference_logs.txt") },
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
/**
 * Main single-screen Compose UI for image input, model selection, prediction,
 * runtime metrics, and exportable inference logs.
 */
@Composable
private fun MainScreen(
    selectedImageUri: Uri?,
    selectedBitmap: Bitmap?,
    selectedSource: InferenceSource?,
    isCameraActive: Boolean,
    cameraPermissionGranted: Boolean,
    onRequestCameraPermission: () -> Unit,
    onToggleCamera: () -> Unit,
    onExportLogs: () -> Unit,
    classificationState: ClassificationUiState,
    metricsState: MetricsUiState,
    logs: List<InferenceLogEntry>,
    modelInfo: ModelInfo,
    labels: List<String>,
    expectedLabel: String?,
    onExpectedLabelChange: (String?) -> Unit,
    onResetMetrics: () -> Unit,
    onPickImage: () -> Unit,
    onTakeSnapshot: (Bitmap) -> Unit,
    onClearInput: () -> Unit,
    onPredict: () -> Unit,
    onFrameObserved: () -> Unit,
    onCameraError: (String) -> Unit,
    availableModels: List<String>,
    onLoadModel: (String) -> Unit,
    onSetThreshold: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Text(
                        text = stringResource(R.string.app_name),
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onSurface,
                ),
            )
        },
    ) { innerPadding ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            ) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text(
                        text = "Input image",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = selectedSource?.displayName ?: "Load camera or choose a gallery image.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Button(
                            onClick = onPickImage,
                            modifier = Modifier.weight(1f),
                        ) {
                            Text(text = "Choose image")
                        }
                        OutlinedButton(
                            onClick = onToggleCamera,
                            modifier = Modifier.weight(1f),
                        ) {
                            Text(text = if (isCameraActive) "Stop camera" else "Load camera")
                        }
                    }

                    if (!cameraPermissionGranted && isCameraActive) {
                        Text(
                            text = "Camera permission is required. Grant permission to start camera.",
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Button(onClick = onRequestCameraPermission, modifier = Modifier.fillMaxWidth()) {
                            Text(text = "Grant camera permission")
                        }
                    }

                    if (isCameraActive) {
                        CameraCaptureCard(
                            enabled = cameraPermissionGranted,
                            capturedBitmap = selectedBitmap.takeIf { selectedSource == InferenceSource.SNAPSHOT },
                            onSnapshot = onTakeSnapshot,
                            onRetake = onClearInput,
                            onFrameObserved = onFrameObserved,
                            onError = onCameraError,
                        )
                    } else {
                        ImagePreview(selectedBitmap, selectedImageUri)
                    }
                }
            }

            ModelSelectionPanel(
                modelInfo = modelInfo,
                availableModels = availableModels,
                onLoadModel = onLoadModel,
            )
            GroundTruthPanel(
                labels = labels,
                selectedLabel = expectedLabel,
                onSelectedLabelChange = onExpectedLabelChange,
            )
            PredictPanel(
                canPredict = selectedBitmap != null && expectedLabel != null && !classificationState.isLoading,
                hasImage = selectedBitmap != null,
                hasGroundTruth = expectedLabel != null,
                onPredict = onPredict,
                onClearInput = onClearInput,
            )
            ResultPanel(
                classificationState = classificationState,
                expectedLabel = expectedLabel,
                threshold = metricsState.confidenceThreshold,
            )
            ThresholdPanel(
                threshold = metricsState.confidenceThreshold,
                onSetThreshold = onSetThreshold,
            )
            MetricsPanel(metricsState, onResetMetrics)
            LogsPanel(logs, onExportLogs)
            ModelPanel(modelInfo, metricsState.confidenceThreshold)
        }
    }
}

/** Shows either the selected gallery/snapshot bitmap or an empty input state. */
@Composable
private fun ImagePreview(selectedBitmap: Bitmap?, selectedImageUri: Uri?) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        if (selectedBitmap != null) {
            Image(
                bitmap = selectedBitmap.asImageBitmap(),
                contentDescription = stringResource(R.string.app_name),
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
        } else if (selectedImageUri == null) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "No image selected",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            AsyncImage(
                model = selectedImageUri,
                contentDescription = stringResource(R.string.app_name),
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
        }
    }
}

@Composable
private fun ModelSelectionPanel(
    modelInfo: ModelInfo,
    availableModels: List<String>,
    onLoadModel: (String) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Model selection",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Current: ${modelInfo.modelFile}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (availableModels.isEmpty()) {
                Text(
                    text = "No .tflite models found in assets",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                var expanded by remember { mutableStateOf(false) }
                var selected by remember(modelInfo.modelFile, availableModels) {
                    mutableStateOf(modelInfo.modelFile.takeIf { it in availableModels } ?: availableModels.first())
                }

                Box(modifier = Modifier.fillMaxWidth()) {
                    OutlinedButton(
                        onClick = { expanded = true },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            text = selected,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                    DropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        availableModels.forEach { model ->
                            DropdownMenuItem(
                                text = {
                                    Text(
                                        text = model,
                                        maxLines = 2,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                },
                                onClick = {
                                    selected = model
                                    expanded = false
                                },
                            )
                        }
                    }
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { onLoadModel(selected) }) {
                        Text(text = "Load model")
                    }
                    OutlinedButton(onClick = { onLoadModel(ModelConfig.MODEL_FILE) }) {
                        Text(text = "Load default")
                    }
                }
            }
        }
    }
}

@Composable
private fun ThresholdPanel(
    threshold: Float,
    onSetThreshold: (Float) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(
                text = "Confidence threshold",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Slider(
                value = threshold,
                onValueChange = onSetThreshold,
                valueRange = 0f..1f,
            )
            Text(text = String.format(Locale.US, "Current: %.1f%%", threshold * 100.0f))
        }
    }
}

@Composable
private fun GroundTruthPanel(
    labels: List<String>,
    selectedLabel: String?,
    onSelectedLabelChange: (String?) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Ground truth for session accuracy",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Accuracy is computed only when a ground-truth label is selected.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            labels.chunked(2).forEach { rowLabels ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    rowLabels.forEach { label ->
                        val selected = selectedLabel == label
                        if (selected) {
                            Button(
                                onClick = { onSelectedLabelChange(label) },
                                modifier = Modifier.weight(1f),
                            ) {
                                Text(text = label)
                            }
                        } else {
                            OutlinedButton(
                                onClick = { onSelectedLabelChange(label) },
                                modifier = Modifier.weight(1f),
                            ) {
                                Text(text = label)
                            }
                        }
                    }
                    if (rowLabels.size == 1) {
                        Spacer(modifier = Modifier.weight(1f))
                    }
                }
            }

            TextButton(onClick = { onSelectedLabelChange(null) }) {
                Text(text = "Clear ground truth")
            }
        }
    }
}

@Composable
private fun PredictPanel(
    canPredict: Boolean,
    hasImage: Boolean,
    hasGroundTruth: Boolean,
    onPredict: () -> Unit,
    onClearInput: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Predict",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            if (!hasImage || !hasGroundTruth) {
                Text(
                    text = when {
                        !hasImage -> "Load camera/take snapshot or choose an image first."
                        !hasGroundTruth -> "Select ground truth before predicting."
                        else -> ""
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    enabled = canPredict,
                    onClick = onPredict,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Predict")
                }
                OutlinedButton(
                    enabled = hasImage,
                    onClick = onClearInput,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Clear image")
                }
            }
        }
    }
}

@Composable
private fun ResultPanel(
    classificationState: ClassificationUiState,
    expectedLabel: String?,
    threshold: Float,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Result",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )

            when {
                classificationState.isLoading -> {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(text = "Running inference...", style = MaterialTheme.typography.bodyMedium)
                }

                classificationState.errorMessage != null -> {
                    Text(
                        text = classificationState.errorMessage,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }

                classificationState.result != null -> {
                    val result = classificationState.result
                    val groundTruthMatches = expectedLabel?.let { it == result.top1.label }
                    val passesThreshold = result.top1.confidence >= threshold
                    val accepted = if (expectedLabel == null) {
                        passesThreshold
                    } else {
                        groundTruthMatches == true && passesThreshold
                    }
                    ResultRow("Disease", result.top1.label)
                    ResultRow("Confidence", result.top1.confidence.asPercent())
                    ResultRow("Inference time", result.inferenceTimeMs.msText())
                    ResultRow("Ground truth", expectedLabel ?: "Not selected")
                    ResultRow(
                        "Ground truth check",
                        when (groundTruthMatches) {
                            true -> "Match"
                            false -> "Mismatch"
                            null -> "N/A"
                        },
                    )
                    ResultRow("Threshold", threshold.asPercent())
                    ResultRow("Threshold check", if (passesThreshold) "Pass" else "Fail")
                    ResultRow("Decision", if (accepted) "Accept" else "Reject")

                    if (!accepted) {
                        Text(
                            text = when {
                                groundTruthMatches == false -> "Reject: prediction does not match the selected ground truth."
                                !passesThreshold -> "Reject: confidence is below the selected threshold."
                                else -> "Reject: select ground truth and predict again."
                            },
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }

                else -> Text(text = "Awaiting image", style = MaterialTheme.typography.bodyLarge)
            }
        }
    }
}

@Composable
private fun MetricsPanel(
    metrics: MetricsUiState,
    onResetMetrics: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Runtime metrics",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                )
                TextButton(onClick = onResetMetrics) {
                    Text(text = "Reset")
                }
            }

            ResultRow("Total inferences", metrics.totalInferences.toString())
            ResultRow("Accepted / rejected", "${metrics.acceptedInferences} / ${metrics.rejectedInferences}")
            ResultRow("Last inference", metrics.lastInferenceTimeMs?.msText() ?: "N/A")
            ResultRow("Avg inference", metrics.averageInferenceTimeMs?.msText() ?: "N/A")
            ResultRow("Speed", metrics.speedIps?.ipsText() ?: "N/A")
            ResultRow("Avg speed", metrics.averageSpeedIps?.ipsText() ?: "N/A")
            ResultRow("Camera FPS", metrics.cameraFps?.fpsText() ?: "N/A")
            ResultRow("Session accuracy", metrics.accuracyText())
            if (metrics.perClassPredictionStats.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Predict fail by class",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                metrics.perClassPredictionStats.entries.sortedBy { it.key }.forEach { (label, stats) ->
                    ResultRow(label, stats.failureText())
                }
            }
        }
    }
}

@Composable
private fun LogsPanel(logs: List<InferenceLogEntry>, onExportLogs: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Inference logs",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                )
                if (logs.isNotEmpty()) {
                    TextButton(onClick = onExportLogs) {
                        Text(text = "Export logs")
                    }
                }
            }
            if (logs.isEmpty()) {
                Text(text = "No logs yet", style = MaterialTheme.typography.bodyLarge)
            } else {
                logs.take(8).forEach { entry ->
                    Text(
                        text = entry.logText(),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelPanel(
    modelInfo: ModelInfo,
    threshold: Float,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Model",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
            ResultRow("File", modelInfo.modelFile)
            ResultRow("Label file", modelInfo.labelFile)
            ResultRow("Family", modelInfo.modelFamily)
            ResultRow("Input", modelInfo.input.shortText())
            ResultRow("Layout", modelInfo.inputLayout.name)
            ResultRow("Output", modelInfo.output.shortText())
            ResultRow("Classes / labels", "${modelInfo.outputClassCount} / ${modelInfo.labelsCount}")
            ResultRow("Threshold", threshold.asPercent())
            if (modelInfo.warnings.isNotEmpty()) {
                Spacer(modifier = Modifier.height(4.dp))
                modelInfo.warnings.forEach { warning ->
                    Text(
                        text = warning,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

@Composable
private fun ResultRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            modifier = Modifier
                .weight(1.2f)
                .padding(start = 12.dp),
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.SemiBold,
        )
    }
}

private fun Float.asPercent(): String = String.format(Locale.US, "%.1f%%", this * 100.0f)

private fun Double.msText(): String = String.format(Locale.US, "%.1f ms", this)

private fun Double.ipsText(): String = String.format(Locale.US, "%.2f img/s", this)

private fun Double.fpsText(): String = String.format(Locale.US, "%.1f fps", this)

private fun TensorInfo.shortText(): String {
    return "${shape.joinToString(prefix = "[", postfix = "]")} $dataType"
}

private fun MetricsUiState.accuracyText(): String {
    if (evaluatedInferences == 0) return "N/A"
    val accuracy = correctInferences * 100.0 / evaluatedInferences
    return String.format(Locale.US, "%.1f%% (%d/%d)", accuracy, correctInferences, evaluatedInferences)
}

private fun MetricsUiState.summaryText(): String {
    return buildString {
        append("total=$totalInferences accepted=$acceptedInferences rejected=$rejectedInferences evaluated=$evaluatedInferences correct=$correctInferences")
        append(" last=${lastInferenceTimeMs?.msText() ?: "N/A"} avg=${averageInferenceTimeMs?.msText() ?: "N/A"}")
        append(" speed=${speedIps?.ipsText() ?: "N/A"} avgSpeed=${averageSpeedIps?.ipsText() ?: "N/A"} fps=${cameraFps?.fpsText() ?: "N/A"}")
        append(" threshold=${String.format(Locale.US, "%.1f%%", confidenceThreshold * 100.0f)}")
    }
}

private fun MetricsUiState.classFailText(): String? {
    if (perClassPredictionStats.isEmpty()) return null
    return perClassPredictionStats.entries.sortedBy { it.key }
        .joinToString(", ") { "${it.key} ${it.value.failureText()}" }
}

private fun InferenceLogEntry.logText(): String {
    val status = if (accepted) "accept" else "reject"
    val correctness = correct?.let { if (it) "correct" else "wrong" } ?: "unscored"
    return buildString {
        appendLine("[${timestampMs}ms] $source")
        appendLine("Prediction: $label (${confidence * 100.0f}%)")
        appendLine("Threshold: ${threshold * 100.0f}% | Result: $status / $correctness")
        appendLine("Expected: ${expectedLabel ?: "N/A"}")
        appendLine("Inference time: ${inferenceTimeMs.msText()}")
        appendLine("Metrics: ${metricsSnapshot.summaryText()}")
        metricsSnapshot.classFailText()?.let {
            appendLine("Predict fail by class: $it")
        }
    }.trimEnd()
}

private fun InferenceLogEntry.exportText(): String {
    val status = if (accepted) "accept" else "reject"
    val correctness = correct?.let { if (it) "correct" else "wrong" } ?: "unscored"
    return buildString {
        appendLine("[${timestampMs}ms] $source")
        appendLine("Prediction: $label (${confidence * 100.0f}%)")
        appendLine("Threshold: ${threshold * 100.0f}% | Result: $status / $correctness")
        appendLine("Expected: ${expectedLabel ?: "N/A"}")
        appendLine("Inference time: ${inferenceTimeMs.msText()}")
        appendLine("Metrics: ${metricsSnapshot.summaryText()}")
        metricsSnapshot.classFailText()?.let {
            appendLine("Predict fail by class: $it")
        }
    }.trimEnd()
}

private fun saveInferenceLogsToUri(context: Context, uri: Uri, logs: List<InferenceLogEntry>): Boolean {
    return try {
        context.contentResolver.openOutputStream(uri)?.bufferedWriter()?.use { writer ->
            writer.appendLine("Inference logs export")
            if (logs.isNotEmpty()) {
                val latestMetrics = logs.first().metricsSnapshot
                writer.appendLine("Metrics summary: ${latestMetrics.summaryText()}")
                latestMetrics.classFailText()?.let { writer.appendLine("Predict fail by class: $it") }
                writer.appendLine("---")
            }
            logs.reversed().forEach { entry ->
                writer.appendLine(entry.exportText())
                writer.appendLine()
            }
        } != null
    } catch (error: Throwable) {
        Log.e("MainActivity", "Failed to save inference logs", error)
        false
    }
}

/**
 * Decode gallery images into software bitmaps and downscale large photos before inference
 * to keep memory usage stable on mid-range phones.
 */
private fun decodeBitmapFromUri(context: Context, uri: Uri): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        val width = info.size.width
        val height = info.size.height
        val longestSide = max(width, height)
        if (longestSide > MAX_DECODE_SIDE) {
            val scale = MAX_DECODE_SIDE / longestSide.toFloat()
            decoder.setTargetSize(
                (width * scale).roundToInt().coerceAtLeast(1),
                (height * scale).roundToInt().coerceAtLeast(1),
            )
        }
    }
}

private const val MAX_DECODE_SIDE = 1280

@Preview(showBackground = true)
@Composable
private fun MainScreenPreview() {
    ShrimpDiseaseTheme {
        MainScreen(
            selectedImageUri = null,
            selectedBitmap = null,
            selectedSource = null,
            isCameraActive = false,
            cameraPermissionGranted = false,
            onRequestCameraPermission = {},
            onToggleCamera = {},
            classificationState = ClassificationUiState(
                result = ClassificationResult(
                    top1 = Prediction("WSSV", 0.942f),
                    topK = listOf(
                        Prediction("WSSV", 0.942f),
                        Prediction("BG", 0.041f),
                        Prediction("Healthy", 0.017f),
                    ),
                    inferenceTimeMs = 18.5,
                ),
            ),
            metricsState = MetricsUiState(
                totalInferences = 4,
                acceptedInferences = 4,
                evaluatedInferences = 4,
                correctInferences = 3,
                lastInferenceTimeMs = 18.5,
                averageInferenceTimeMs = 20.0,
                speedIps = 54.05,
                averageSpeedIps = 50.0,
                cameraFps = 28.3,
            ),
            logs = emptyList(),
            modelInfo = ModelInfo(
                modelFile = "shrimp_disease.tflite",
                labelFile = "labels.txt",
                modelFamily = "efficientnet_imagenet",
                input = TensorInfo("serving_default_input:0", listOf(1, 224, 224, 3), "FLOAT32", 0f, 0),
                output = TensorInfo("PartitionedCall:0", listOf(1, 4), "FLOAT32", 0f, 0),
                inputLayout = InputLayout.NHWC,
                outputClassCount = 4,
                labelsCount = 4,
                warnings = emptyList(),
            ),
            labels = listOf("Healthy", "BG", "WSSV", "WSSV_BG"),
            expectedLabel = "WSSV",
            onExpectedLabelChange = {},
            onResetMetrics = {},
            onPickImage = {},
            onTakeSnapshot = {},
            onClearInput = {},
            onPredict = {},
            onFrameObserved = {},
            onCameraError = {},
            availableModels = emptyList(),
            onLoadModel = {},
            onSetThreshold = {},
            onExportLogs = {},
        )
    }
}
