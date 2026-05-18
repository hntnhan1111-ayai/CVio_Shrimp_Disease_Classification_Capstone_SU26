package rs.smobile.shrimpdisease

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import rs.smobile.shrimpdisease.classifier.ClassificationResult
import rs.smobile.shrimpdisease.classifier.ModelConfig
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.classifier.ShrimpClassifier
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import javax.inject.Inject

enum class InferenceSource(val displayName: String) {
    GALLERY("gallery"),
    SNAPSHOT("snapshot"),
    LIVE_CAMERA("live_camera"),
}

data class ClassificationUiState(
    val result: ClassificationResult? = null,
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
)

data class ClassPredictionStats(
    val total: Int = 0,
    val incorrect: Int = 0,
) {
    fun failureText(): String = "$incorrect/$total"
}

data class MetricsUiState(
    val totalInferences: Int = 0,
    val acceptedInferences: Int = 0,
    val rejectedInferences: Int = 0,
    val evaluatedInferences: Int = 0,
    val correctInferences: Int = 0,
    val lastInferenceTimeMs: Double? = null,
    val averageInferenceTimeMs: Double? = null,
    val speedIps: Double? = null,
    val averageSpeedIps: Double? = null,
    val cameraFps: Double? = null,
    val confidenceThreshold: Float = ModelConfig.CONFIDENCE_THRESHOLD,
    val perClassPredictionStats: Map<String, ClassPredictionStats> = emptyMap(),
)

data class InferenceLogEntry(
    val timestampMs: Long,
    val source: String,
    val label: String,
    val confidence: Float,
    val inferenceTimeMs: Double,
    val threshold: Float,
    val accepted: Boolean,
    val expectedLabel: String?,
    val correct: Boolean?,
    val metricsSnapshot: MetricsUiState,
)

@HiltViewModel
class MainViewModel @Inject constructor(
    private val shrimpClassifier: ShrimpClassifier,
) : ViewModel() {

    private val _classificationState = MutableStateFlow(ClassificationUiState())
    val classificationState: StateFlow<ClassificationUiState> = _classificationState

    private val _metricsState = MutableStateFlow(MetricsUiState())
    val metricsState: StateFlow<MetricsUiState> = _metricsState

    private val _inferenceLogs = MutableStateFlow<List<InferenceLogEntry>>(emptyList())
    val inferenceLogs: StateFlow<List<InferenceLogEntry>> = _inferenceLogs

    private val _expectedLabel = MutableStateFlow<String?>(null)
    val expectedLabel: StateFlow<String?> = _expectedLabel

    private val _modelInfo = MutableStateFlow(shrimpClassifier.modelInfo)
    val modelInfo: StateFlow<ModelInfo> = _modelInfo

    private val _labels = MutableStateFlow(shrimpClassifier.labels)
    val labels: StateFlow<List<String>> = _labels

    private val _availableModels = MutableStateFlow(shrimpClassifier.availableModelsInAssets())
    val availableModels: StateFlow<List<String>> = _availableModels

    fun loadModel(modelFile: String) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                _classificationState.value = ClassificationUiState(isLoading = true)
                val family = shrimpClassifier.modelFamilyForAsset(modelFile)
                shrimpClassifier.loadModelFromAssets(modelFile, ModelConfig.LABEL_FILE, family)
                _modelInfo.value = shrimpClassifier.modelInfo
                _labels.value = shrimpClassifier.labels
                if (_expectedLabel.value !in shrimpClassifier.labels) {
                    _expectedLabel.value = null
                }
                resetMetrics(keepThreshold = true)
                _classificationState.value = ClassificationUiState()
            } catch (e: Throwable) {
                showError("Failed to load model: ${e.message}")
            }
        }
    }

    private val liveInferenceRunning = AtomicBoolean(false)
    private var totalInferenceTimeMs = 0.0
    private var fpsWindowStartNanos = SystemClock.elapsedRealtimeNanos()
    private var fpsWindowFrames = 0

    /**
     * Runs one classification request from a gallery image or captured snapshot.
     * The actual TensorFlow Lite call stays on Dispatchers.Default to avoid blocking UI.
     */
    fun classifyBitmap(bitmap: Bitmap, source: InferenceSource = InferenceSource.GALLERY) {
        _classificationState.value = ClassificationUiState(isLoading = true)
        viewModelScope.launch(Dispatchers.Default) {
            runClassification(bitmap, source)
        }
    }

    /**
     * Optional live-camera inference gate. Only one frame is processed at a time so
     * the analyzer cannot queue stale frames faster than the model can consume them.
     */
    fun classifyCameraFrame(bitmap: Bitmap): Boolean {
        if (!liveInferenceRunning.compareAndSet(false, true)) return false

        viewModelScope.launch(Dispatchers.Default) {
            try {
                runClassification(bitmap, InferenceSource.LIVE_CAMERA)
            } finally {
                liveInferenceRunning.set(false)
            }
        }
        return true
    }

    /**
     * Tracks camera FPS independently from model inference so the preview health is visible.
     */
    fun recordCameraFrame() {
        fpsWindowFrames += 1
        val now = SystemClock.elapsedRealtimeNanos()
        val elapsedNanos = now - fpsWindowStartNanos
        if (elapsedNanos < FPS_WINDOW_NANOS) return

        val fps = fpsWindowFrames * 1_000_000_000.0 / elapsedNanos
        fpsWindowFrames = 0
        fpsWindowStartNanos = now
        _metricsState.value = _metricsState.value.copy(cameraFps = fps)
    }

    fun setExpectedLabel(label: String?) {
        _expectedLabel.value = label
    }

    fun resetMetrics() {
        resetMetrics(keepThreshold = true)
    }

    private fun resetMetrics(keepThreshold: Boolean) {
        val threshold = if (keepThreshold) {
            _metricsState.value.confidenceThreshold
        } else {
            ModelConfig.CONFIDENCE_THRESHOLD
        }
        totalInferenceTimeMs = 0.0
        fpsWindowStartNanos = SystemClock.elapsedRealtimeNanos()
        fpsWindowFrames = 0
        _metricsState.value = MetricsUiState(confidenceThreshold = threshold)
        _inferenceLogs.value = emptyList()
    }

    fun setConfidenceThreshold(value: Float) {
        _metricsState.value = _metricsState.value.copy(confidenceThreshold = value.coerceIn(0f, 1f))
    }

    fun clearClassification() {
        _classificationState.value = ClassificationUiState()
    }

    fun showError(message: String) {
        Log.e(TAG, message)
        _classificationState.value = ClassificationUiState(errorMessage = message)
    }

    private fun runClassification(bitmap: Bitmap, source: InferenceSource) {
        try {
            val result = shrimpClassifier.classify(bitmap)
            _classificationState.value = ClassificationUiState(result = result)
            updateMetricsAndLogs(result, source)
        } catch (error: Throwable) {
            val message = error.message ?: "Inference failed."
            Log.e(TAG, message, error)
            _classificationState.value = ClassificationUiState(errorMessage = message)
        }
    }

    private fun updateMetricsAndLogs(result: ClassificationResult, source: InferenceSource) {
        val current = _metricsState.value
        val threshold = current.confidenceThreshold
        val expected = _expectedLabel.value
        val correct = expected?.let { it == result.top1.label }
        val accepted = isAccepted(result, expected, threshold)
        val totalInferences = current.totalInferences + 1
        val evaluatedInferences = current.evaluatedInferences + if (correct != null) 1 else 0
        val correctInferences = current.correctInferences + if (correct == true) 1 else 0

        totalInferenceTimeMs += result.inferenceTimeMs
        val averageInferenceTimeMs = totalInferenceTimeMs / totalInferences
        val speedIps = if (result.inferenceTimeMs > 0.0) 1000.0 / result.inferenceTimeMs else null
        val averageSpeedIps = if (averageInferenceTimeMs > 0.0) {
            1000.0 / averageInferenceTimeMs
        } else {
            null
        }

        val updatedClassStats = current.perClassPredictionStats.toMutableMap()
        if (expected != null) {
            val previousStats = updatedClassStats[expected] ?: ClassPredictionStats()
            updatedClassStats[expected] = previousStats.copy(
                total = previousStats.total + 1,
                incorrect = previousStats.incorrect + if (correct == true) 0 else 1,
            )
        }

        val updatedMetricsState = current.copy(
            totalInferences = totalInferences,
            acceptedInferences = current.acceptedInferences + if (accepted) 1 else 0,
            rejectedInferences = current.rejectedInferences + if (accepted) 0 else 1,
            evaluatedInferences = evaluatedInferences,
            correctInferences = correctInferences,
            lastInferenceTimeMs = result.inferenceTimeMs,
            averageInferenceTimeMs = averageInferenceTimeMs,
            speedIps = speedIps,
            averageSpeedIps = averageSpeedIps,
            perClassPredictionStats = updatedClassStats.toMap(),
        )

        _metricsState.value = updatedMetricsState

        val entry = InferenceLogEntry(
            timestampMs = SystemClock.elapsedRealtime(),
            source = source.displayName,
            label = result.top1.label,
            confidence = result.top1.confidence,
            inferenceTimeMs = result.inferenceTimeMs,
            threshold = threshold,
            accepted = accepted,
            expectedLabel = expected,
            correct = correct,
            metricsSnapshot = updatedMetricsState,
        )
        _inferenceLogs.value = (listOf(entry) + _inferenceLogs.value).take(MAX_LOG_ENTRIES)

        Log.i(
            TAG,
            String.format(
                Locale.US,
                "source=%s label=%s confidence=%.4f inferenceMs=%.2f threshold=%.2f accepted=%s expected=%s correct=%s",
                entry.source,
                entry.label,
                entry.confidence,
                entry.inferenceTimeMs,
                entry.threshold,
                entry.accepted,
                entry.expectedLabel ?: "N/A",
                entry.correct?.toString() ?: "N/A",
            )
        )
    }

    private fun isAccepted(
        result: ClassificationResult,
        expectedLabel: String?,
        threshold: Float,
    ): Boolean {
        val passesThreshold = result.top1.confidence >= threshold
        return if (expectedLabel == null) {
            passesThreshold
        } else {
            result.top1.label == expectedLabel && passesThreshold
        }
    }

    private companion object {
        private const val TAG = "ShrimpMetrics"
        private const val MAX_LOG_ENTRIES = 20
        private const val FPS_WINDOW_NANOS = 1_000_000_000L
    }
}
