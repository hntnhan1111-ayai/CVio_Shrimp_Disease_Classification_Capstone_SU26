package rs.smobile.shrimpdisease

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import rs.smobile.shrimpdisease.auth.AuthRepository
import rs.smobile.shrimpdisease.auth.AuthResult
import rs.smobile.shrimpdisease.auth.AuthRole
import rs.smobile.shrimpdisease.classifier.ClassificationResult
import rs.smobile.shrimpdisease.classifier.ModelDefaults
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.classifier.ShrimpClassifier
import rs.smobile.shrimpdisease.data.AdminDashboardRepository
import rs.smobile.shrimpdisease.data.AdminDashboardUiState
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.data.HistoryFilter
import rs.smobile.shrimpdisease.data.HistoryUiState
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.data.PredictionLogRepository
import rs.smobile.shrimpdisease.profile.FarmerProfileRepository
import rs.smobile.shrimpdisease.profile.FarmerProfileUiState
import rs.smobile.shrimpdisease.profile.FarmerProfileUpdate
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import javax.inject.Inject

enum class InferenceSource(val displayName: String) {
    GALLERY("Gallery image"),
    SNAPSHOT("Camera snapshot"),
}

enum class RuntimeDelegate(val displayName: String) {
    CPU("CPU"),
    GPU("GPU"),
}

data class InferenceInputUiState(
    val imageUri: Uri? = null,
    val bitmap: Bitmap? = null,
    val source: InferenceSource? = null,
    val isCameraActive: Boolean = false,
)

data class ClassificationUiState(
    val result: ClassificationResult? = null,
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
)

data class SettingsUiState(
    val runtimeDelegate: RuntimeDelegate = RuntimeDelegate.CPU,
    val confidenceThreshold: Float = ModelDefaults.CONFIDENCE_THRESHOLD,
    val showDebugInfo: Boolean = false,
    val cameraFps: Double? = null,
)

@HiltViewModel
class MainViewModel @Inject constructor(
    private val shrimpClassifier: ShrimpClassifier,
    private val predictionLogRepository: PredictionLogRepository,
    private val authRepository: AuthRepository,
    private val farmerProfileRepository: FarmerProfileRepository,
    private val adminDashboardRepository: AdminDashboardRepository,
) : ViewModel() {

    private val _inputState = MutableStateFlow(InferenceInputUiState())
    val inputState: StateFlow<InferenceInputUiState> = _inputState

    private val _classificationState = MutableStateFlow(ClassificationUiState())
    val classificationState: StateFlow<ClassificationUiState> = _classificationState

    private val _settingsState = MutableStateFlow(SettingsUiState())
    val settingsState: StateFlow<SettingsUiState> = _settingsState

    private val _modelInfo = MutableStateFlow(shrimpClassifier.modelInfo)
    val modelInfo: StateFlow<ModelInfo> = _modelInfo

    private val _labels = MutableStateFlow(shrimpClassifier.labels)
    val labels: StateFlow<List<String>> = _labels

    private val _availableModels = MutableStateFlow(shrimpClassifier.availableModelsInAssets())
    val availableModels: StateFlow<List<String>> = _availableModels

    private val _selectedGroundTruthLabel = MutableStateFlow<String?>(null)
    val selectedGroundTruthLabel: StateFlow<String?> = _selectedGroundTruthLabel

    private val _adminDashboardUiState = MutableStateFlow(AdminDashboardUiState())
    val adminDashboardUiState: StateFlow<AdminDashboardUiState> = _adminDashboardUiState

    val logs: StateFlow<List<PredictionLogItem>> = predictionLogRepository.logs
    val benchmarkMetrics: StateFlow<BenchmarkMetrics> = predictionLogRepository.benchmarkMetrics
    val historyUiState: StateFlow<HistoryUiState> = predictionLogRepository.historyUiState
    val farmerProfileUiState: StateFlow<FarmerProfileUiState> = farmerProfileRepository.profile
    val authSession = authRepository.session

    private var fpsWindowStartNanos = SystemClock.elapsedRealtimeNanos()
    private var fpsWindowFrames = 0
    private val loggedResultTimestamps = mutableSetOf<Long>()

    init {
        predictionLogRepository.setOwner(authRepository.session.value.user?.id)
        farmerProfileRepository.setUser(authRepository.session.value.user)
        refreshAdminDashboard()
        viewModelScope.launch {
            authRepository.session.collect { session ->
                loggedResultTimestamps.clear()
                predictionLogRepository.setOwner(session.user?.id)
                farmerProfileRepository.setUser(session.user)
                refreshAdminDashboard()
            }
        }
    }

    fun setGalleryImage(uri: Uri, bitmap: Bitmap) {
        _inputState.value = InferenceInputUiState(
            imageUri = uri,
            bitmap = bitmap,
            source = InferenceSource.GALLERY,
            isCameraActive = false,
        )
        clearClassification()
    }

    fun startCameraInput() {
        _inputState.value = InferenceInputUiState(
            source = InferenceSource.SNAPSHOT,
            isCameraActive = true,
        )
        clearClassification()
    }

    fun setCameraSnapshot(bitmap: Bitmap) {
        _inputState.value = InferenceInputUiState(
            bitmap = bitmap,
            source = InferenceSource.SNAPSHOT,
            isCameraActive = true,
        )
        clearClassification()
    }

    fun clearInput() {
        _inputState.value = InferenceInputUiState()
        clearClassification()
    }

    fun clearClassification() {
        _classificationState.value = ClassificationUiState()
    }

    fun showError(message: String) {
        Log.e(TAG, message)
        _classificationState.value = ClassificationUiState(errorMessage = message)
    }

    fun runInference() {
        val bitmap = _inputState.value.bitmap
        if (bitmap == null) {
            showError("Choose an image or take a camera snapshot before running inference.")
            return
        }

        _classificationState.value = ClassificationUiState(isLoading = true)
        viewModelScope.launch(Dispatchers.Default) {
            runCatching {
                shrimpClassifier.classify(
                    bitmap = bitmap,
                    threshold = _settingsState.value.confidenceThreshold,
                    groundTruthLabel = _selectedGroundTruthLabel.value,
                )
            }
                .onSuccess { result ->
                    _classificationState.value = ClassificationUiState(result = result)
                }
                .onFailure { error ->
                    val message = error.message ?: "Inference failed."
                    Log.e(TAG, message, error)
                    _classificationState.value = ClassificationUiState(errorMessage = message)
                }
        }
    }

    fun saveCurrentResult(): Boolean {
        val result = _classificationState.value.result ?: return false
        return addResultToLog(result)
    }

    fun setGroundTruthLabel(label: String?) {
        _selectedGroundTruthLabel.value = label
    }

    fun setConfidenceThreshold(value: Float) {
        _settingsState.value = _settingsState.value.copy(
            confidenceThreshold = value.coerceIn(0f, 1f),
        )
    }

    fun clearLogs() {
        predictionLogRepository.clearLogs()
        loggedResultTimestamps.clear()
        refreshAdminDashboard()
    }

    fun setHistoryFilter(filter: HistoryFilter) {
        predictionLogRepository.setHistoryFilter(filter)
    }

    fun updateFarmerProfile(update: FarmerProfileUpdate): Boolean {
        val updated = farmerProfileRepository.updateProfile(update)
        if (updated) {
            authRepository.updateDisplayName(farmerProfileRepository.profile.value.displayName)
            refreshAdminDashboard()
        }
        return updated
    }

    fun setFarmerDataPermission(enabled: Boolean): Boolean {
        val updated = farmerProfileRepository.setDataPermissionEnabled(enabled)
        if (updated) refreshAdminDashboard()
        return updated
    }

    fun exportLogs(context: Context, uri: Uri): Boolean {
        return predictionLogRepository.exportCsv(context, uri)
    }

    fun login(
        role: AuthRole,
        account: String,
        password: String,
    ): AuthResult {
        return authRepository.login(role, account, password).also { refreshAdminDashboard() }
    }

    fun register(
        role: AuthRole,
        account: String,
        password: String,
    ): AuthResult {
        return authRepository.register(role, account, password).also { refreshAdminDashboard() }
    }

    fun logout() {
        authRepository.logout()
        clearInput()
        refreshAdminDashboard()
    }

    fun refreshAdminDashboard() {
        _adminDashboardUiState.value = adminDashboardRepository.loadDashboard()
    }

    fun markAdminDataReviewed(itemId: String): Boolean {
        val updated = adminDashboardRepository.markReviewed(itemId)
        refreshAdminDashboard()
        return updated
    }

    fun excludeAdminDataFromTraining(itemId: String): Boolean {
        val updated = adminDashboardRepository.excludeFromTraining(itemId)
        refreshAdminDashboard()
        return updated
    }

    fun exportAdminMetadata(context: Context, uri: Uri): Boolean {
        return adminDashboardRepository.exportMetadata(context, uri)
    }

    fun loadModel(modelFile: String) {
        viewModelScope.launch(Dispatchers.Default) {
            _classificationState.value = ClassificationUiState(isLoading = true)
            runCatching {
                val family = shrimpClassifier.modelFamilyForAsset(modelFile)
                shrimpClassifier.loadModelFromAssets(modelFile, ModelDefaults.LABEL_FILE, family)
            }.onSuccess {
                _modelInfo.value = shrimpClassifier.modelInfo
                _labels.value = shrimpClassifier.labels
                _availableModels.value = shrimpClassifier.availableModelsInAssets()
                clearClassification()
            }.onFailure { error ->
                val message = "Failed to load model: ${error.message}"
                Log.e(TAG, message, error)
                _classificationState.value = ClassificationUiState(errorMessage = message)
            }
        }
    }

    fun setRuntimeDelegate(delegate: RuntimeDelegate) {
        if (delegate == RuntimeDelegate.GPU) {
            showError("GPU delegate is not bundled in this build. CPU runtime remains active.")
            _settingsState.value = _settingsState.value.copy(runtimeDelegate = RuntimeDelegate.CPU)
            return
        }
        _settingsState.value = _settingsState.value.copy(runtimeDelegate = delegate)
    }

    fun setShowDebugInfo(enabled: Boolean) {
        shrimpClassifier.setPreprocessDebugEnabled(enabled)
        _settingsState.value = _settingsState.value.copy(showDebugInfo = enabled)
    }

    fun recordCameraFrame() {
        fpsWindowFrames += 1
        val now = SystemClock.elapsedRealtimeNanos()
        val elapsedNanos = now - fpsWindowStartNanos
        if (elapsedNanos < FPS_WINDOW_NANOS) return

        val fps = fpsWindowFrames * 1_000_000_000.0 / elapsedNanos
        fpsWindowFrames = 0
        fpsWindowStartNanos = now
        _settingsState.value = _settingsState.value.copy(cameraFps = fps)
    }

    private fun addResultToLog(result: ClassificationResult): Boolean {
        if (!loggedResultTimestamps.add(result.timestamp)) return false

        val input = _inputState.value
        val saved = predictionLogRepository.addLog(
            PredictionLogItem(
                imageUri = input.imageUri?.toString(),
                predictedClass = result.predictedClass,
                confidence = result.confidence,
                top3Predictions = BenchmarkUtils.top3CsvText(result.top3Predictions),
                inferenceTimeMs = result.inferenceTimeMs,
                speed = result.speed,
                fps = result.fps,
                modelName = result.modelName,
                threshold = result.threshold,
                isAboveThreshold = result.isAboveThreshold,
                groundTruthLabel = result.groundTruthLabel,
                isCorrect = result.isCorrect,
                timestamp = result.timestamp,
                thumbnail = input.bitmap.takeIf { input.imageUri == null },
            )
        )
        if (!saved) {
            loggedResultTimestamps.remove(result.timestamp)
        } else {
            refreshAdminDashboard()
        }
        return saved
    }

    private companion object {
        private const val TAG = "ShrimpDisease"
        private const val FPS_WINDOW_NANOS = 1_000_000_000L
    }
}
