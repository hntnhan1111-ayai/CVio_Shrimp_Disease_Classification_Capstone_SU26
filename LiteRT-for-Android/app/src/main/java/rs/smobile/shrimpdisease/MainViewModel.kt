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
import rs.smobile.shrimpdisease.data.AdminCreateUserInput
import rs.smobile.shrimpdisease.data.AdminDataMutationInput
import rs.smobile.shrimpdisease.data.AdminDashboardRepository
import rs.smobile.shrimpdisease.data.AdminDashboardUiState
import rs.smobile.shrimpdisease.data.AdminDiagnosisUiState
import rs.smobile.shrimpdisease.data.AdminInferenceLogsUiState
import rs.smobile.shrimpdisease.data.AdminModelConfigUiState
import rs.smobile.shrimpdisease.data.AdminModelConfigUpdate
import rs.smobile.shrimpdisease.data.AdminUpdateUserInput
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
    GALLERY("Ảnh có sẵn"),
    SNAPSHOT("Ảnh chụp máy ảnh"),
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

    private val _adminDiagnosisUiState = MutableStateFlow(AdminDiagnosisUiState())
    val adminDiagnosisUiState: StateFlow<AdminDiagnosisUiState> = _adminDiagnosisUiState

    private val _adminModelConfigUiState = MutableStateFlow(AdminModelConfigUiState())
    val adminModelConfigUiState: StateFlow<AdminModelConfigUiState> = _adminModelConfigUiState

    private val _adminInferenceLogsUiState = MutableStateFlow(AdminInferenceLogsUiState())
    val adminInferenceLogsUiState: StateFlow<AdminInferenceLogsUiState> = _adminInferenceLogsUiState

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
        _selectedGroundTruthLabel.value = null
        _inputState.value = InferenceInputUiState(
            imageUri = uri,
            bitmap = bitmap,
            source = InferenceSource.GALLERY,
            isCameraActive = false,
        )
        clearClassification()
    }

    fun startCameraInput() {
        _selectedGroundTruthLabel.value = null
        _inputState.value = InferenceInputUiState(
            source = InferenceSource.SNAPSHOT,
            isCameraActive = true,
        )
        clearClassification()
    }

    fun setCameraSnapshot(bitmap: Bitmap) {
        _selectedGroundTruthLabel.value = null
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
            showError("Hãy chọn ảnh hoặc chụp ảnh tôm trước khi kiểm tra.")
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
                    val message = error.message ?: "AI kiểm tra chưa thành công."
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

    fun updateFarmerAvatar(avatarUri: String?): Boolean {
        val updated = farmerProfileRepository.setAvatarUri(avatarUri)
        if (updated) refreshAdminDashboard()
        return updated
    }

    fun exportLogs(context: Context, uri: Uri): Boolean {
        return predictionLogRepository.exportCsv(context, uri)
    }

    fun login(
        account: String,
        password: String,
    ): AuthResult {
        return authRepository.login(account, password).also { refreshAdminDashboard() }
    }

    fun register(
        account: String,
        password: String,
    ): AuthResult {
        return authRepository.register(account, password).also { refreshAdminDashboard() }
    }

    fun logout() {
        authRepository.logout()
        clearInput()
        refreshAdminDashboard()
    }

    fun refreshAdminDashboard() {
        _adminDashboardUiState.value = adminDashboardRepository.loadDashboard()
        _adminDiagnosisUiState.value = adminDashboardRepository.loadDiagnosis()
        refreshAdminSettings()
    }

    fun refreshAdminSettings() {
        _adminModelConfigUiState.value = adminDashboardRepository.loadModelConfig(
            activeModelFile = _modelInfo.value.modelFile,
            availableModels = _availableModels.value,
            threshold = _settingsState.value.confidenceThreshold,
        )
        _adminInferenceLogsUiState.value = adminDashboardRepository.loadInferenceLogs()
    }

    fun saveAdminModelConfig(update: AdminModelConfigUpdate): Boolean {
        adminDashboardRepository.saveModelConfig(update)
        setConfidenceThreshold(update.threshold)
        refreshAdminSettings()
        return true
    }

    fun deployAdminModel(modelFile: String) {
        adminDashboardRepository.saveActiveModel(modelFile)
        loadModel(modelFile)
        refreshAdminSettings()
    }

    fun createAdminUser(input: AdminCreateUserInput): AuthResult {
        val result = adminDashboardRepository.createUser(input)
        if (result is AuthResult.Success) refreshAdminDashboard()
        return result
    }

    fun updateAdminUser(
        userId: String,
        input: AdminUpdateUserInput,
    ): AuthResult {
        val result = adminDashboardRepository.updateUser(userId, input)
        if (result is AuthResult.Success) refreshAdminDashboard()
        return result
    }

    fun deleteAdminUser(userId: String): Boolean {
        val deleted = adminDashboardRepository.deleteUser(userId)
        if (deleted) refreshAdminDashboard()
        return deleted
    }

    fun createAdminData(input: AdminDataMutationInput): Boolean {
        val created = adminDashboardRepository.createDataItem(input)
        if (created) refreshAdminDashboard()
        return created
    }

    fun updateAdminData(
        itemId: String,
        input: AdminDataMutationInput,
    ): Boolean {
        val updated = adminDashboardRepository.updateDataItem(itemId, input)
        if (updated) refreshAdminDashboard()
        return updated
    }

    fun deleteAdminData(itemId: String): Boolean {
        val deleted = adminDashboardRepository.deleteDataItem(itemId)
        if (deleted) refreshAdminDashboard()
        return deleted
    }

    fun markAdminDataReviewed(itemId: String): Boolean {
        val updated = adminDashboardRepository.markReviewed(itemId)
        refreshAdminDashboard()
        return updated
    }

    fun confirmAdminDiagnosis(itemId: String): Boolean {
        val updated = adminDashboardRepository.confirmDiagnosis(itemId)
        refreshAdminDashboard()
        return updated
    }

    fun correctAdminDiagnosis(
        itemId: String,
        correctedLabel: String,
    ): Boolean {
        val updated = adminDashboardRepository.correctDiagnosis(itemId, correctedLabel)
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
                refreshAdminSettings()
            }.onFailure { error ->
                val message = "Không tải được mô hình: ${error.message}"
                Log.e(TAG, message, error)
                _classificationState.value = ClassificationUiState(errorMessage = message)
            }
        }
    }

    fun setRuntimeDelegate(delegate: RuntimeDelegate) {
        if (delegate == RuntimeDelegate.GPU) {
            showError("Bản này chưa hỗ trợ GPU. Ứng dụng sẽ tiếp tục chạy bằng CPU.")
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
