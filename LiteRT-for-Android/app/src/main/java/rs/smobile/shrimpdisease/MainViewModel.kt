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
import kotlinx.coroutines.withContext
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
        viewModelScope.launch {
            authRepository.session.collect { session ->
                loggedResultTimestamps.clear()
                withContext(Dispatchers.IO) {
                    predictionLogRepository.setOwner(session.user?.id)
                    farmerProfileRepository.setUser(session.user)
                }
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

    fun saveCurrentResult(onResult: (saved: Boolean, hasResult: Boolean) -> Unit) {
        val result = _classificationState.value.result
        if (result == null) {
            onResult(false, false)
            return
        }
        viewModelScope.launch {
            val saved = addResultToLog(result)
            onResult(saved, true)
        }
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
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                predictionLogRepository.clearLogs()
            }
            loggedResultTimestamps.clear()
            refreshAdminDashboard()
        }
    }

    fun setHistoryFilter(filter: HistoryFilter) {
        predictionLogRepository.setHistoryFilter(filter)
    }

    fun updateFarmerProfile(
        update: FarmerProfileUpdate,
        onResult: (Boolean) -> Unit,
    ) {
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) {
                val saved = farmerProfileRepository.updateProfile(update)
                if (saved) {
                    authRepository.updateDisplayName(farmerProfileRepository.profile.value.displayName)
                }
                saved
            }
            if (updated) refreshAdminDashboard()
            onResult(updated)
        }
    }

    fun setFarmerDataPermission(enabled: Boolean) {
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) {
                farmerProfileRepository.setDataPermissionEnabled(enabled)
            }
            if (updated) refreshAdminDashboard()
        }
    }

    fun updateFarmerAvatar(
        avatarUri: String?,
        onResult: (Boolean) -> Unit,
    ) {
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) {
                farmerProfileRepository.setAvatarUri(avatarUri)
            }
            if (updated) refreshAdminDashboard()
            onResult(updated)
        }
    }

    fun exportLogs(context: Context, uri: Uri): Boolean {
        return predictionLogRepository.exportCsv(context, uri)
    }

    fun login(
        account: String,
        password: String,
        onResult: (AuthResult) -> Unit,
    ) {
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                authRepository.login(account, password)
            }
            refreshAdminDashboard()
            onResult(result)
        }
    }

    fun register(
        account: String,
        password: String,
        onResult: (AuthResult) -> Unit,
    ) {
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                authRepository.register(account, password)
            }
            refreshAdminDashboard()
            onResult(result)
        }
    }

    fun logout() {
        authRepository.logout()
        clearInput()
        refreshAdminDashboard()
    }

    fun refreshAdminDashboard() {
        val activeModelFile = _modelInfo.value.modelFile
        val availableModels = _availableModels.value
        val threshold = _settingsState.value.confidenceThreshold
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    AdminDashboardSnapshot(
                        dashboard = adminDashboardRepository.loadDashboard(),
                        diagnosis = adminDashboardRepository.loadDiagnosis(),
                        modelConfig = adminDashboardRepository.loadModelConfig(
                            activeModelFile = activeModelFile,
                            availableModels = availableModels,
                            threshold = threshold,
                        ),
                        inferenceLogs = adminDashboardRepository.loadInferenceLogs(),
                    )
                }
            }.onSuccess { snapshot ->
                _adminDashboardUiState.value = snapshot.dashboard
                _adminDiagnosisUiState.value = snapshot.diagnosis
                _adminModelConfigUiState.value = snapshot.modelConfig
                _adminInferenceLogsUiState.value = snapshot.inferenceLogs
            }.onFailure { error ->
                Log.e(TAG, "Khong tai duoc dashboard quan tri.", error)
            }
        }
    }

    fun refreshAdminSettings() {
        val activeModelFile = _modelInfo.value.modelFile
        val availableModels = _availableModels.value
        val threshold = _settingsState.value.confidenceThreshold
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    AdminSettingsSnapshot(
                        modelConfig = adminDashboardRepository.loadModelConfig(
                            activeModelFile = activeModelFile,
                            availableModels = availableModels,
                            threshold = threshold,
                        ),
                        inferenceLogs = adminDashboardRepository.loadInferenceLogs(),
                    )
                }
            }.onSuccess { snapshot ->
                _adminModelConfigUiState.value = snapshot.modelConfig
                _adminInferenceLogsUiState.value = snapshot.inferenceLogs
            }.onFailure { error ->
                Log.e(TAG, "Khong tai duoc cau hinh quan tri.", error)
            }
        }
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

    fun createAdminUser(
        input: AdminCreateUserInput,
        onResult: (AuthResult) -> Unit,
    ) {
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                adminDashboardRepository.createUser(input)
            }
            if (result is AuthResult.Success) refreshAdminDashboard()
            onResult(result)
        }
    }

    fun updateAdminUser(
        userId: String,
        input: AdminUpdateUserInput,
        onResult: (AuthResult) -> Unit,
    ) {
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                adminDashboardRepository.updateUser(userId, input)
            }
            if (result is AuthResult.Success) refreshAdminDashboard()
            onResult(result)
        }
    }

    fun deleteAdminUser(
        userId: String,
        onResult: (Boolean) -> Unit,
    ) {
        viewModelScope.launch {
            val deleted = withContext(Dispatchers.IO) {
                adminDashboardRepository.deleteUser(userId)
            }
            if (deleted) refreshAdminDashboard()
            onResult(deleted)
        }
    }

    fun createAdminData(
        input: AdminDataMutationInput,
        onResult: (Boolean) -> Unit,
    ) {
        viewModelScope.launch {
            val created = withContext(Dispatchers.IO) {
                adminDashboardRepository.createDataItem(input)
            }
            if (created) refreshAdminDashboard()
            onResult(created)
        }
    }

    fun updateAdminData(
        itemId: String,
        input: AdminDataMutationInput,
        onResult: (Boolean) -> Unit,
    ) {
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) {
                adminDashboardRepository.updateDataItem(itemId, input)
            }
            if (updated) refreshAdminDashboard()
            onResult(updated)
        }
    }

    fun deleteAdminData(
        itemId: String,
        onResult: (Boolean) -> Unit,
    ) {
        viewModelScope.launch {
            val deleted = withContext(Dispatchers.IO) {
                adminDashboardRepository.deleteDataItem(itemId)
            }
            if (deleted) refreshAdminDashboard()
            onResult(deleted)
        }
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

    private suspend fun addResultToLog(result: ClassificationResult): Boolean {
        if (!loggedResultTimestamps.add(result.timestamp)) return false

        val input = _inputState.value
        val logItem = PredictionLogItem(
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
        val saved = withContext(Dispatchers.IO) {
            predictionLogRepository.addLog(logItem)
        }
        if (!saved) {
            loggedResultTimestamps.remove(result.timestamp)
        } else {
            refreshAdminDashboard()
        }
        return saved
    }

    private data class AdminDashboardSnapshot(
        val dashboard: AdminDashboardUiState,
        val diagnosis: AdminDiagnosisUiState,
        val modelConfig: AdminModelConfigUiState,
        val inferenceLogs: AdminInferenceLogsUiState,
    )

    private data class AdminSettingsSnapshot(
        val modelConfig: AdminModelConfigUiState,
        val inferenceLogs: AdminInferenceLogsUiState,
    )

    private companion object {
        private const val TAG = "ShrimpDisease"
        private const val FPS_WINDOW_NANOS = 1_000_000_000L
    }
}
