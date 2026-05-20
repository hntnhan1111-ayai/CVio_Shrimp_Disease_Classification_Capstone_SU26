package rs.smobile.shrimpdisease.data

import android.content.Context
import android.net.Uri
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import org.json.JSONObject
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import rs.smobile.shrimpdisease.utils.CsvExportUtils
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class PredictionLogRepository @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    private val _logs = MutableStateFlow<List<PredictionLogItem>>(emptyList())
    val logs: StateFlow<List<PredictionLogItem>> = _logs

    private val _benchmarkMetrics = MutableStateFlow(emptyMetrics())
    val benchmarkMetrics: StateFlow<BenchmarkMetrics> = _benchmarkMetrics

    private val _historyUiState = MutableStateFlow(HistoryUiState())
    val historyUiState: StateFlow<HistoryUiState> = _historyUiState

    private var ownerId: String? = null
    private var selectedFilter = HistoryFilter.All

    fun setOwner(ownerId: String?) {
        if (this.ownerId == ownerId) return
        this.ownerId = ownerId
        selectedFilter = HistoryFilter.All
        refreshFromStorage()
    }

    fun setHistoryFilter(filter: HistoryFilter) {
        if (selectedFilter == filter) return
        selectedFilter = filter
        publish(_logs.value)
    }

    fun addLog(logItem: PredictionLogItem): Boolean {
        val owner = ownerId ?: return false
        val existingLogs = readLogs(owner)
        if (existingLogs.any { it.id == logItem.id }) return false

        val updatedMetrics = getBenchmarkMetrics(existingLogs + logItem)
        val enrichedLog = logItem.copy(
            averageInferenceTimeMs = updatedMetrics.averageInferenceTimeMs,
            averageSpeed = updatedMetrics.averageSpeed,
            averageFps = updatedMetrics.averageFps,
            runningAccuracy = updatedMetrics.accuracy,
        )
        val updatedLogs = (listOf(enrichedLog) + existingLogs).take(MAX_LOG_ITEMS)
        writeLogs(owner, updatedLogs)
        publish(updatedLogs)
        return true
    }

    fun getAllLogs(): List<PredictionLogItem> = _logs.value

    fun getLogsForOwner(ownerId: String): List<PredictionLogItem> {
        return readLogs(ownerId)
    }

    fun clearLogs() {
        val owner = ownerId ?: return publish(emptyList())
        preferences.edit()
            .remove(keyFor(owner))
            .apply()
        publish(emptyList())
    }

    fun getBenchmarkMetrics(): BenchmarkMetrics {
        return _benchmarkMetrics.value
    }

    fun exportCsv(context: Context, uri: Uri): Boolean {
        return CsvExportUtils.exportLogsToUri(context, uri, _logs.value)
    }

    private fun refreshFromStorage() {
        val owner = ownerId
        publish(if (owner == null) emptyList() else readLogs(owner))
    }

    private fun publish(logs: List<PredictionLogItem>) {
        val metrics = getBenchmarkMetrics(logs)
        _logs.value = logs
        _benchmarkMetrics.value = metrics
        _historyUiState.value = HistoryUiState(
            logs = logs,
            filteredLogs = logs.filter { selectedFilter.matches(it) },
            selectedFilter = selectedFilter,
            filterCounts = getFilterCounts(logs),
            benchmarkMetrics = metrics,
        )
    }

    private fun getFilterCounts(logs: List<PredictionLogItem>): HistoryFilterCounts {
        var healthy = 0
        var disease = 0
        var lowConfidence = 0

        logs.forEach { item ->
            when (DiagnosisHistoryStatus.from(item)) {
                DiagnosisHistoryStatus.Healthy -> healthy += 1
                DiagnosisHistoryStatus.DiseaseDetected -> disease += 1
                DiagnosisHistoryStatus.LowConfidence -> lowConfidence += 1
            }
        }

        return HistoryFilterCounts(
            all = logs.size,
            healthy = healthy,
            disease = disease,
            lowConfidence = lowConfidence,
        )
    }

    private fun readLogs(owner: String): List<PredictionLogItem> {
        val json = preferences.getString(keyFor(owner), null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(json)
            List(array.length()) { index ->
                array.getJSONObject(index).toPredictionLogItem()
            }
        }.getOrElse { emptyList() }
    }

    private fun writeLogs(owner: String, logs: List<PredictionLogItem>) {
        val array = JSONArray()
        logs.forEach { item -> array.put(item.toJson()) }
        preferences.edit()
            .putString(keyFor(owner), array.toString())
            .apply()
    }

    private fun keyFor(owner: String): String {
        return "$KEY_LOGS_PREFIX$owner"
    }

    private fun getBenchmarkMetrics(logs: List<PredictionLogItem>): BenchmarkMetrics {
        val evaluatedLogs = logs.filter { it.isCorrect != null }
        val weekStart = System.currentTimeMillis() - ONE_WEEK_MS
        return BenchmarkMetrics(
            totalRuns = logs.size,
            weeklyRuns = logs.count { it.timestamp >= weekStart },
            evaluatedRuns = evaluatedLogs.size,
            correctRuns = evaluatedLogs.count { it.isCorrect == true },
            averageInferenceTimeMs = BenchmarkUtils.calculateAverageTime(logs.map { it.inferenceTimeMs }),
            averageSpeed = BenchmarkUtils.calculateAverageSpeed(logs.map { it.speed }),
            averageFps = BenchmarkUtils.calculateAverageFps(logs.map { it.fps }),
            accuracy = BenchmarkUtils.calculateAccuracy(
                correctRuns = evaluatedLogs.count { it.isCorrect == true },
                totalRuns = evaluatedLogs.size,
            ),
        )
    }

    private fun PredictionLogItem.toJson(): JSONObject {
        return JSONObject()
            .put("imageUri", imageUri)
            .put("predictedClass", predictedClass)
            .put("confidence", confidence.toDouble())
            .put("top3Predictions", top3Predictions)
            .put("inferenceTimeMs", inferenceTimeMs)
            .put("speed", speed.toDouble())
            .put("fps", fps.toDouble())
            .put("modelName", modelName)
            .put("threshold", threshold.toDouble())
            .put("isAboveThreshold", isAboveThreshold)
            .put("groundTruthLabel", groundTruthLabel)
            .put("isCorrect", isCorrect)
            .put("timestamp", timestamp)
            .put("averageInferenceTimeMs", averageInferenceTimeMs.toDouble())
            .put("averageSpeed", averageSpeed.toDouble())
            .put("averageFps", averageFps.toDouble())
            .put("runningAccuracy", runningAccuracy?.toDouble())
    }

    private fun JSONObject.toPredictionLogItem(): PredictionLogItem {
        return PredictionLogItem(
            imageUri = optString("imageUri").takeIf { it.isNotBlank() && it != "null" },
            predictedClass = getString("predictedClass"),
            confidence = getDouble("confidence").toFloat(),
            top3Predictions = getString("top3Predictions"),
            inferenceTimeMs = getLong("inferenceTimeMs"),
            speed = getDouble("speed").toFloat(),
            fps = getDouble("fps").toFloat(),
            modelName = getString("modelName"),
            threshold = getDouble("threshold").toFloat(),
            isAboveThreshold = getBoolean("isAboveThreshold"),
            groundTruthLabel = optString("groundTruthLabel").takeIf { it.isNotBlank() && it != "null" },
            isCorrect = when {
                isNull("isCorrect") -> null
                else -> getBoolean("isCorrect")
            },
            timestamp = getLong("timestamp"),
            thumbnail = null,
            averageInferenceTimeMs = optDouble("averageInferenceTimeMs", 0.0).toFloat(),
            averageSpeed = optDouble("averageSpeed", 0.0).toFloat(),
            averageFps = optDouble("averageFps", 0.0).toFloat(),
            runningAccuracy = when {
                isNull("runningAccuracy") -> null
                else -> getDouble("runningAccuracy").toFloat()
            },
        )
    }

    private companion object {
        private const val PREFERENCES_NAME = "aquapulse_prediction_logs"
        private const val KEY_LOGS_PREFIX = "logs_"
        private const val MAX_LOG_ITEMS = 200
        private const val ONE_WEEK_MS = 7L * 24L * 60L * 60L * 1000L

        private fun emptyMetrics(): BenchmarkMetrics {
            return BenchmarkMetrics(
                totalRuns = 0,
                weeklyRuns = 0,
                evaluatedRuns = 0,
                correctRuns = 0,
                averageInferenceTimeMs = 0f,
                averageSpeed = 0f,
                averageFps = 0f,
                accuracy = null,
            )
        }
    }
}
