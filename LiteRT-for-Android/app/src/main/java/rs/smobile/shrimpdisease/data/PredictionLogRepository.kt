package rs.smobile.shrimpdisease.data

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import rs.smobile.shrimpdisease.utils.CsvExportUtils
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class PredictionLogRepository @Inject constructor() {
    private val _logs = MutableStateFlow<List<PredictionLogItem>>(emptyList())
    val logs: StateFlow<List<PredictionLogItem>> = _logs

    private val _benchmarkMetrics = MutableStateFlow(emptyMetrics())
    val benchmarkMetrics: StateFlow<BenchmarkMetrics> = _benchmarkMetrics

    fun addLog(logItem: PredictionLogItem) {
        val existingLogs = _logs.value
        val updatedMetrics = getBenchmarkMetrics(existingLogs + logItem)
        val enrichedLog = logItem.copy(
            averageInferenceTimeMs = updatedMetrics.averageInferenceTimeMs,
            averageSpeed = updatedMetrics.averageSpeed,
            averageFps = updatedMetrics.averageFps,
            runningAccuracy = updatedMetrics.accuracy,
        )

        _logs.value = (listOf(enrichedLog) + existingLogs).take(MAX_LOG_ITEMS)
        _benchmarkMetrics.value = getBenchmarkMetrics(_logs.value)
    }

    fun getAllLogs(): List<PredictionLogItem> = _logs.value

    fun clearLogs() {
        _logs.value = emptyList()
        _benchmarkMetrics.value = emptyMetrics()
    }

    fun getBenchmarkMetrics(): BenchmarkMetrics {
        return _benchmarkMetrics.value
    }

    fun exportCsv(context: Context, uri: Uri): Boolean {
        return CsvExportUtils.exportLogsToUri(context, uri, _logs.value)
    }

    private fun getBenchmarkMetrics(logs: List<PredictionLogItem>): BenchmarkMetrics {
        val evaluatedLogs = logs.filter { it.isCorrect != null }
        return BenchmarkMetrics(
            totalRuns = logs.size,
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

    private companion object {
        private const val MAX_LOG_ITEMS = 200

        private fun emptyMetrics(): BenchmarkMetrics {
            return BenchmarkMetrics(
                totalRuns = 0,
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
