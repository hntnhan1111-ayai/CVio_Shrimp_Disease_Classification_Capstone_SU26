package rs.smobile.shrimpdisease.data

enum class DiagnosisHistoryStatus {
    Healthy,
    DiseaseDetected,
    LowConfidence;

    companion object {
        fun from(item: PredictionLogItem): DiagnosisHistoryStatus {
            return when {
                !item.isAboveThreshold -> LowConfidence
                item.predictedClass.contains("healthy", ignoreCase = true) -> Healthy
                else -> DiseaseDetected
            }
        }
    }
}

enum class HistoryFilter(val label: String) {
    All("Tất cả"),
    Healthy("Tôm khỏe"),
    Disease("Có dấu hiệu bệnh"),
    LowConfidence("Ảnh chưa rõ");

    fun matches(item: PredictionLogItem): Boolean {
        return when (this) {
            All -> true
            Healthy -> DiagnosisHistoryStatus.from(item) == DiagnosisHistoryStatus.Healthy
            Disease -> DiagnosisHistoryStatus.from(item) == DiagnosisHistoryStatus.DiseaseDetected
            LowConfidence -> DiagnosisHistoryStatus.from(item) == DiagnosisHistoryStatus.LowConfidence
        }
    }
}

data class HistoryFilterCounts(
    val all: Int = 0,
    val healthy: Int = 0,
    val disease: Int = 0,
    val lowConfidence: Int = 0,
) {
    fun countFor(filter: HistoryFilter): Int {
        return when (filter) {
            HistoryFilter.All -> all
            HistoryFilter.Healthy -> healthy
            HistoryFilter.Disease -> disease
            HistoryFilter.LowConfidence -> lowConfidence
        }
    }
}

data class HistoryUiState(
    val logs: List<PredictionLogItem> = emptyList(),
    val filteredLogs: List<PredictionLogItem> = emptyList(),
    val selectedFilter: HistoryFilter = HistoryFilter.All,
    val filterCounts: HistoryFilterCounts = HistoryFilterCounts(),
    val benchmarkMetrics: BenchmarkMetrics = BenchmarkMetrics(
        totalRuns = 0,
        weeklyRuns = 0,
        evaluatedRuns = 0,
        correctRuns = 0,
        averageInferenceTimeMs = 0f,
        averageSpeed = 0f,
        averageFps = 0f,
        accuracy = null,
    ),
)
