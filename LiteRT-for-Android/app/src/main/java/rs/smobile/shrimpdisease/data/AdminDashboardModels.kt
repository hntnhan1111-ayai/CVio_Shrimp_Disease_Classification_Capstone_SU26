package rs.smobile.shrimpdisease.data

data class AdminDashboardUiState(
    val totalFarmers: Int = 0,
    val totalDiseaseChecks: Int = 0,
    val imagesProcessed: Int = 0,
    val activeAlerts: Int = 0,
    val farmersTrend: String = "+0%",
    val checksTrend: String = "+0%",
    val diagnosisVolume: List<AdminChartPoint> = emptyAdminChartPoints(),
    val recentActivities: List<AdminActivityItem> = emptyList(),
    val users: List<AdminUserSummary> = emptyList(),
    val dataItems: List<AdminDataReviewItem> = emptyList(),
)

data class AdminDiagnosisUiState(
    val overallAccuracyText: String = "N/A",
    val truePositiveRateText: String = "N/A",
    val falseNegativeRateText: String = "N/A",
    val accuracyTrendText: String = "+0.0% from last week",
    val totalScans: Int = 0,
    val pendingCount: Int = 0,
    val verifiedCount: Int = 0,
    val flaggedCount: Int = 0,
    val activeOutbreaks: Int = 0,
    val queueItems: List<AdminDiagnosisReviewItem> = emptyList(),
    val flaggedDiagnoses: List<AdminDiagnosisReviewItem> = emptyList(),
    val prevalence: List<AdminDiseasePrevalenceItem> = emptyList(),
    val outbreakTrend: List<AdminChartPoint> = emptyList(),
    val regionalBreakdown: List<AdminRegionBreakdownItem> = emptyList(),
    val predictiveInsight: String = "No elevated disease risk detected in the monitored regions.",
)

data class AdminModelConfigUiState(
    val activeModelFile: String = "",
    val activeModelName: String = "AquaNet",
    val activeVersion: String = "v1.0",
    val deployedDateText: String = "Local deployment",
    val statusText: String = "Healthy",
    val threshold: Float = 0.85f,
    val batchSize: String = "32 frames/s",
    val autoScalingEnabled: Boolean = true,
    val availableRevisions: List<AdminModelRevision> = emptyList(),
)

data class AdminModelRevision(
    val modelFile: String,
    val displayName: String,
    val version: String,
    val releaseLabel: String,
    val isActive: Boolean,
)

data class AdminModelConfigUpdate(
    val threshold: Float,
    val batchSize: String,
    val autoScalingEnabled: Boolean,
)

data class AdminInferenceLogsUiState(
    val averageInferenceTimeText: String = "N/A",
    val successRateText: String = "N/A",
    val activeModelFilter: String = "All Models",
    val logs: List<AdminInferenceLogItem> = emptyList(),
)

data class AdminInferenceLogItem(
    val id: String,
    val farmerId: String,
    val farmerName: String,
    val timestampText: String,
    val resultLabel: String,
    val confidenceText: String,
    val inferenceTimeText: String,
    val modelName: String,
    val kind: AdminInferenceLogKind,
)

enum class AdminInferenceLogKind {
    Healthy,
    Disease,
    Warning,
}

data class AdminChartPoint(
    val label: String,
    val value: Int,
    val highlighted: Boolean = false,
)

data class AdminActivityItem(
    val title: String,
    val subtitle: String,
    val relativeTime: String,
    val kind: AdminActivityKind,
)

enum class AdminActivityKind {
    Alert,
    Diagnosis,
    User,
    Sync,
}

data class AdminDiagnosisReviewItem(
    val id: String,
    val ownerId: String,
    val logId: Long,
    val specimenId: String,
    val farmerId: String,
    val farmerName: String,
    val farmLocation: String,
    val imageUri: String?,
    val predictedClass: String,
    val displayLabel: String,
    val confidence: Float,
    val confidenceText: String,
    val modelName: String,
    val submittedAt: String,
    val relativeTime: String,
    val status: AdminDiagnosisReviewStatus,
    val needsReviewReason: String,
    val correctedLabel: String?,
)

enum class AdminDiagnosisReviewStatus {
    Pending,
    Verified,
    Flagged,
    Excluded,
}

enum class AdminDiagnosisQueueFilter(val label: String) {
    Pending("Pending"),
    Verified("Verified"),
    Flagged("Flagged"),
}

data class AdminDiseasePrevalenceItem(
    val label: String,
    val count: Int,
    val percentage: Int,
)

data class AdminRegionBreakdownItem(
    val region: String,
    val activeCases: Int,
    val severity: AdminRegionSeverity,
)

enum class AdminRegionSeverity {
    Critical,
    Stable,
    Healthy,
}

data class AdminUserSummary(
    val id: String,
    val name: String,
    val account: String,
    val status: String,
    val diseaseCheckCount: Int,
    val lastActive: String,
    val farmLocation: String,
    val dataPermissionEnabled: Boolean,
)

data class AdminCreateUserInput(
    val displayName: String,
    val account: String,
    val password: String,
    val farmLocation: String,
    val phoneNumber: String,
    val email: String,
)

data class AdminDataReviewItem(
    val id: String,
    val ownerId: String,
    val logId: Long,
    val farmerName: String,
    val pond: String,
    val label: String,
    val permissionStatus: String,
    val reviewed: Boolean,
    val excluded: Boolean,
    val confidence: Float,
    val timestamp: Long,
)

fun emptyAdminChartPoints(): List<AdminChartPoint> {
    return listOf("M", "T", "W", "T", "F", "S", "S").map { label ->
        AdminChartPoint(label = label, value = 0)
    }
}
