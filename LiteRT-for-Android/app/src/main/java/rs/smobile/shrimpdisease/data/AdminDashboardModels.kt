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

data class AdminUserSummary(
    val id: String,
    val name: String,
    val status: String,
    val diseaseCheckCount: Int,
    val lastActive: String,
    val farmLocation: String,
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
