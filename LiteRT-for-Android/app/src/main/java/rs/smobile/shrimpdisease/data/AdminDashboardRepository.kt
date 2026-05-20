package rs.smobile.shrimpdisease.data

import android.content.Context
import android.net.Uri
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import rs.smobile.shrimpdisease.auth.AuthRepository
import rs.smobile.shrimpdisease.auth.AuthRole
import rs.smobile.shrimpdisease.auth.AuthUser
import rs.smobile.shrimpdisease.profile.FarmerProfileRepository
import rs.smobile.shrimpdisease.profile.FarmerProfileUiState
import rs.smobile.shrimpdisease.utils.DateTimeUtils
import java.util.Calendar
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.max
import kotlin.math.roundToInt

@Singleton
class AdminDashboardRepository @Inject constructor(
    @ApplicationContext context: Context,
    private val authRepository: AuthRepository,
    private val predictionLogRepository: PredictionLogRepository,
    private val farmerProfileRepository: FarmerProfileRepository,
) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    fun loadDashboard(): AdminDashboardUiState {
        val now = System.currentTimeMillis()
        val farmers = authRepository.getUsers()
            .filter { user -> user.role == AuthRole.Farmer }
            .sortedByDescending { user -> user.createdAt }
        val profiles = farmers.associateWith { user -> farmerProfileRepository.getProfileForUser(user) }
        val logsByFarmer = farmers.associateWith { user ->
            predictionLogRepository.getLogsForOwner(user.id)
        }
        val allLogs = logsByFarmer.values.flatten()
        val permittedLogs = logsByFarmer.flatMap { (farmer, logs) ->
            val profile = profiles.getValue(farmer)
            if (profile.dataPermissionEnabled) {
                logs.map { log -> farmer to log }
            } else {
                emptyList()
            }
        }

        val reviewedIds = readIdSet(KEY_REVIEWED_IDS)
        val excludedIds = readIdSet(KEY_EXCLUDED_IDS)
        val dataItems = permittedLogs
            .sortedByDescending { (_, log) -> log.timestamp }
            .map { (farmer, log) ->
                val profile = profiles.getValue(farmer)
                val itemId = dataItemId(farmer.id, log.id)
                AdminDataReviewItem(
                    id = itemId,
                    ownerId = farmer.id,
                    logId = log.id,
                    farmerName = profile.displayName,
                    pond = profile.farmLocation,
                    label = log.predictedClass,
                    permissionStatus = "Allowed",
                    reviewed = reviewedIds.contains(itemId),
                    excluded = excludedIds.contains(itemId),
                    confidence = log.confidence,
                    timestamp = log.timestamp,
                )
            }

        return AdminDashboardUiState(
            totalFarmers = farmers.size,
            totalDiseaseChecks = allLogs.size,
            imagesProcessed = permittedLogs.size,
            activeAlerts = dataItems.count { item ->
                !item.reviewed && !item.excluded && item.label.isDiseaseLabel()
            },
            farmersTrend = trendLabel(
                current = farmers.count { user -> user.createdAt >= now - ONE_WEEK_MS },
                previous = farmers.count { user ->
                    user.createdAt in (now - TWO_WEEKS_MS) until (now - ONE_WEEK_MS)
                },
            ),
            checksTrend = trendLabel(
                current = allLogs.count { log -> log.timestamp >= now - ONE_WEEK_MS },
                previous = allLogs.count { log ->
                    log.timestamp in (now - TWO_WEEKS_MS) until (now - ONE_WEEK_MS)
                },
            ),
            diagnosisVolume = weeklyVolume(allLogs, now),
            recentActivities = recentActivities(farmers, profiles, logsByFarmer, now),
            users = farmers.map { user ->
                val logs = logsByFarmer.getValue(user)
                val profile = profiles.getValue(user)
                AdminUserSummary(
                    id = user.id,
                    name = profile.displayName,
                    status = statusFor(user, logs, now),
                    diseaseCheckCount = logs.size,
                    lastActive = lastActiveText(user, logs, now),
                    farmLocation = profile.farmLocation,
                )
            },
            dataItems = dataItems,
        )
    }

    fun markReviewed(itemId: String): Boolean {
        updateIdSet(KEY_REVIEWED_IDS, itemId, enabled = true)
        return true
    }

    fun excludeFromTraining(itemId: String): Boolean {
        updateIdSet(KEY_EXCLUDED_IDS, itemId, enabled = true)
        updateIdSet(KEY_REVIEWED_IDS, itemId, enabled = true)
        return true
    }

    fun exportMetadata(context: Context, uri: Uri): Boolean {
        val items = loadDashboard().dataItems
        return try {
            context.contentResolver.openOutputStream(uri)?.bufferedWriter()?.use { writer ->
                writer.appendLine(
                    listOf(
                        "item_id",
                        "owner_id",
                        "log_id",
                        "farmer_name",
                        "farm_location",
                        "predicted_class",
                        "confidence",
                        "permission_status",
                        "reviewed",
                        "excluded",
                        "timestamp",
                    ).joinToString(",")
                )
                items.forEach { item ->
                    writer.appendLine(item.toCsvRow())
                }
            } != null
        } catch (error: Throwable) {
            Log.e(TAG, "Failed to export admin metadata", error)
            false
        }
    }

    private fun recentActivities(
        farmers: List<AuthUser>,
        profiles: Map<AuthUser, FarmerProfileUiState>,
        logsByFarmer: Map<AuthUser, List<PredictionLogItem>>,
        now: Long,
    ): List<AdminActivityItem> {
        val logActivities = logsByFarmer.flatMap { (farmer, logs) ->
            val profile = profiles.getValue(farmer)
            logs.map { log ->
                val status = DiagnosisHistoryStatus.from(log)
                when (status) {
                    DiagnosisHistoryStatus.DiseaseDetected -> AdminActivityItem(
                        title = "High Mortality Alert",
                        subtitle = "${profile.farmLocation} - ${log.predictedClass}",
                        relativeTime = relativeTime(log.timestamp, now),
                        kind = AdminActivityKind.Alert,
                    )

                    DiagnosisHistoryStatus.LowConfidence -> AdminActivityItem(
                        title = "Low Confidence Review",
                        subtitle = "${profile.farmLocation} needs a clearer image",
                        relativeTime = relativeTime(log.timestamp, now),
                        kind = AdminActivityKind.Diagnosis,
                    )

                    DiagnosisHistoryStatus.Healthy -> AdminActivityItem(
                        title = "Diagnosis Completed",
                        subtitle = "${profile.farmLocation} - Healthy result",
                        relativeTime = relativeTime(log.timestamp, now),
                        kind = AdminActivityKind.Diagnosis,
                    )
                } to log.timestamp
            }
        }

        val userActivities = farmers.map { farmer ->
            val profile = profiles.getValue(farmer)
            AdminActivityItem(
                title = "New Farmer Registered",
                subtitle = "${profile.displayName} joined ${profile.farmLocation}",
                relativeTime = relativeTime(farmer.createdAt, now),
                kind = AdminActivityKind.User,
            ) to farmer.createdAt
        }

        val activities = (logActivities + userActivities)
            .sortedByDescending { (_, timestamp) -> timestamp }
            .take(MAX_RECENT_ACTIVITIES)
            .map { (activity, _) -> activity }

        return activities.ifEmpty {
            listOf(
                AdminActivityItem(
                    title = "Model Sync Complete",
                    subtitle = "Diagnostic model is ready for on-device checks",
                    relativeTime = "Now",
                    kind = AdminActivityKind.Sync,
                )
            )
        }
    }

    private fun weeklyVolume(logs: List<PredictionLogItem>, now: Long): List<AdminChartPoint> {
        val todayStart = startOfDay(now)
        return (6 downTo 0).map { dayOffset ->
            val start = todayStart - dayOffset * ONE_DAY_MS
            val end = start + ONE_DAY_MS
            AdminChartPoint(
                label = dayLabel(start),
                value = logs.count { log -> log.timestamp in start until end },
                highlighted = dayOffset == 0,
            )
        }
    }

    private fun statusFor(user: AuthUser, logs: List<PredictionLogItem>, now: Long): String {
        val lastSeen = max(logs.maxOfOrNull { log -> log.timestamp } ?: 0L, user.createdAt)
        return if (lastSeen >= now - ONE_WEEK_MS) "Active" else "Inactive"
    }

    private fun lastActiveText(user: AuthUser, logs: List<PredictionLogItem>, now: Long): String {
        val lastSeen = max(logs.maxOfOrNull { log -> log.timestamp } ?: 0L, user.createdAt)
        return relativeTime(lastSeen, now)
    }

    private fun trendLabel(current: Int, previous: Int): String {
        val percent = when {
            current == 0 && previous == 0 -> 0
            previous == 0 -> 100
            else -> (((current - previous) * 100f) / previous).roundToInt()
        }
        return if (percent >= 0) "+$percent%" else "$percent%"
    }

    private fun relativeTime(timestamp: Long, now: Long): String {
        val elapsed = (now - timestamp).coerceAtLeast(0L)
        return when {
            elapsed < ONE_MINUTE_MS -> "Just now"
            elapsed < ONE_HOUR_MS -> "${elapsed / ONE_MINUTE_MS}m ago"
            elapsed < ONE_DAY_MS -> "${elapsed / ONE_HOUR_MS}h ago"
            elapsed < ONE_WEEK_MS -> "${elapsed / ONE_DAY_MS}d ago"
            else -> DateTimeUtils.formatTimestamp(timestamp).substringBefore(" ")
        }
    }

    private fun startOfDay(timestamp: Long): Long {
        return Calendar.getInstance().apply {
            timeInMillis = timestamp
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
    }

    private fun dayLabel(timestamp: Long): String {
        val calendar = Calendar.getInstance().apply { timeInMillis = timestamp }
        return when (calendar.get(Calendar.DAY_OF_WEEK)) {
            Calendar.MONDAY -> "M"
            Calendar.TUESDAY -> "T"
            Calendar.WEDNESDAY -> "W"
            Calendar.THURSDAY -> "T"
            Calendar.FRIDAY -> "F"
            Calendar.SATURDAY -> "S"
            else -> "S"
        }
    }

    private fun readIdSet(key: String): Set<String> {
        return preferences.getStringSet(key, emptySet()).orEmpty().toSet()
    }

    private fun updateIdSet(key: String, itemId: String, enabled: Boolean) {
        val updated = readIdSet(key).toMutableSet()
        if (enabled) {
            updated.add(itemId)
        } else {
            updated.remove(itemId)
        }
        preferences.edit()
            .putStringSet(key, updated)
            .apply()
    }

    private fun AdminDataReviewItem.toCsvRow(): String {
        return listOf(
            id,
            ownerId,
            logId.toString(),
            farmerName,
            pond,
            label,
            String.format(Locale.US, "%.6f", confidence),
            permissionStatus,
            reviewed.toString(),
            excluded.toString(),
            DateTimeUtils.formatTimestamp(timestamp),
        ).joinToString(",") { value -> value.csvEscape() }
    }

    private fun String.csvEscape(): String {
        val escaped = replace("\"", "\"\"")
        return if (any { char -> char == ',' || char == '"' || char == '\n' || char == '\r' }) {
            "\"$escaped\""
        } else {
            escaped
        }
    }

    private fun String.isDiseaseLabel(): Boolean {
        return !contains("healthy", ignoreCase = true)
    }

    private fun dataItemId(ownerId: String, logId: Long): String {
        return "$ownerId-$logId"
    }

    private companion object {
        private const val TAG = "AdminDashboardRepo"
        private const val PREFERENCES_NAME = "aquapulse_admin_dashboard"
        private const val KEY_REVIEWED_IDS = "reviewed_ids"
        private const val KEY_EXCLUDED_IDS = "excluded_ids"
        private const val MAX_RECENT_ACTIVITIES = 3
        private const val ONE_MINUTE_MS = 60L * 1000L
        private const val ONE_HOUR_MS = 60L * ONE_MINUTE_MS
        private const val ONE_DAY_MS = 24L * ONE_HOUR_MS
        private const val ONE_WEEK_MS = 7L * ONE_DAY_MS
        private const val TWO_WEEKS_MS = 14L * ONE_DAY_MS
    }
}
