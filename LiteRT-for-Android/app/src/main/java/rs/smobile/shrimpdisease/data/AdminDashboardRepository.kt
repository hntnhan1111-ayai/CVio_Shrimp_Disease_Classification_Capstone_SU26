package rs.smobile.shrimpdisease.data

import android.content.Context
import android.net.Uri
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import rs.smobile.shrimpdisease.auth.AuthResult
import rs.smobile.shrimpdisease.auth.AuthRepository
import rs.smobile.shrimpdisease.auth.AuthRole
import rs.smobile.shrimpdisease.auth.AuthUser
import rs.smobile.shrimpdisease.profile.FarmerProfileRepository
import rs.smobile.shrimpdisease.profile.FarmerProfileUiState
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import rs.smobile.shrimpdisease.utils.DateTimeUtils
import org.json.JSONObject
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
                    account = user.account,
                    status = statusFor(user, logs, now),
                    diseaseCheckCount = logs.size,
                    lastActive = lastActiveText(user, logs, now),
                    farmLocation = profile.farmLocation,
                    dataPermissionEnabled = profile.dataPermissionEnabled,
                )
            },
            dataItems = dataItems,
        )
    }

    fun createUser(input: AdminCreateUserInput): AuthResult {
        val result = authRepository.createManagedFarmer(
            account = input.account,
            password = input.password,
            displayName = input.displayName,
        )
        if (result is AuthResult.Success) {
            farmerProfileRepository.saveProfileForUser(
                user = result.user,
                profile = FarmerProfileUiState(
                    userId = result.user.id,
                    displayName = input.displayName.trim().ifBlank { result.user.displayName },
                    farmLocation = input.farmLocation.trim().ifBlank { "Coastal pond" },
                    phoneNumber = input.phoneNumber.trim().ifBlank { "Not set" },
                    email = input.email.trim().ifBlank {
                        result.user.account.takeIf { account -> account.contains("@") } ?: "Not set"
                    },
                    dataPermissionEnabled = true,
                ),
            )
        }
        return result
    }

    fun loadDiagnosis(): AdminDiagnosisUiState {
        val now = System.currentTimeMillis()
        val contexts = diagnosisContexts()
        val reviewedIds = readIdSet(KEY_REVIEWED_IDS)
        val excludedIds = readIdSet(KEY_EXCLUDED_IDS)
        val correctedLabels = readCorrectionMap()
        val items = contexts.map { context ->
            context.toDiagnosisReviewItem(
                now = now,
                reviewedIds = reviewedIds,
                excludedIds = excludedIds,
                correctedLabels = correctedLabels,
            )
        }.sortedByDescending { item -> item.logId }
        val visibleItems = items.filter { item -> item.status != AdminDiagnosisReviewStatus.Excluded }
        val evaluatedLogs = contexts.map { context -> context.log }.filter { log -> log.isCorrect != null }
        val activeOutbreaks = visibleItems.count { item ->
            item.displayLabel.isDiseaseLabel() && item.status != AdminDiagnosisReviewStatus.Verified
        }

        return AdminDiagnosisUiState(
            overallAccuracyText = percentText(accuracy(evaluatedLogs)),
            truePositiveRateText = percentText(truePositiveRate(contexts)),
            falseNegativeRateText = percentText(falseNegativeRate(contexts)),
            accuracyTrendText = accuracyTrendText(contexts, now),
            totalScans = contexts.size,
            pendingCount = visibleItems.count { item -> item.status == AdminDiagnosisReviewStatus.Pending },
            verifiedCount = visibleItems.count { item -> item.status == AdminDiagnosisReviewStatus.Verified },
            flaggedCount = visibleItems.count { item -> item.status == AdminDiagnosisReviewStatus.Flagged },
            activeOutbreaks = activeOutbreaks,
            queueItems = visibleItems,
            flaggedDiagnoses = visibleItems
                .filter { item -> item.status == AdminDiagnosisReviewStatus.Flagged }
                .take(MAX_FLAGGED_DIAGNOSES),
            prevalence = diseasePrevalence(contexts),
            outbreakTrend = outbreakTrend(contexts, now),
            regionalBreakdown = regionalBreakdown(visibleItems),
            predictiveInsight = predictiveInsight(visibleItems, activeOutbreaks),
        )
    }

    fun loadModelConfig(
        activeModelFile: String,
        availableModels: List<String>,
        threshold: Float,
    ): AdminModelConfigUiState {
        val stored = loadModelRuntimeConfig(defaultThreshold = threshold)
        val activeFile = preferences.getString(KEY_ACTIVE_MODEL_FILE, activeModelFile)
            ?.takeIf { file -> file.isNotBlank() } ?: activeModelFile
        return AdminModelConfigUiState(
            activeModelFile = activeFile,
            activeModelName = displayModelName(activeFile),
            activeVersion = versionForModel(activeFile),
            deployedDateText = preferences.getString(KEY_MODEL_DEPLOYED_DATE, null) ?: "Local deployment",
            statusText = "Healthy",
            threshold = stored.threshold,
            batchSize = stored.batchSize,
            autoScalingEnabled = stored.autoScalingEnabled,
            availableRevisions = availableModels.map { modelFile ->
                AdminModelRevision(
                    modelFile = modelFile,
                    displayName = displayModelName(modelFile),
                    version = versionForModel(modelFile),
                    releaseLabel = releaseLabelForModel(modelFile),
                    isActive = modelFile == activeFile,
                )
            },
        )
    }

    fun saveModelConfig(update: AdminModelConfigUpdate): AdminModelConfigUiState {
        preferences.edit()
            .putFloat(KEY_MODEL_THRESHOLD, update.threshold.coerceIn(0f, 1f))
            .putString(KEY_MODEL_BATCH_SIZE, update.batchSize)
            .putBoolean(KEY_MODEL_AUTO_SCALING, update.autoScalingEnabled)
            .apply()
        return loadModelConfig(
            activeModelFile = preferences.getString(KEY_ACTIVE_MODEL_FILE, null).orEmpty(),
            availableModels = emptyList(),
            threshold = update.threshold,
        )
    }

    fun saveActiveModel(modelFile: String) {
        preferences.edit()
            .putString(KEY_ACTIVE_MODEL_FILE, modelFile)
            .putString(KEY_MODEL_DEPLOYED_DATE, DateTimeUtils.formatTimestamp(System.currentTimeMillis()).substringBefore(" "))
            .apply()
    }

    fun loadInferenceLogs(activeModelFilter: String? = null): AdminInferenceLogsUiState {
        val contexts = diagnosisContexts()
        val filteredContexts = if (activeModelFilter.isNullOrBlank() || activeModelFilter == ALL_MODELS_FILTER) {
            contexts
        } else {
            contexts.filter { context -> context.log.modelName == activeModelFilter }
        }
        val logs = filteredContexts
            .sortedByDescending { context -> context.log.timestamp }
            .map { context -> context.toInferenceLogItem() }
        val successful = filteredContexts.count { context -> context.log.isAboveThreshold }
        val averageTime = BenchmarkUtils.calculateAverageTime(filteredContexts.map { context -> context.log.inferenceTimeMs })
        val successRate = if (filteredContexts.isEmpty()) {
            null
        } else {
            successful.toFloat() / filteredContexts.size
        }

        return AdminInferenceLogsUiState(
            averageInferenceTimeText = if (filteredContexts.isEmpty()) {
                "N/A"
            } else {
                BenchmarkUtils.latencyText(averageTime.toLong())
            },
            successRateText = successRate?.let { value ->
                String.format(Locale.US, "%.1f%%", value * 100f)
            } ?: "N/A",
            activeModelFilter = activeModelFilter ?: ALL_MODELS_FILTER,
            logs = logs,
        )
    }

    fun confirmDiagnosis(itemId: String): Boolean {
        updateIdSet(KEY_REVIEWED_IDS, itemId, enabled = true)
        return true
    }

    fun correctDiagnosis(
        itemId: String,
        correctedLabel: String,
    ): Boolean {
        val cleanLabel = correctedLabel.trim()
        if (cleanLabel.isBlank()) return false
        val correctedLabels = readCorrectionMap().toMutableMap()
        correctedLabels[itemId] = cleanLabel
        writeCorrectionMap(correctedLabels)
        updateIdSet(KEY_REVIEWED_IDS, itemId, enabled = true)
        return true
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

    private fun diagnosisContexts(): List<AdminDiagnosisContext> {
        val farmers = authRepository.getUsers()
            .filter { user -> user.role == AuthRole.Farmer }
        return farmers.flatMap { user ->
            val profile = farmerProfileRepository.getProfileForUser(user)
            predictionLogRepository.getLogsForOwner(user.id).map { log ->
                AdminDiagnosisContext(
                    user = user,
                    profile = profile,
                    log = log,
                )
            }
        }
    }

    private fun AdminDiagnosisContext.toInferenceLogItem(): AdminInferenceLogItem {
        val status = DiagnosisHistoryStatus.from(log)
        return AdminInferenceLogItem(
            id = dataItemId(user.id, log.id),
            farmerId = "FRM-${user.id.takeLast(4).uppercase()}",
            farmerName = profile.displayName,
            timestampText = DateTimeUtils.formatTimestamp(log.timestamp),
            resultLabel = when (status) {
                DiagnosisHistoryStatus.Healthy -> "Healthy"
                DiagnosisHistoryStatus.DiseaseDetected -> "${log.predictedClass} detected"
                DiagnosisHistoryStatus.LowConfidence -> "Warning"
            },
            confidenceText = String.format(Locale.US, "%.1f%%", log.confidence * 100f),
            inferenceTimeText = BenchmarkUtils.latencyText(log.inferenceTimeMs),
            modelName = log.modelName,
            kind = when (status) {
                DiagnosisHistoryStatus.Healthy -> AdminInferenceLogKind.Healthy
                DiagnosisHistoryStatus.DiseaseDetected -> AdminInferenceLogKind.Disease
                DiagnosisHistoryStatus.LowConfidence -> AdminInferenceLogKind.Warning
            },
        )
    }

    private fun AdminDiagnosisContext.toDiagnosisReviewItem(
        now: Long,
        reviewedIds: Set<String>,
        excludedIds: Set<String>,
        correctedLabels: Map<String, String>,
    ): AdminDiagnosisReviewItem {
        val itemId = dataItemId(user.id, log.id)
        val correctedLabel = correctedLabels[itemId]
        val displayLabel = correctedLabel ?: log.predictedClass
        val reviewReason = reviewReasonFor(log, correctedLabel)
        val status = when {
            excludedIds.contains(itemId) -> AdminDiagnosisReviewStatus.Excluded
            reviewedIds.contains(itemId) -> AdminDiagnosisReviewStatus.Verified
            reviewReason != REVIEW_REASON_READY -> AdminDiagnosisReviewStatus.Flagged
            else -> AdminDiagnosisReviewStatus.Pending
        }
        return AdminDiagnosisReviewItem(
            id = itemId,
            ownerId = user.id,
            logId = log.id,
            specimenId = specimenIdFor(log.id),
            farmerId = "AQ-${user.id.takeLast(6).uppercase()}",
            farmerName = profile.displayName,
            farmLocation = profile.farmLocation,
            imageUri = log.imageUri,
            predictedClass = log.predictedClass,
            displayLabel = displayLabel,
            confidence = log.confidence,
            confidenceText = String.format(Locale.US, "%.0f%%", log.confidence * 100f),
            modelName = log.modelName,
            submittedAt = DateTimeUtils.formatTimestamp(log.timestamp).substringBefore(" "),
            relativeTime = relativeTime(log.timestamp, now),
            status = status,
            needsReviewReason = reviewReason,
            correctedLabel = correctedLabel,
        )
    }

    private fun reviewReasonFor(
        log: PredictionLogItem,
        correctedLabel: String?,
    ): String {
        return when {
            correctedLabel != null -> "Corrected Label"
            !log.isAboveThreshold -> "Low Confidence"
            log.predictedClass.isDiseaseLabel() && log.confidence < DISEASE_REVIEW_THRESHOLD -> "Needs Review"
            else -> REVIEW_REASON_READY
        }
    }

    private fun accuracy(logs: List<PredictionLogItem>): Float? {
        if (logs.isEmpty()) return null
        return logs.count { log -> log.isCorrect == true }.toFloat() / logs.size
    }

    private fun truePositiveRate(contexts: List<AdminDiagnosisContext>): Float? {
        val actualDisease = contexts.map { context -> context.log }
            .filter { log -> log.groundTruthLabel?.isDiseaseLabel() == true }
        if (actualDisease.isEmpty()) return null
        val truePositive = actualDisease.count { log -> log.predictedClass.isDiseaseLabel() }
        return truePositive.toFloat() / actualDisease.size
    }

    private fun falseNegativeRate(contexts: List<AdminDiagnosisContext>): Float? {
        val actualDisease = contexts.map { context -> context.log }
            .filter { log -> log.groundTruthLabel?.isDiseaseLabel() == true }
        if (actualDisease.isEmpty()) return null
        val falseNegative = actualDisease.count { log ->
            log.predictedClass.contains("healthy", ignoreCase = true)
        }
        return falseNegative.toFloat() / actualDisease.size
    }

    private fun accuracyTrendText(
        contexts: List<AdminDiagnosisContext>,
        now: Long,
    ): String {
        val currentLogs = contexts.map { context -> context.log }
            .filter { log -> log.timestamp >= now - ONE_WEEK_MS && log.isCorrect != null }
        val previousLogs = contexts.map { context -> context.log }
            .filter { log ->
                log.timestamp in (now - TWO_WEEKS_MS) until (now - ONE_WEEK_MS) &&
                    log.isCorrect != null
            }
        val currentAccuracy = accuracy(currentLogs)
        val previousAccuracy = accuracy(previousLogs)
        val delta = when {
            currentAccuracy == null || previousAccuracy == null -> 0f
            else -> (currentAccuracy - previousAccuracy) * 100f
        }
        val prefix = if (delta >= 0f) "+" else ""
        return String.format(Locale.US, "%s%.1f%% from last week", prefix, delta)
    }

    private fun diseasePrevalence(contexts: List<AdminDiagnosisContext>): List<AdminDiseasePrevalenceItem> {
        val logs = contexts.map { context -> context.log }
        if (logs.isEmpty()) return emptyList()
        val grouped = logs.groupingBy { log -> diseaseBucket(log.predictedClass) }.eachCount()
        return grouped.entries
            .sortedByDescending { entry -> entry.value }
            .map { entry ->
                AdminDiseasePrevalenceItem(
                    label = entry.key,
                    count = entry.value,
                    percentage = ((entry.value * 100f) / logs.size).roundToInt(),
                )
            }
    }

    private fun outbreakTrend(
        contexts: List<AdminDiagnosisContext>,
        now: Long,
    ): List<AdminChartPoint> {
        return (3 downTo 0).map { weekOffset ->
            val end = now - weekOffset * ONE_WEEK_MS
            val start = end - ONE_WEEK_MS
            AdminChartPoint(
                label = "Week ${4 - weekOffset}",
                value = contexts.count { context ->
                    context.log.timestamp in start until end &&
                        context.log.predictedClass.isDiseaseLabel()
                },
                highlighted = weekOffset == 0,
            )
        }
    }

    private fun regionalBreakdown(items: List<AdminDiagnosisReviewItem>): List<AdminRegionBreakdownItem> {
        return items
            .groupBy { item -> item.farmLocation.ifBlank { "Unknown Region" } }
            .map { (region, regionItems) ->
                val activeCases = regionItems.count { item ->
                    item.displayLabel.isDiseaseLabel() &&
                        item.status != AdminDiagnosisReviewStatus.Verified
                }
                AdminRegionBreakdownItem(
                    region = region,
                    activeCases = activeCases,
                    severity = when {
                        activeCases >= CRITICAL_REGION_CASES -> AdminRegionSeverity.Critical
                        activeCases > 0 -> AdminRegionSeverity.Stable
                        else -> AdminRegionSeverity.Healthy
                    },
                )
            }
            .sortedWith(
                compareByDescending<AdminRegionBreakdownItem> { item -> item.activeCases }
                    .thenBy { item -> item.region }
            )
            .take(MAX_REGIONS)
    }

    private fun predictiveInsight(
        items: List<AdminDiagnosisReviewItem>,
        activeOutbreaks: Int,
    ): String {
        val riskiestRegion = items
            .filter { item -> item.displayLabel.isDiseaseLabel() }
            .groupingBy { item -> item.farmLocation.ifBlank { "Unknown Region" } }
            .eachCount()
            .maxByOrNull { entry -> entry.value }
        return if (riskiestRegion == null || activeOutbreaks == 0) {
            "No elevated disease risk detected in the monitored regions."
        } else {
            "Machine learning review suggests elevated disease risk in ${riskiestRegion.key}; $activeOutbreaks active cases should be reviewed in the next 48 hours."
        }
    }

    private fun diseaseBucket(label: String): String {
        return when {
            label.contains("healthy", ignoreCase = true) -> "Healthy"
            label.contains("wssv", ignoreCase = true) -> "WSSV"
            label.contains("bg", ignoreCase = true) -> "BG"
            else -> label.ifBlank { "Unknown" }
        }
    }

    private fun percentText(value: Float?): String {
        return value?.let { String.format(Locale.US, "%.1f%%", it * 100f) } ?: "N/A"
    }

    private fun loadModelRuntimeConfig(defaultThreshold: Float): AdminModelConfigUpdate {
        return AdminModelConfigUpdate(
            threshold = preferences.getFloat(KEY_MODEL_THRESHOLD, defaultThreshold),
            batchSize = preferences.getString(KEY_MODEL_BATCH_SIZE, DEFAULT_BATCH_SIZE) ?: DEFAULT_BATCH_SIZE,
            autoScalingEnabled = preferences.getBoolean(KEY_MODEL_AUTO_SCALING, true),
        )
    }

    private fun displayModelName(modelFile: String): String {
        return modelFile
            .substringBeforeLast(".")
            .replace("_", " ")
            .replace("-", " ")
            .split(" ")
            .filter { token -> token.isNotBlank() }
            .joinToString(" ") { token -> token.replaceFirstChar { char -> char.uppercase() } }
            .ifBlank { "AquaNet" }
    }

    private fun versionForModel(modelFile: String): String {
        val normalized = modelFile.lowercase()
        return when {
            "yolo" in normalized -> "v2.6"
            "efficientnet" in normalized -> "v2.4"
            "mobilenet" in normalized -> "v2.3"
            else -> "v1.0"
        }
    }

    private fun releaseLabelForModel(modelFile: String): String {
        val normalized = modelFile.lowercase()
        return when {
            "float16" in normalized -> "Optimized release"
            "float32" in normalized -> "Stable release"
            "dynamic" in normalized -> "Mobile quantized"
            else -> "Packaged model"
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

    private fun readCorrectionMap(): Map<String, String> {
        val json = preferences.getString(KEY_CORRECTED_LABELS, null) ?: return emptyMap()
        return runCatching {
            val values = JSONObject(json)
            values.keys().asSequence().associateWith { key -> values.getString(key) }
        }.getOrElse { emptyMap() }
    }

    private fun writeCorrectionMap(labels: Map<String, String>) {
        val json = JSONObject()
        labels.forEach { (key, value) -> json.put(key, value) }
        preferences.edit()
            .putString(KEY_CORRECTED_LABELS, json.toString())
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

    private fun specimenIdFor(logId: Long): String {
        return "#${logId.toString().takeLast(4)}-${(logId % 26 + 'A'.code).toInt().toChar()}"
    }

    private data class AdminDiagnosisContext(
        val user: AuthUser,
        val profile: FarmerProfileUiState,
        val log: PredictionLogItem,
    )

    private companion object {
        private const val TAG = "AdminDashboardRepo"
        private const val PREFERENCES_NAME = "aquapulse_admin_dashboard"
        private const val KEY_REVIEWED_IDS = "reviewed_ids"
        private const val KEY_EXCLUDED_IDS = "excluded_ids"
        private const val KEY_CORRECTED_LABELS = "corrected_labels"
        private const val KEY_ACTIVE_MODEL_FILE = "active_model_file"
        private const val KEY_MODEL_DEPLOYED_DATE = "model_deployed_date"
        private const val KEY_MODEL_THRESHOLD = "model_threshold"
        private const val KEY_MODEL_BATCH_SIZE = "model_batch_size"
        private const val KEY_MODEL_AUTO_SCALING = "model_auto_scaling"
        private const val DEFAULT_BATCH_SIZE = "32 frames/s"
        private const val ALL_MODELS_FILTER = "All Models"
        private const val MAX_RECENT_ACTIVITIES = 3
        private const val MAX_FLAGGED_DIAGNOSES = 5
        private const val MAX_REGIONS = 3
        private const val CRITICAL_REGION_CASES = 4
        private const val DISEASE_REVIEW_THRESHOLD = 0.9f
        private const val REVIEW_REASON_READY = "Ready for Review"
        private const val ONE_MINUTE_MS = 60L * 1000L
        private const val ONE_HOUR_MS = 60L * ONE_MINUTE_MS
        private const val ONE_DAY_MS = 24L * ONE_HOUR_MS
        private const val ONE_WEEK_MS = 7L * ONE_DAY_MS
        private const val TWO_WEEKS_MS = 14L * ONE_DAY_MS
    }
}
