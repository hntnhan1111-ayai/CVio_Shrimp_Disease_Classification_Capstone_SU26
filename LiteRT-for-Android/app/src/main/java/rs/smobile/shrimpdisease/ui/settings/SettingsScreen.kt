package rs.smobile.shrimpdisease.ui.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.RuntimeDelegate
import rs.smobile.shrimpdisease.SettingsUiState
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.data.AdminInferenceLogItem
import rs.smobile.shrimpdisease.data.AdminInferenceLogKind
import rs.smobile.shrimpdisease.data.AdminInferenceLogsUiState
import rs.smobile.shrimpdisease.data.AdminModelConfigUiState
import rs.smobile.shrimpdisease.data.AdminModelConfigUpdate
import rs.smobile.shrimpdisease.data.AdminModelRevision
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.ui.components.BenchmarkSummaryCard
import rs.smobile.shrimpdisease.ui.components.CVioMetricTile
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.components.CompactInfoRow
import rs.smobile.shrimpdisease.ui.components.DataPermissionToggle
import rs.smobile.shrimpdisease.ui.components.ModelSelector
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.components.SettingsSectionCard
import rs.smobile.shrimpdisease.ui.components.ThresholdSlider
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.ui.theme.WarningOrange
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import java.util.Locale

@Composable
fun SettingsScreen(
    modelInfo: ModelInfo,
    labels: List<String>,
    availableModels: List<String>,
    settingsState: SettingsUiState,
    benchmarkMetrics: BenchmarkMetrics,
    isLoading: Boolean,
    errorMessage: String?,
    onSelectModel: (String) -> Unit,
    onDelegateSelected: (RuntimeDelegate) -> Unit,
    onThresholdChange: (Float) -> Unit,
    onShowDebugInfoChange: (Boolean) -> Unit,
    onResetMetrics: () -> Unit,
    userDisplayName: String? = null,
    userRole: String? = null,
    onLogout: (() -> Unit)? = null,
    adminModelConfigUiState: AdminModelConfigUiState? = null,
    adminInferenceLogsUiState: AdminInferenceLogsUiState? = null,
    onAdminModelConfigSave: ((AdminModelConfigUpdate) -> Unit)? = null,
    onAdminModelDeploy: ((String) -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    if (
        userRole == "Admin" &&
        adminModelConfigUiState != null &&
        adminInferenceLogsUiState != null &&
        onAdminModelConfigSave != null &&
        onAdminModelDeploy != null
    ) {
        AdminSettingsContent(
            modelConfig = adminModelConfigUiState,
            inferenceLogs = adminInferenceLogsUiState,
            benchmarkMetrics = benchmarkMetrics,
            userDisplayName = userDisplayName,
            onLogout = onLogout,
            onSaveConfig = onAdminModelConfigSave,
            onDeployModel = onAdminModelDeploy,
            modifier = modifier,
        )
        return
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "Cài đặt",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface,
        )

        if (userDisplayName != null && userRole != null && onLogout != null) {
            SettingsSectionCard(
                title = "Tài khoản",
                subtitle = "Thông tin đang đăng nhập",
                containerColor = CVioSurfaceContainerLow,
            ) {
                CompactInfoRow("Tên", displayFarmerName(userDisplayName))
                CompactInfoRow("Vai trò", displayRoleName(userRole))
                SecondaryActionButton(
                    text = "Đăng xuất",
                    onClick = onLogout,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        SettingsSectionCard(
            title = "Cài đặt AI",
            subtitle = "Chọn mô hình AI dùng để kiểm tra tôm khi không có mạng.",
        ) {
            ModelSelector(
                selectedModel = modelInfo.modelFile,
                availableModels = availableModels,
                onModelSelected = onSelectModel,
            )
            if (isLoading) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Text(
                    text = "Đang tải mô hình...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (errorMessage != null) {
                Text(
                    text = errorMessage,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }

        SettingsSectionCard(
            title = "Cách chạy AI",
            subtitle = "AI chạy trực tiếp trên điện thoại.",
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = settingsState.runtimeDelegate == RuntimeDelegate.CPU,
                    onClick = { onDelegateSelected(RuntimeDelegate.CPU) },
                    label = { Text(text = "CPU") },
                )
                FilterChip(
                    selected = false,
                    onClick = { onDelegateSelected(RuntimeDelegate.GPU) },
                    enabled = false,
                    label = { Text(text = "GPU chưa dùng được") },
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                CVioMetricTile(
                    label = "Cỡ ảnh vào",
                    value = BenchmarkUtils.inputSizeText(modelInfo),
                    modifier = Modifier.weight(1f),
                )
                CVioMetricTile(
                    label = "Luồng xử lý",
                    value = "4",
                    modifier = Modifier.weight(1f),
                    accent = MaterialTheme.colorScheme.secondary,
                )
            }
        }

        SettingsSectionCard(
            title = "Ngưỡng tin cậy",
            subtitle = "Kết quả thấp hơn ngưỡng sẽ được báo là ảnh chưa đủ rõ.",
            containerColor = CVioSurfaceContainerLow,
        ) {
            ThresholdSlider(
                threshold = settingsState.confidenceThreshold,
                onThresholdChange = onThresholdChange,
            )
        }

        SettingsSectionCard(
            title = "Thông tin kỹ thuật",
            subtitle = "Chỉ bật khi cần xem chi tiết kỹ thuật.",
        ) {
            DataPermissionToggle(
                title = "Hiện thông tin kỹ thuật",
                message = "Hiện nguồn ảnh, mô hình, FPS và thời gian xử lý trên màn hình kết quả.",
                checked = settingsState.showDebugInfo,
                onCheckedChange = onShowDebugInfoChange,
            )
            CVioStatusChip(
                text = if (settingsState.showDebugInfo) "Đã hiện thông tin kỹ thuật" else "Đang ẩn thông tin kỹ thuật",
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        BenchmarkSummaryCard(metrics = benchmarkMetrics)

        SecondaryActionButton(
            text = "Xóa lịch sử kiểm tra",
            enabled = benchmarkMetrics.totalRuns > 0,
            onClick = onResetMetrics,
            modifier = Modifier.fillMaxWidth(),
        )

        SettingsSectionCard(
            title = "Thông tin mô hình",
            subtitle = "Thông tin đầu vào, đầu ra và các loại AI nhận biết.",
        ) {
            CompactInfoRow("Tên mô hình", modelInfo.modelFile)
            CompactInfoRow("Dòng mô hình", modelInfo.modelFamily)
            CompactInfoRow("Dạng dữ liệu vào", BenchmarkUtils.shapeText(modelInfo.input.shape))
            CompactInfoRow("Dạng kết quả", BenchmarkUtils.shapeText(modelInfo.output.shape))
            CompactInfoRow("Số loại nhận biết", modelInfo.outputClassCount.toString())
            Text(
                text = labels.joinToString(", "),
                maxLines = 4,
                overflow = TextOverflow.Ellipsis,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (modelInfo.warnings.isNotEmpty()) {
                modelInfo.warnings.forEach { warning ->
                    Text(
                        text = warning,
                        color = DiseaseRed,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

@Composable
private fun AdminSettingsContent(
    modelConfig: AdminModelConfigUiState,
    inferenceLogs: AdminInferenceLogsUiState,
    benchmarkMetrics: BenchmarkMetrics,
    userDisplayName: String?,
    onLogout: (() -> Unit)?,
    onSaveConfig: (AdminModelConfigUpdate) -> Unit,
    onDeployModel: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var threshold by rememberSaveable(modelConfig.activeModelFile, modelConfig.threshold) {
        mutableStateOf(modelConfig.threshold)
    }
    var batchSize by rememberSaveable(modelConfig.activeModelFile, modelConfig.batchSize) {
        mutableStateOf(modelConfig.batchSize)
    }
    var autoScaling by rememberSaveable(modelConfig.activeModelFile, modelConfig.autoScalingEnabled) {
        mutableStateOf(modelConfig.autoScalingEnabled)
    }
    var searchQuery by rememberSaveable { mutableStateOf("") }
    var selectedResult by rememberSaveable { mutableStateOf("All") }
    var selectedModel by rememberSaveable { mutableStateOf("All Models") }
    val modelFilters = listOf("All Models") + inferenceLogs.logs.map { log -> log.modelName }.distinct()
    val visibleLogs = inferenceLogs.logs.filter { log ->
        val matchesSearch = searchQuery.isBlank() ||
            log.farmerId.contains(searchQuery, ignoreCase = true) ||
            log.farmerName.contains(searchQuery, ignoreCase = true) ||
            log.resultLabel.contains(searchQuery, ignoreCase = true)
        val matchesResult = selectedResult == "All" || log.kind.name == selectedResult
        val matchesModel = selectedModel == "All Models" || log.modelName == selectedModel
        matchesSearch && matchesResult && matchesModel
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        AdminSettingsTopBar(userDisplayName = userDisplayName)

        if (onLogout != null) {
            SettingsSectionCard(
                title = "Account",
                subtitle = "Signed in as Admin",
                containerColor = CVioSurfaceContainerLow,
            ) {
                CompactInfoRow("Name", userDisplayName ?: "Admin")
                CompactInfoRow("Role", "Admin")
                SecondaryActionButton(
                    text = "Logout",
                    onClick = onLogout,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        ModelConfigurationSection(
            modelConfig = modelConfig,
            threshold = threshold,
            batchSize = batchSize,
            autoScaling = autoScaling,
            onThresholdChange = { threshold = it },
            onBatchSizeChange = { batchSize = it },
            onAutoScalingChange = { autoScaling = it },
            onSaveConfig = {
                onSaveConfig(
                    AdminModelConfigUpdate(
                        threshold = threshold,
                        batchSize = batchSize,
                        autoScalingEnabled = autoScaling,
                    )
                )
            },
            onDeployModel = onDeployModel,
        )

        InferenceLogsSection(
            inferenceLogs = inferenceLogs,
            visibleLogs = visibleLogs,
            searchQuery = searchQuery,
            selectedResult = selectedResult,
            selectedModel = selectedModel,
            modelFilters = modelFilters,
            onSearchChange = { searchQuery = it },
            onResultSelected = { selectedResult = it },
            onModelSelected = { selectedModel = it },
        )

        BenchmarkSummaryCard(metrics = benchmarkMetrics)
    }
}

@Composable
private fun AdminSettingsTopBar(userDisplayName: String?) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = "MarineCare AI Admin",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = userDisplayName ?: "System Administrator",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(999.dp))
                .background(MaterialTheme.colorScheme.secondaryContainer)
                .padding(horizontal = 14.dp, vertical = 10.dp),
        ) {
            Text(
                text = "A",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

@Composable
private fun ModelConfigurationSection(
    modelConfig: AdminModelConfigUiState,
    threshold: Float,
    batchSize: String,
    autoScaling: Boolean,
    onThresholdChange: (Float) -> Unit,
    onBatchSizeChange: (String) -> Unit,
    onAutoScalingChange: (Boolean) -> Unit,
    onSaveConfig: () -> Unit,
    onDeployModel: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Model Configuration",
                style = MaterialTheme.typography.displayLarge,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "M",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(32.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
        ) {
            Column {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.3f))
                        .padding(horizontal = 20.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "ACTIVE DEPLOYMENT",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                        fontWeight = FontWeight.Bold,
                    )
                    CVioStatusChip(
                        text = modelConfig.activeVersion,
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                    )
                }
                Row(
                    modifier = Modifier.padding(20.dp),
                    horizontalArrangement = Arrangement.spacedBy(18.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(24.dp))
                            .background(MaterialTheme.colorScheme.primaryContainer)
                            .padding(22.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = "AI",
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Text(
                            text = modelConfig.activeModelName,
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Text(
                            text = "Deployed: ${modelConfig.deployedDateText}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        CVioStatusChip(
                            text = "Status: ${modelConfig.statusText}",
                            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.42f),
                            contentColor = MaterialTheme.colorScheme.secondary,
                        )
                    }
                }
            }
        }

        Text(
            text = "AVAILABLE REVISIONS",
            modifier = Modifier.padding(horizontal = 8.dp),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        modelConfig.availableRevisions.forEach { revision ->
            ModelRevisionCard(
                revision = revision,
                onDeployModel = onDeployModel,
            )
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(32.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
        ) {
            Column(
                modifier = Modifier.padding(22.dp),
                verticalArrangement = Arrangement.spacedBy(18.dp),
            ) {
                Text(
                    text = "Runtime Parameters",
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = "Inference Threshold",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            text = String.format(Locale.US, "%.2f", threshold),
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    Slider(
                        value = threshold,
                        onValueChange = onThresholdChange,
                        valueRange = 0f..1f,
                    )
                    Text(
                        text = "Confidence score required to flag health anomalies.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        text = "Batch Size",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        listOf("16 frames/s", "32 frames/s", "64 frames/s").forEach { option ->
                            ConfigPill(
                                label = option,
                                selected = batchSize == option,
                                onClick = { onBatchSizeChange(option) },
                            )
                        }
                    }
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(22.dp))
                        .background(CVioSurfaceContainerLow)
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column {
                        Text(
                            text = "Auto-Scaling",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurface,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = "Load based",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = autoScaling,
                        onCheckedChange = onAutoScalingChange,
                    )
                }

                PrimarySaveButton(onClick = onSaveConfig)
            }
        }
    }
}

@Composable
private fun ModelRevisionCard(
    revision: AdminModelRevision,
    onDeployModel: (String) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = CVioSurfaceContainerLowest),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Row(
            modifier = Modifier.padding(18.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = if (revision.isActive) "A" else "H",
                style = MaterialTheme.typography.labelMedium,
                color = if (revision.isActive) HealthyGreen else MaterialTheme.colorScheme.onSurfaceVariant,
                fontWeight = FontWeight.Bold,
            )
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                Text(
                    text = revision.displayName,
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = "${revision.releaseLabel} - ${revision.version}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            ConfigPill(
                label = if (revision.isActive) "Active" else "Deploy",
                selected = revision.isActive,
                onClick = {
                    if (!revision.isActive) onDeployModel(revision.modelFile)
                },
            )
        }
    }
}

@Composable
private fun ConfigPill(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(
                if (selected) MaterialTheme.colorScheme.primaryContainer else CVioSurfaceContainerLow
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 10.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = if (selected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
            maxLines = 1,
        )
    }
}

@Composable
private fun PrimarySaveButton(onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(999.dp))
            .background(MaterialTheme.colorScheme.primary)
            .clickable(onClick = onClick)
            .padding(vertical = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = "Save Changes",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onPrimary,
            fontWeight = FontWeight.Bold,
        )
    }
}

@Composable
private fun InferenceLogsSection(
    inferenceLogs: AdminInferenceLogsUiState,
    visibleLogs: List<AdminInferenceLogItem>,
    searchQuery: String,
    selectedResult: String,
    selectedModel: String,
    modelFilters: List<String>,
    onSearchChange: (String) -> Unit,
    onResultSelected: (String) -> Unit,
    onModelSelected: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = "Inference Logs",
                style = MaterialTheme.typography.displayLarge,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Real-time health monitoring & diagnostic data.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            InferenceStatCard(
                label = "Avg. Inference Time",
                value = inferenceLogs.averageInferenceTimeText,
                accent = MaterialTheme.colorScheme.primary,
                modifier = Modifier.weight(1f),
            )
            InferenceStatCard(
                label = "Success Rate",
                value = inferenceLogs.successRateText,
                accent = MaterialTheme.colorScheme.secondary,
                modifier = Modifier.weight(1f),
            )
        }

        OutlinedTextField(
            value = searchQuery,
            onValueChange = onSearchChange,
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            shape = RoundedCornerShape(999.dp),
            leadingIcon = {
                Text(
                    text = "S",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.outline,
                    fontWeight = FontWeight.Bold,
                )
            },
            placeholder = {
                Text(
                    text = "Search by Farmer ID",
                    color = MaterialTheme.colorScheme.outline,
                )
            },
        )

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            listOf("All", "Healthy", "Disease", "Warning").forEach { result ->
                ConfigPill(
                    label = result,
                    selected = selectedResult == result,
                    onClick = { onResultSelected(result) },
                )
            }
            modelFilters.forEach { model ->
                ConfigPill(
                    label = model,
                    selected = selectedModel == model,
                    onClick = { onModelSelected(model) },
                )
            }
        }

        if (visibleLogs.isEmpty()) {
            SettingsSectionCard(
                title = "No logs found",
                subtitle = "Saved diagnosis results will appear here.",
                containerColor = CVioSurfaceContainerLow,
            ) {}
        } else {
            visibleLogs.forEach { log ->
                InferenceLogCard(log = log)
            }
        }

        Text(
            text = "Showing ${visibleLogs.size} of ${inferenceLogs.logs.size} logs",
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(CVioSurfaceContainerLow)
                .padding(14.dp),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary,
        )
    }
}

@Composable
private fun InferenceStatCard(
    label: String,
    value: String,
    accent: Color,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(28.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = value,
                style = MaterialTheme.typography.headlineSmall,
                color = accent,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

@Composable
private fun InferenceLogCard(log: AdminInferenceLogItem) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(28.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Text(
                        text = "ID: ${log.farmerId}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = log.timestampText,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline,
                    )
                }
                CVioStatusChip(
                    text = log.resultLabel,
                    containerColor = logKindColor(log.kind).copy(alpha = 0.16f),
                    contentColor = logKindColor(log.kind),
                )
            }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(CVioSurfaceContainerLow)
                    .padding(12.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                LogMetric(
                    label = "Confidence",
                    value = log.confidenceText,
                    modifier = Modifier.weight(1f),
                )
                LogMetric(
                    label = "Time",
                    value = log.inferenceTimeText,
                    modifier = Modifier.weight(1f),
                    alignEnd = true,
                )
            }
            Text(
                text = "${log.farmerName} - ${log.modelName}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun LogMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    alignEnd: Boolean = false,
) {
    Column(
        modifier = modifier,
        horizontalAlignment = if (alignEnd) Alignment.End else Alignment.Start,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(
            text = label.uppercase(),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.outline,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.Bold,
        )
    }
}

@Composable
private fun logKindColor(kind: AdminInferenceLogKind): Color {
    return when (kind) {
        AdminInferenceLogKind.Healthy -> HealthyGreen
        AdminInferenceLogKind.Disease -> DiseaseRed
        AdminInferenceLogKind.Warning -> WarningOrange
    }
}

private fun displayRoleName(role: String): String {
    return when (role) {
        "Farmer" -> "Nông dân"
        "Admin" -> "Quản trị"
        else -> role
    }
}

private fun displayFarmerName(name: String): String {
    return when (name) {
        "Shrimp Farmer" -> "Nông dân nuôi tôm"
        else -> name
    }
}
