package rs.smobile.shrimpdisease.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.RuntimeDelegate
import rs.smobile.shrimpdisease.SettingsUiState
import rs.smobile.shrimpdisease.classifier.ModelInfo
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
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.utils.BenchmarkUtils

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
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "Settings",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface,
        )

        if (userDisplayName != null && userRole != null && onLogout != null) {
            SettingsSectionCard(
                title = "Account",
                subtitle = "Signed in session",
                containerColor = CVioSurfaceContainerLow,
            ) {
                CompactInfoRow("Name", userDisplayName)
                CompactInfoRow("Role", userRole)
                SecondaryActionButton(
                    text = "Logout",
                    onClick = onLogout,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        SettingsSectionCard(
            title = "Model Settings",
            subtitle = "Choose the packaged LiteRT/TFLite model used for offline inference.",
        ) {
            ModelSelector(
                selectedModel = modelInfo.modelFile,
                availableModels = availableModels,
                onModelSelected = onSelectModel,
            )
            if (isLoading) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Text(
                    text = "Loading model...",
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
            title = "Runtime",
            subtitle = "On-device classification runtime.",
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
                    label = { Text(text = "GPU unavailable") },
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                CVioMetricTile(
                    label = "Input size",
                    value = BenchmarkUtils.inputSizeText(modelInfo),
                    modifier = Modifier.weight(1f),
                )
                CVioMetricTile(
                    label = "Threads",
                    value = "4",
                    modifier = Modifier.weight(1f),
                    accent = MaterialTheme.colorScheme.secondary,
                )
            }
        }

        SettingsSectionCard(
            title = "Confidence Threshold",
            subtitle = "Predictions below the threshold are shown as low confidence.",
            containerColor = CVioSurfaceContainerLow,
        ) {
            ThresholdSlider(
                threshold = settingsState.confidenceThreshold,
                onThresholdChange = onThresholdChange,
            )
        }

        SettingsSectionCard(
            title = "Debug",
            subtitle = "Technical information stays hidden from the farmer flow unless enabled.",
        ) {
            DataPermissionToggle(
                title = "Show debug info",
                message = "Display source, model tensors, delegate, FPS, and pipeline timings on the diagnosis screen.",
                checked = settingsState.showDebugInfo,
                onCheckedChange = onShowDebugInfoChange,
            )
            CVioStatusChip(
                text = if (settingsState.showDebugInfo) "Debug details enabled" else "Debug details hidden",
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        BenchmarkSummaryCard(metrics = benchmarkMetrics)

        SecondaryActionButton(
            text = "Reset metrics/logs",
            enabled = benchmarkMetrics.totalRuns > 0,
            onClick = onResetMetrics,
            modifier = Modifier.fillMaxWidth(),
        )

        SettingsSectionCard(
            title = "Model Info",
            subtitle = "Tensor metadata and label contract.",
        ) {
            CompactInfoRow("Model name", modelInfo.modelFile)
            CompactInfoRow("Model family", modelInfo.modelFamily)
            CompactInfoRow("Input shape", BenchmarkUtils.shapeText(modelInfo.input.shape))
            CompactInfoRow("Output shape", BenchmarkUtils.shapeText(modelInfo.output.shape))
            CompactInfoRow("Classes", modelInfo.outputClassCount.toString())
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
