package rs.smobile.shrimpdisease.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.RuntimeDelegate
import rs.smobile.shrimpdisease.SettingsUiState
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.ui.components.BenchmarkSummaryCard
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioIconBubble
import rs.smobile.shrimpdisease.ui.components.CVioMetricTile
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.components.ModelSelector
import rs.smobile.shrimpdisease.ui.components.ThresholdSlider
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
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        ProfileCard()

        CVioCard {
            CVioSectionHeader(
                title = "Data Permissions",
                subtitle = "Local diagnostics controls for field testing.",
            )
            SettingSwitchRow(
                title = "Detailed diagnostics",
                message = "Show tensors, camera FPS, and pipeline timing on the inference screen.",
                checked = settingsState.showDebugInfo,
                onCheckedChange = onShowDebugInfoChange,
            )
            CVioStatusChip(
                text = if (settingsState.showDebugInfo) "Diagnostics details enabled" else "Diagnostics details hidden",
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        CVioCard {
            CVioSectionHeader(
                title = "Select model",
                subtitle = "Choose the packaged TFLite model used for offline inference.",
            )
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

        CVioCard {
            CVioSectionHeader(
                title = "Runtime",
                subtitle = "On-device classification settings.",
            )
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
                    label = "Input",
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

        CVioCard {
            ThresholdSlider(
                threshold = settingsState.confidenceThreshold,
                onThresholdChange = onThresholdChange,
            )
            Text(
                text = "Predictions below this value are shown as Unknown / Low confidence while Top-3 remains visible.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        BenchmarkSummaryCard(metrics = benchmarkMetrics)

        Button(
            enabled = benchmarkMetrics.totalRuns > 0,
            onClick = onResetMetrics,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(text = "Reset metrics/logs")
        }

        CVioCard {
            CVioSectionHeader(title = "Model info")
            SettingRow("Model name", modelInfo.modelFile)
            SettingRow("Model family", modelInfo.modelFamily)
            SettingRow("Input shape", BenchmarkUtils.shapeText(modelInfo.input.shape))
            SettingRow("Output shape", BenchmarkUtils.shapeText(modelInfo.output.shape))
            SettingRow("Classes", modelInfo.outputClassCount.toString())
            SettingRow("Labels", labels.joinToString(", "))
            if (modelInfo.warnings.isNotEmpty()) {
                modelInfo.warnings.forEach { warning ->
                    Text(
                        text = warning,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

@Composable
private fun ProfileCard() {
    CVioCard {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            CVioIconBubble(
                label = "F",
                size = 80.dp,
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
            )
            Text(
                text = "CVio Farmer",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Coastal farm profile | offline diagnostics",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun SettingSwitchRow(
    title: String,
    message: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
        )
    }
}

@Composable
private fun SettingRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(0.8f),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            modifier = Modifier
                .weight(1.2f)
                .padding(start = 12.dp),
            maxLines = 3,
            overflow = TextOverflow.Ellipsis,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
        )
    }
}
