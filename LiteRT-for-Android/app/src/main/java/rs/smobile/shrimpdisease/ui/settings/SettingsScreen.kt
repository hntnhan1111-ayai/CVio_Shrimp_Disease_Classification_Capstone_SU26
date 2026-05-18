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
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
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
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    text = "Select model",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
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
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    text = "Runtime",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
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
                SettingRow("Input size", BenchmarkUtils.inputSizeText(modelInfo))
                SettingRow("Threads", "4")
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
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
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text(
                        text = "Show debug info",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "Adds model tensors and camera FPS on the inference screen.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(
                    checked = settingsState.showDebugInfo,
                    onCheckedChange = onShowDebugInfoChange,
                )
            }
        }

        BenchmarkSummaryCard(metrics = benchmarkMetrics)

        Button(
            enabled = benchmarkMetrics.totalRuns > 0,
            onClick = onResetMetrics,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(text = "Reset metrics/logs")
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "Model info",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                )
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
