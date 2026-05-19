package rs.smobile.shrimpdisease.ui.history

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.ui.components.BenchmarkSummaryCard
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.components.EmptyState
import rs.smobile.shrimpdisease.ui.components.PredictionLogItemCard

@Composable
fun HistoryScreen(
    logs: List<PredictionLogItem>,
    benchmarkMetrics: BenchmarkMetrics,
    onExportCsv: () -> Unit,
    onClearLogs: () -> Unit,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            CVioSectionHeader(
                title = "Diagnosis History",
                subtitle = "Review past health assessments and track pond trends.",
            )
        }

        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CVioStatusChip(text = "All")
                CVioStatusChip(
                    text = "Healthy",
                    containerColor = MaterialTheme.colorScheme.surface,
                    contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                CVioStatusChip(
                    text = "Disease",
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer,
                )
            }
        }

        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                OutlinedButton(
                    enabled = logs.isNotEmpty(),
                    onClick = onClearLogs,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Clear")
                }
                Button(
                    enabled = logs.isNotEmpty(),
                    onClick = onExportCsv,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Export CSV")
                }
            }
        }

        item {
            BenchmarkSummaryCard(metrics = benchmarkMetrics)
        }

        if (logs.isEmpty()) {
            item {
                CVioCard {
                    EmptyState(
                        title = "No prediction logs",
                        message = "Run inference to log latency, speed, FPS, threshold, and accuracy data.",
                    )
                }
            }
        } else {
            items(
                items = logs,
                key = { item -> item.id },
            ) { item ->
                PredictionLogItemCard(item = item)
            }
        }
    }
}
