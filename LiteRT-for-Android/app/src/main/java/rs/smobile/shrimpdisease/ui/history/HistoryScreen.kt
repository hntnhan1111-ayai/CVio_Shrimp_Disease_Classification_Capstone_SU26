package rs.smobile.shrimpdisease.ui.history

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.ui.components.BenchmarkSummaryCard
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
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Prediction logs",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    enabled = logs.isNotEmpty(),
                    onClick = onClearLogs,
                ) {
                    Text(text = "Clear logs")
                }
                Button(
                    enabled = logs.isNotEmpty(),
                    onClick = onExportCsv,
                ) {
                    Text(text = "Export CSV")
                }
            }
        }

        BenchmarkSummaryCard(metrics = benchmarkMetrics)

        if (logs.isEmpty()) {
            EmptyState(
                title = "No prediction logs",
                message = "Run inference to log latency, speed, FPS, threshold, and accuracy data.",
                modifier = Modifier.weight(1f),
            )
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                items(
                    items = logs,
                    key = { item -> item.id },
                ) { item ->
                    PredictionLogItemCard(item = item)
                }
            }
        }
    }
}
