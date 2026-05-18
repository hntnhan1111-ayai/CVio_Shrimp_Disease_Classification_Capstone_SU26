package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.utils.BenchmarkUtils

@Composable
fun BenchmarkSummaryCard(
    metrics: BenchmarkMetrics,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Benchmark Summary",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
            MetricRow("Total runs", metrics.totalRuns.toString())
            MetricRow("Correct runs", "${metrics.correctRuns}/${metrics.evaluatedRuns}")
            MetricRow("Accuracy", metrics.accuracy?.let(BenchmarkUtils::confidenceText) ?: "N/A")
            MetricRow("Average inference time", "${BenchmarkUtils.decimalText(metrics.averageInferenceTimeMs)} ms")
            MetricRow("Average speed", BenchmarkUtils.speedText(metrics.averageSpeed))
            MetricRow("Average FPS", BenchmarkUtils.fpsText(metrics.averageFps))
        }
    }
}
