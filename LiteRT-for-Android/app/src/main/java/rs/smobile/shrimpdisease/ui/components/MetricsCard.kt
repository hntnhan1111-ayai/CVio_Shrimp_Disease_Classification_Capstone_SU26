package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.classifier.ClassificationResult
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.utils.BenchmarkUtils

@Composable
fun MetricsCard(
    result: ClassificationResult?,
    benchmarkMetrics: BenchmarkMetrics,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier) {
        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Metrics",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            MetricRow("Inference time", result?.inferenceTimeMs?.let(BenchmarkUtils::latencyText) ?: "N/A")
            MetricRow("Average time", "${BenchmarkUtils.decimalText(benchmarkMetrics.averageInferenceTimeMs)} ms")
            MetricRow("Speed", result?.speed?.let(BenchmarkUtils::speedText) ?: "N/A")
            MetricRow("Average speed", BenchmarkUtils.speedText(benchmarkMetrics.averageSpeed))
            MetricRow("FPS", result?.fps?.let(BenchmarkUtils::fpsText) ?: "N/A")
            MetricRow("Average FPS", BenchmarkUtils.fpsText(benchmarkMetrics.averageFps))
            MetricRow(
                "Accuracy",
                benchmarkMetrics.accuracy?.let { BenchmarkUtils.confidenceText(it) } ?: "N/A",
            )
        }
    }
}

@Composable
fun MetricRow(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(0.9f),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            modifier = Modifier
                .weight(1.1f)
                .padding(start = 12.dp),
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
        )
    }
}
