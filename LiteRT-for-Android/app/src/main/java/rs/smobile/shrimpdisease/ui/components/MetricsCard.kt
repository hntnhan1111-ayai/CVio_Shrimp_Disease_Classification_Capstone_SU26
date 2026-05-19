package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
        Text(
            text = "Performance Metrics",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )
        MetricTileRow {
            CVioMetricTile(
                label = "Inference",
                value = result?.inferenceTimeMs?.let(BenchmarkUtils::latencyText) ?: "N/A",
                modifier = Modifier.weight(1f),
            )
            CVioMetricTile(
                label = "Avg time",
                value = "${BenchmarkUtils.decimalText(benchmarkMetrics.averageInferenceTimeMs)} ms",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.tertiary,
            )
        }
        MetricTileRow {
            CVioMetricTile(
                label = "Speed",
                value = result?.speed?.let(BenchmarkUtils::speedText) ?: "N/A",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.secondary,
            )
            CVioMetricTile(
                label = "Avg speed",
                value = BenchmarkUtils.speedText(benchmarkMetrics.averageSpeed),
                modifier = Modifier.weight(1f),
            )
        }
        MetricTileRow {
            CVioMetricTile(
                label = "FPS",
                value = result?.fps?.let(BenchmarkUtils::fpsText) ?: "N/A",
                modifier = Modifier.weight(1f),
            )
            CVioMetricTile(
                label = "Accuracy",
                value = benchmarkMetrics.accuracy?.let(BenchmarkUtils::confidenceText) ?: "N/A",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.secondary,
            )
        }
        MetricTileRow {
            CVioMetricTile(
                label = "Threshold",
                value = result?.threshold?.let(BenchmarkUtils::confidenceText) ?: "N/A",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.primary,
            )
            CVioMetricTile(
                label = "Kappa",
                value = "N/A",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun MetricTileRow(content: @Composable RowScope.() -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        content = content,
    )
}
