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
            text = "Số liệu xử lý",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )
        MetricTileRow {
            CVioMetricTile(
                label = "AI xử lý",
                value = result?.inferenceTimeMs?.let(BenchmarkUtils::latencyText) ?: "Chưa có",
                modifier = Modifier.weight(1f),
            )
            CVioMetricTile(
                label = "Thời gian TB",
                value = "${BenchmarkUtils.decimalText(benchmarkMetrics.averageInferenceTimeMs)} ms",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.tertiary,
            )
        }
        MetricTileRow {
            CVioMetricTile(
                label = "Tốc độ",
                value = result?.speed?.let(BenchmarkUtils::speedText) ?: "Chưa có",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.secondary,
            )
            CVioMetricTile(
                label = "Tốc độ TB",
                value = BenchmarkUtils.speedText(benchmarkMetrics.averageSpeed),
                modifier = Modifier.weight(1f),
            )
        }
        MetricTileRow {
            CVioMetricTile(
                label = "FPS",
                value = result?.fps?.let(BenchmarkUtils::fpsText) ?: "Chưa có",
                modifier = Modifier.weight(1f),
            )
            CVioMetricTile(
                label = "Độ đúng",
                value = benchmarkMetrics.accuracy?.let(BenchmarkUtils::confidenceText) ?: "Chưa có",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.secondary,
            )
        }
        MetricTileRow {
            CVioMetricTile(
                label = "Ngưỡng",
                value = result?.threshold?.let(BenchmarkUtils::confidenceText) ?: "Chưa có",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.primary,
            )
            CVioMetricTile(
                label = "Kappa",
                value = "Chưa có",
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
