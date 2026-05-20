package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
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
    CVioCard(modifier = modifier) {
        Text(
            text = "Tóm tắt kiểm tra",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            CVioMetricTile(
                label = "Số lần",
                value = metrics.totalRuns.toString(),
                modifier = Modifier.weight(1f),
            )
            CVioMetricTile(
                label = "Độ đúng",
                value = metrics.accuracy?.let(BenchmarkUtils::confidenceText) ?: "Chưa có",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.secondary,
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            CVioMetricTile(
                label = "Thời gian TB",
                value = "${BenchmarkUtils.decimalText(metrics.averageInferenceTimeMs)} ms",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.tertiary,
            )
            CVioMetricTile(
                label = "FPS TB",
                value = BenchmarkUtils.fpsText(metrics.averageFps),
                modifier = Modifier.weight(1f),
            )
        }
        Text(
            text = "Đúng ${metrics.correctRuns}/${metrics.evaluatedRuns} lần | Tốc độ TB ${BenchmarkUtils.speedText(metrics.averageSpeed)}",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
