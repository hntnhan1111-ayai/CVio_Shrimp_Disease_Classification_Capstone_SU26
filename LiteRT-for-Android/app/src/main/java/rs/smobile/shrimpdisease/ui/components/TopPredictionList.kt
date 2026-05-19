package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.data.PredictionItem
import rs.smobile.shrimpdisease.utils.BenchmarkUtils

@Composable
fun TopPredictionList(
    predictions: List<PredictionItem>,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier) {
        Text(
            text = "Confidence Breakdown",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (predictions.isEmpty()) {
            Text(
                text = "Run inference to see ranked predictions.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodyMedium,
            )
        } else {
            predictions.forEachIndexed { index, prediction ->
                PredictionRow(
                    rank = index + 1,
                    label = prediction.label,
                    confidence = prediction.confidence,
                )
            }
        }
    }
}

@Composable
private fun PredictionRow(
    rank: Int,
    label: String,
    confidence: Float,
) {
    val accent = when {
        rank == 1 && "healthy" !in label.lowercase() -> MaterialTheme.colorScheme.error
        rank == 1 -> MaterialTheme.colorScheme.secondary
        else -> MaterialTheme.colorScheme.primaryContainer
    }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = BenchmarkUtils.confidenceText(confidence),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        LinearProgressIndicator(
            progress = { confidence.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(),
            color = accent,
            trackColor = MaterialTheme.colorScheme.surfaceVariant,
        )
    }
}
