package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.data.PredictionItem
import rs.smobile.shrimpdisease.utils.DiseaseTextUtils
import java.util.Locale

@Composable
fun TopPredictionList(
    predictions: List<PredictionItem>,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier) {
        Text(
            text = "3 dự đoán hàng đầu",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface,
        )
        if (predictions.isEmpty()) {
            Text(
                text = "Hãy kiểm tra ảnh để xem các dự đoán của AI.",
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
    val normalizedConfidence = confidence.normalizedConfidence()
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
                text = "$rank. ${DiseaseTextUtils.displayLabel(label)}",
                modifier = Modifier.weight(1f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = topPredictionConfidenceText(normalizedConfidence),
                modifier = Modifier.width(52.dp),
                style = MaterialTheme.typography.labelMedium,
                textAlign = TextAlign.End,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        LinearProgressIndicator(
            progress = { normalizedConfidence },
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp),
            color = accent,
            trackColor = MaterialTheme.colorScheme.surfaceVariant,
        )
    }
}

private fun Float.normalizedConfidence(): Float {
    return when {
        isNaN() -> 0f
        this > 1f -> this / 100f
        else -> this
    }.coerceIn(0f, 1f)
}

private fun topPredictionConfidenceText(confidence: Float): String {
    return String.format(Locale.US, "%.0f%%", confidence * 100f)
}
