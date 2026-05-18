package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import rs.smobile.shrimpdisease.utils.DateTimeUtils

@Composable
fun PredictionLogItemCard(
    item: PredictionLogItem,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.Top,
        ) {
            PredictionThumbnail(item = item)
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = item.predictedClass,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Confidence ${BenchmarkUtils.confidenceText(item.confidence)} | ${BenchmarkUtils.latencyText(item.inferenceTimeMs)}",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Text(
                    text = "FPS ${BenchmarkUtils.fpsText(item.fps)} | ${BenchmarkUtils.speedText(item.speed)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = "${item.modelName} | Threshold ${BenchmarkUtils.confidenceText(item.threshold)}",
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = "Ground truth ${item.groundTruthLabel ?: "N/A"} | ${correctnessText(item)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = DateTimeUtils.formatTimestamp(item.timestamp),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun PredictionThumbnail(item: PredictionLogItem) {
    val modifier = Modifier.size(72.dp)
    when {
        item.thumbnail != null -> {
            Image(
                bitmap = item.thumbnail.asImageBitmap(),
                contentDescription = "Prediction thumbnail",
                modifier = modifier,
                contentScale = ContentScale.Crop,
            )
        }

        item.imageUri != null -> {
            AsyncImage(
                model = item.imageUri,
                contentDescription = "Prediction thumbnail",
                modifier = modifier,
                contentScale = ContentScale.Crop,
            )
        }

        else -> {
            Box(
                modifier = modifier.background(MaterialTheme.colorScheme.surfaceVariant),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "No image",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

private fun correctnessText(item: PredictionLogItem): String {
    return when (item.isCorrect) {
        true -> "Correct"
        false -> "Incorrect"
        null -> "N/A"
    }
}
