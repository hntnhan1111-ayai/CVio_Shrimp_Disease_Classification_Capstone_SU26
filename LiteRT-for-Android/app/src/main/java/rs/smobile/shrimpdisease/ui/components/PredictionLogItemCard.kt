package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainer
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.ui.theme.WarningOrange
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import rs.smobile.shrimpdisease.utils.DateTimeUtils

@Composable
fun PredictionLogItemCard(
    item: PredictionLogItem,
    modifier: Modifier = Modifier,
) {
    val status = rememberHistoryStatus(item)
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(32.dp),
        colors = CardDefaults.cardColors(containerColor = CVioSurfaceContainerLowest),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
        border = BorderStroke(1.dp, status.borderColor),
    ) {
        Box(modifier = Modifier.fillMaxWidth()) {
            if (status.kind == HistoryStatusKind.Disease) {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .size(128.dp)
                        .clip(RoundedCornerShape(bottomStart = 999.dp))
                        .background(MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.2f)),
                )
            }

            Row(
                modifier = Modifier.padding(12.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                PredictionThumbnail(item = item, status = status)
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Top,
                    ) {
                        HistoryStatusPill(status = status)
                        Text(
                            text = DateTimeUtils.formatTimestamp(item.timestamp),
                            modifier = Modifier
                                .weight(1f)
                                .padding(start = 8.dp),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.outline,
                        )
                    }
                    Text(
                        text = item.predictedClass,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = status.metricPrefix,
                            style = MaterialTheme.typography.bodyMedium,
                            color = status.contentColor,
                            fontWeight = if (status.kind == HistoryStatusKind.Healthy) {
                                FontWeight.Normal
                            } else {
                                FontWeight.Medium
                            },
                        )
                        Text(
                            text = status.metricText,
                            style = MaterialTheme.typography.bodyMedium,
                            color = status.contentColor,
                            fontWeight = if (status.kind == HistoryStatusKind.Healthy) {
                                FontWeight.Normal
                            } else {
                                FontWeight.Medium
                            },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun HistoryStatusPill(status: HistoryStatus) {
    Surface(
        shape = RoundedCornerShape(999.dp),
        color = status.containerColor,
        contentColor = status.contentColor,
        border = if (status.kind == HistoryStatusKind.LowConfidence) {
            BorderStroke(1.dp, WarningOrange.copy(alpha = 0.22f))
        } else {
            null
        },
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(status.dotColor),
            )
            Text(
                text = status.label,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun PredictionThumbnail(
    item: PredictionLogItem,
    status: HistoryStatus,
) {
    val modifier = Modifier
        .size(96.dp)
        .clip(RoundedCornerShape(16.dp))
        .background(MaterialTheme.colorScheme.surfaceVariant)
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
                modifier = modifier,
                contentAlignment = Alignment.Center,
            ) {
                if (status.kind == HistoryStatusKind.LowConfidence) {
                    Text(
                        text = "No image",
                        style = MaterialTheme.typography.labelSmall,
                        color = WarningOrange,
                        fontWeight = FontWeight.Bold,
                    )
                } else {
                    ShrimpIllustration(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(10.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun rememberHistoryStatus(item: PredictionLogItem): HistoryStatus {
    val isHealthy = item.predictedClass.contains("healthy", ignoreCase = true)
    return when {
        !item.isAboveThreshold -> HistoryStatus(
            label = "Blurry Image",
            metricPrefix = "",
            metricText = "Please Retake",
            kind = HistoryStatusKind.LowConfidence,
            containerColor = WarningOrange.copy(alpha = 0.16f),
            contentColor = WarningOrange,
            dotColor = WarningOrange.copy(alpha = 0.58f),
            borderColor = WarningOrange.copy(alpha = 0.28f),
        )

        isHealthy -> HistoryStatus(
            label = "Healthy",
            metricPrefix = "",
            metricText = "${BenchmarkUtils.confidenceText(item.confidence)} Confidence",
            kind = HistoryStatusKind.Healthy,
            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f),
            contentColor = HealthyGreen,
            dotColor = HealthyGreen,
            borderColor = CVioSurfaceContainer,
        )

        else -> HistoryStatus(
            label = "${item.predictedClass} Detected",
            metricPrefix = "",
            metricText = "${BenchmarkUtils.confidenceText(item.confidence)} Confidence",
            kind = HistoryStatusKind.Disease,
            containerColor = MaterialTheme.colorScheme.errorContainer,
            contentColor = DiseaseRed,
            dotColor = DiseaseRed,
            borderColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.32f),
        )
    }
}

private data class HistoryStatus(
    val label: String,
    val metricPrefix: String,
    val metricText: String,
    val kind: HistoryStatusKind,
    val containerColor: Color,
    val contentColor: Color,
    val dotColor: Color,
    val borderColor: Color,
)

private enum class HistoryStatusKind {
    Healthy,
    Disease,
    LowConfidence,
}
