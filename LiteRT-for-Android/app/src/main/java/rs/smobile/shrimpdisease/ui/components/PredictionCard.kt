package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.classifier.ClassificationResult
import rs.smobile.shrimpdisease.utils.BenchmarkUtils

@Composable
fun PredictionCard(
    result: ClassificationResult?,
    isLoading: Boolean,
    errorMessage: String?,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier) {
        Text(
            text = "Scan Result",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
        )
        when {
            isLoading -> {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Text(
                    text = "Running inference...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            errorMessage != null -> {
                CVioStatusChip(
                    text = "Analysis failed",
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer,
                )
                Text(
                    text = errorMessage,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }

            result != null -> {
                val statusColor = statusColor(result)
                val statusContainer = statusContainerColor(result)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.Top,
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        CVioStatusChip(
                            text = if (result.isAboveThreshold) "Status: ${result.displayPredictionText()}" else "Low confidence",
                            containerColor = statusContainer,
                            contentColor = statusColor,
                        )
                        Text(
                            text = result.displayPredictionText(),
                            style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                    Column(horizontalAlignment = Alignment.End) {
                        Text(
                            text = BenchmarkUtils.confidenceText(result.confidence),
                            style = MaterialTheme.typography.displayLarge,
                            color = statusColor,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = "Confidence",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            color = MaterialTheme.colorScheme.surfaceVariant,
                            shape = RoundedCornerShape(24.dp),
                        )
                        .padding(16.dp),
                ) {
                    Text(
                        text = resultExplanation(result),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    CVioMetricTile(
                        label = "Latency",
                        value = BenchmarkUtils.latencyText(result.inferenceTimeMs),
                        modifier = Modifier.weight(1f),
                    )
                    CVioMetricTile(
                        label = "FPS",
                        value = BenchmarkUtils.fpsText(result.fps),
                        modifier = Modifier.weight(1f),
                        accent = MaterialTheme.colorScheme.secondary,
                    )
                }

                Text(
                    text = "Model ${result.modelName} | Threshold ${BenchmarkUtils.confidenceText(result.threshold)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (result.groundTruthLabel != null) {
                    Text(
                        text = "Ground truth ${result.groundTruthLabel} | ${correctnessText(result.isCorrect)}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            else -> {
                Text(
                    text = "No result yet.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

private fun statusColor(result: ClassificationResult): Color {
    val label = result.displayPredictionText().lowercase()
    return when {
        !result.isAboveThreshold -> Color(0xFF743B24)
        "healthy" in label -> Color(0xFF236863)
        else -> Color(0xFFBA1A1A)
    }
}

@Composable
private fun statusContainerColor(result: ClassificationResult): Color {
    val label = result.displayPredictionText().lowercase()
    return when {
        !result.isAboveThreshold -> MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.25f)
        "healthy" in label -> MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f)
        else -> MaterialTheme.colorScheme.errorContainer
    }
}

private fun resultExplanation(result: ClassificationResult): String {
    return if (result.isAboveThreshold) {
        "The model detected ${result.predictedClass} with enough confidence for field triage. Confirm with pond conditions and recent behavior before acting."
    } else {
        "The top prediction did not pass the confidence threshold. Retake the image with better lighting or review the top-3 breakdown."
    }
}

private fun ClassificationResult.displayPredictionText(): String {
    return if (isAboveThreshold) {
        predictedClass
    } else {
        "$rawTop1Label / Low confidence"
    }
}

private fun correctnessText(isCorrect: Boolean?): String {
    return when (isCorrect) {
        true -> "Correct"
        false -> "Incorrect"
        null -> "N/A"
    }
}
