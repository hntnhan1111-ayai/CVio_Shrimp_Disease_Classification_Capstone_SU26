package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.classifier.ClassificationResult
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.ui.theme.WarningOrange
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
                    text = "Analyzing shrimp image...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            errorMessage != null -> {
                ResultStatusBadge(
                    text = "Analysis failed",
                    kind = ResultStatusKind.Disease,
                )
                Text(
                    text = errorMessage,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }

            result != null -> {
                val statusKind = result.statusKind()
                val statusAccent = result.statusAccent()
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        ResultStatusBadge(
                            text = result.statusText(),
                            kind = statusKind,
                        )
                        Text(
                            text = result.displayPredictionText(),
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                    ConfidenceScoreCard(
                        label = "Confidence",
                        value = BenchmarkUtils.confidenceText(result.confidence),
                        accent = statusAccent,
                        modifier = Modifier.weight(0.78f),
                    )
                }

                LinearProgressIndicator(
                    progress = { result.confidence.coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                    color = statusAccent,
                    trackColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                )

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            color = CVioSurfaceContainerLow,
                            shape = RoundedCornerShape(22.dp),
                        )
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(
                        text = "What this means",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.Bold,
                    )
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
                        label = "Time",
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
                result.groundTruthLabel?.let { groundTruth ->
                    Text(
                        text = "Ground truth $groundTruth | ${correctnessText(result.isCorrect)}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            else -> {
                Text(
                    text = "Run inference to see a shrimp health assessment.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

private fun ClassificationResult.statusKind(): ResultStatusKind {
    val label = displayPredictionText().lowercase()
    return when {
        !isAboveThreshold -> ResultStatusKind.Warning
        "healthy" in label -> ResultStatusKind.Healthy
        else -> ResultStatusKind.Disease
    }
}

private fun ClassificationResult.statusAccent(): Color {
    val label = displayPredictionText().lowercase()
    return when {
        !isAboveThreshold -> WarningOrange
        "healthy" in label -> HealthyGreen
        else -> DiseaseRed
    }
}

private fun ClassificationResult.statusText(): String {
    val label = displayPredictionText().lowercase()
    return when {
        !isAboveThreshold -> "Low confidence"
        "healthy" in label -> "Status: Healthy"
        else -> "Disease detected"
    }
}

private fun resultExplanation(result: ClassificationResult): String {
    return if (result.isAboveThreshold) {
        "The AI classified this sample as ${result.predictedClass}. Use this as field triage and confirm with pond conditions before treatment decisions."
    } else {
        "The top prediction is below the confidence threshold. Retake the image with better lighting or inspect the top-3 prediction breakdown."
    }
}

private fun ClassificationResult.displayPredictionText(): String {
    return if (isAboveThreshold) {
        predictedClass
    } else {
        "$rawTop1Label / Unknown"
    }
}

private fun correctnessText(isCorrect: Boolean?): String {
    return when (isCorrect) {
        true -> "Correct"
        false -> "Incorrect"
        null -> "N/A"
    }
}
