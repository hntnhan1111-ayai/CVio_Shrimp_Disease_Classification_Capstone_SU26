package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
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
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Prediction",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
            when {
                isLoading -> {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(
                        text = "Running inference...",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }

                errorMessage != null -> {
                    Text(
                        text = errorMessage,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }

                result != null -> {
                    Text(
                        text = "Prediction: ${result.displayPredictionText()}",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "Confidence ${BenchmarkUtils.confidenceText(result.confidence)}",
                        style = MaterialTheme.typography.bodyLarge,
                    )
                    Text(
                        text = "Threshold ${BenchmarkUtils.confidenceText(result.threshold)} - ${thresholdStatusText(result.isAboveThreshold)}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (result.isAboveThreshold) {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        } else {
                            MaterialTheme.colorScheme.error
                        },
                    )
                    Text(
                        text = "Model ${result.modelName}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = "Latency ${BenchmarkUtils.latencyText(result.inferenceTimeMs)} | Speed ${BenchmarkUtils.speedText(result.speed)} | FPS ${BenchmarkUtils.fpsText(result.fps)}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    if (result.groundTruthLabel != null) {
                        Text(
                            text = "Ground truth ${result.groundTruthLabel} | ${correctnessText(result.isCorrect)}",
                            style = MaterialTheme.typography.bodyMedium,
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
}

private fun thresholdStatusText(isAboveThreshold: Boolean): String {
    return if (isAboveThreshold) "Passed" else "Low confidence"
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
