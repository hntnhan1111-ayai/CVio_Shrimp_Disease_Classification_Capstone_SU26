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
            text = "Kết quả kiểm tra",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
        )
        when {
            isLoading -> {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Text(
                    text = "Đang phân tích ảnh tôm...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            errorMessage != null -> {
                ResultStatusBadge(
                    text = "Phân tích chưa thành công",
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
                        label = "Độ tin cậy",
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
                        text = "Ý nghĩa kết quả",
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
                        label = "Thời gian",
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
                    text = "Mô hình ${result.modelName} | Ngưỡng ${BenchmarkUtils.confidenceText(result.threshold)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                result.groundTruthLabel?.let { groundTruth ->
                    Text(
                        text = "Nhãn kiểm tra $groundTruth | ${correctnessText(result.isCorrect)}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            else -> {
                Text(
                    text = "Hãy kiểm tra ảnh để xem đánh giá sức khỏe tôm.",
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
        !isAboveThreshold -> "Ảnh chưa đủ rõ"
        "healthy" in label -> "Tình trạng: Khỏe"
        else -> "Phát hiện dấu hiệu bệnh"
    }
}

private fun resultExplanation(result: ClassificationResult): String {
    return if (result.isAboveThreshold) {
        "AI dự đoán mẫu này là ${result.predictedClass}. Bà con nên xem đây là gợi ý ban đầu và đối chiếu thêm tình trạng ao."
    } else {
        "Dự đoán cao nhất chưa đủ tin cậy. Bà con nên chụp lại ảnh rõ hơn trước khi đánh giá."
    }
}

private fun ClassificationResult.displayPredictionText(): String {
    return if (isAboveThreshold) {
        predictedClass
    } else {
        "$rawTop1Label / Chưa rõ"
    }
}

private fun correctnessText(isCorrect: Boolean?): String {
    return when (isCorrect) {
        true -> "Đúng"
        false -> "Chưa đúng"
        null -> "Chưa có"
    }
}
