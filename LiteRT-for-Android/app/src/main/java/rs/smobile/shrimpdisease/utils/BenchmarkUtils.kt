package rs.smobile.shrimpdisease.utils

import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.data.PredictionItem
import java.util.Locale

object BenchmarkUtils {
    fun calculateInferenceTime(startNanos: Long, endNanos: Long): Long {
        return ((endNanos - startNanos) / 1_000_000L).coerceAtLeast(0L)
    }

    fun calculateSpeed(inferenceTimeMs: Long): Float {
        return if (inferenceTimeMs > 0) 1000f / inferenceTimeMs else 0f
    }

    fun calculateFps(inferenceTimeMs: Long): Float {
        return calculateSpeed(inferenceTimeMs)
    }

    fun calculateAverageTime(timesMs: List<Long>): Float {
        return if (timesMs.isEmpty()) 0f else timesMs.average().toFloat()
    }

    fun calculateAverageSpeed(speeds: List<Float>): Float {
        return if (speeds.isEmpty()) 0f else speeds.average().toFloat()
    }

    fun calculateAverageFps(fpsValues: List<Float>): Float {
        return if (fpsValues.isEmpty()) 0f else fpsValues.average().toFloat()
    }

    fun calculateAccuracy(correctRuns: Int, totalRuns: Int): Float? {
        return if (totalRuns > 0) correctRuns.toFloat() / totalRuns else null
    }

    fun confidenceText(confidence: Float): String {
        return String.format(Locale.US, "%.1f%%", confidence * 100.0f)
    }

    fun decimalText(value: Float): String {
        return String.format(Locale.US, "%.2f", value)
    }

    fun speedText(speed: Float): String {
        return String.format(Locale.US, "%.2f img/s", speed)
    }

    fun fpsText(fps: Float): String {
        return String.format(Locale.US, "%.2f", fps)
    }

    fun latencyText(inferenceTimeMs: Long): String {
        return "$inferenceTimeMs ms"
    }

    fun timestampText(timestampMillis: Long): String {
        return DateTimeUtils.formatTimestamp(timestampMillis)
    }

    fun shapeText(shape: List<Int>): String {
        return shape.joinToString(prefix = "[", postfix = "]")
    }

    fun inputSizeText(modelInfo: ModelInfo): String {
        val shape = modelInfo.input.shape
        return when (modelInfo.inputLayout.name) {
            "NHWC" -> "${shape.getOrNull(2) ?: "?"}x${shape.getOrNull(1) ?: "?"}"
            "NCHW" -> "${shape.getOrNull(3) ?: "?"}x${shape.getOrNull(2) ?: "?"}"
            else -> shapeText(shape)
        }
    }

    fun top3Text(predictions: List<PredictionItem>): String {
        return predictions.joinToString(" | ") { prediction ->
            "${prediction.label} ${confidenceText(prediction.confidence)}"
        }
    }

    fun top3CsvText(predictions: List<PredictionItem>): String {
        return predictions.joinToString("|") { prediction ->
            "${prediction.label}:${String.format(Locale.US, "%.6f", prediction.confidence)}"
        }
    }
}
