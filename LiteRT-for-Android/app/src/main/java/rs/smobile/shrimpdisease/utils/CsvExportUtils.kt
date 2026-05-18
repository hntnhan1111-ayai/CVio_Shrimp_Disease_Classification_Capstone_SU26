package rs.smobile.shrimpdisease.utils

import android.content.Context
import android.net.Uri
import android.util.Log
import rs.smobile.shrimpdisease.data.PredictionLogItem
import java.util.Locale

object CsvExportUtils {
    fun exportLogsToUri(
        context: Context,
        uri: Uri,
        logs: List<PredictionLogItem>,
    ): Boolean {
        return try {
            context.contentResolver.openOutputStream(uri)?.bufferedWriter()?.use { writer ->
                writer.appendLine(
                    listOf(
                        "timestamp",
                        "image_uri",
                        "model_name",
                        "predicted_class",
                        "confidence",
                        "top3_predictions",
                        "threshold",
                        "is_above_threshold",
                        "ground_truth_label",
                        "is_correct",
                        "inference_time_ms",
                        "speed_img_per_sec",
                        "fps",
                        "average_inference_time_ms",
                        "average_speed_img_per_sec",
                        "average_fps",
                        "running_accuracy",
                    ).joinToString(",")
                )
                logs.reversed().forEach { item ->
                    writer.appendLine(item.toCsvRow())
                }
            } != null
        } catch (error: Throwable) {
            Log.e(TAG, "Failed to export prediction logs", error)
            false
        }
    }

    private fun PredictionLogItem.toCsvRow(): String {
        return listOf(
            DateTimeUtils.formatTimestamp(timestamp),
            imageUri.orEmpty(),
            modelName,
            predictedClass,
            String.format(Locale.US, "%.6f", confidence),
            top3Predictions,
            String.format(Locale.US, "%.3f", threshold),
            isAboveThreshold.toString(),
            groundTruthLabel.orEmpty(),
            isCorrect?.toString().orEmpty(),
            inferenceTimeMs.toString(),
            String.format(Locale.US, "%.2f", speed),
            String.format(Locale.US, "%.2f", fps),
            String.format(Locale.US, "%.2f", averageInferenceTimeMs),
            String.format(Locale.US, "%.2f", averageSpeed),
            String.format(Locale.US, "%.2f", averageFps),
            runningAccuracy?.let { String.format(Locale.US, "%.3f", it) }.orEmpty(),
        ).joinToString(",") { value -> value.csvEscape() }
    }

    private fun String.csvEscape(): String {
        val escaped = replace("\"", "\"\"")
        return if (any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
            "\"$escaped\""
        } else {
            escaped
        }
    }

    private const val TAG = "CsvExportUtils"
}
