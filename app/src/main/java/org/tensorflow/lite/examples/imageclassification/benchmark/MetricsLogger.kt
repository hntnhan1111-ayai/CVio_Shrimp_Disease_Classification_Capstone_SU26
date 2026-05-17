package org.tensorflow.lite.examples.imageclassification.benchmark

import android.content.Context
import android.util.Log
import java.io.File
import java.util.Locale

class MetricsLogger(private val context: Context) {
    private val logDirectory: File by lazy {
        File(context.getExternalFilesDir(null), LOG_DIR).also { it.mkdirs() }
    }

    private val logFile: File by lazy {
        File(logDirectory, LOG_FILE_NAME)
    }

    fun append(result: ClassificationResult): File {
        if (!logFile.exists()) {
            logFile.appendText(CSV_HEADER + "\n")
        }
        logFile.appendText(result.metrics.toCsvRow() + "\n")
        Log.i(TAG, "Saved metrics row: ${logFile.absolutePath}")
        return logFile
    }

    fun logDirectoryPath(): String = logDirectory.absolutePath

    private fun BenchmarkMetrics.toCsvRow(): String {
        val topK = topKPredictions.joinToString("|") {
            "${it.index}:${it.label}:${String.format(Locale.US, "%.6f", it.confidence)}"
        }
        val values = listOf(
            timestamp,
            deviceModel,
            androidVersion,
            appVersion,
            modelId,
            modelName,
            format,
            precision,
            runtime,
            delegate,
            String.format(Locale.US, "%.3f", modelSizeMb),
            inputWidth.toString(),
            inputHeight.toString(),
            String.format(Locale.US, "%.3f", preprocessMs),
            String.format(Locale.US, "%.3f", inferenceMs),
            String.format(Locale.US, "%.3f", postprocessMs),
            String.format(Locale.US, "%.3f", totalMs),
            String.format(Locale.US, "%.3f", fps),
            flops ?: "N/A",
            top1Label,
            String.format(Locale.US, "%.6f", top1Confidence),
            topK,
            groundTruthLabel ?: "N/A",
            correct?.toString() ?: "N/A",
            sessionAccuracy?.let { String.format(Locale.US, "%.6f", it) } ?: "N/A",
            source.label
        )
        return values.joinToString(",") { it.csvEscape() }
    }

    private fun String.csvEscape(): String {
        val escaped = replace("\"", "\"\"")
        return "\"$escaped\""
    }

    companion object {
        private const val TAG = "CVioMetrics"
        private const val LOG_DIR = "cvio_metrics"
        private const val LOG_FILE_NAME = "inference_metrics.csv"
        private const val CSV_HEADER =
            "timestamp,device_model,android_version,app_version,model_id,model_name,format," +
                "precision,runtime,delegate,model_size_mb,input_width,input_height," +
                "preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,flops,top1_label," +
                "top1_confidence,topk_predictions,ground_truth_label_optional," +
                "correct_optional,session_accuracy_optional,source"
    }
}
