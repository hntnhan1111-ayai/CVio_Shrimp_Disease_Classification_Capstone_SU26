package org.tensorflow.lite.examples.imageclassification.benchmark

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

data class CvioModel(
    val modelId: String,
    val displayName: String,
    val architecture: String,
    val format: String,
    val precision: String,
    val assetPath: String,
    val runtime: String,
    val supported: Boolean,
    val modelSizeMb: Double,
    val expectedInputWidth: Int,
    val expectedInputHeight: Int,
    val expectedNumClasses: Int,
    val labelsFile: String,
    val inputLayout: String,
    val normalizationMean: List<Float>,
    val normalizationStd: List<Float>,
    val outputSoftmaxApplied: Boolean,
    val flops: String?,
    val notes: String
) {
    val selectorLabel: String
        get() = displayName

    val formatLabel: String
        get() = format.uppercase(Locale.US)
}

data class CvioModelRegistry(
    val models: List<CvioModel>,
    val labels: List<String>
) {
    companion object {
        private const val REGISTRY_ASSET = "cvio_model_registry.json"

        fun load(context: Context): CvioModelRegistry {
            val registryText = context.assets.open(REGISTRY_ASSET).bufferedReader().use {
                it.readText()
            }
            val modelsJson = JSONArray(registryText)
            val models = buildList {
                for (i in 0 until modelsJson.length()) {
                    add(modelsJson.getJSONObject(i).toCvioModel())
                }
            }
            val labelsFile = models.firstOrNull()?.labelsFile ?: "cvio_labels.txt"
            val labels = context.assets.open(labelsFile).bufferedReader().useLines { lines ->
                lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
            }
            return CvioModelRegistry(models = models, labels = labels)
        }

        private fun JSONObject.toCvioModel(): CvioModel {
            val inputSize = getJSONArray("expected_input_size")
            return CvioModel(
                modelId = getString("model_id"),
                displayName = getString("display_name"),
                architecture = getString("architecture"),
                format = getString("format"),
                precision = getString("precision"),
                assetPath = getString("asset_path"),
                runtime = getString("runtime"),
                supported = getBoolean("supported"),
                modelSizeMb = getDouble("model_size_mb"),
                expectedInputWidth = inputSize.getInt(0),
                expectedInputHeight = inputSize.getInt(1),
                expectedNumClasses = getInt("expected_num_classes"),
                labelsFile = getString("labels_file"),
                inputLayout = getString("input_layout"),
                normalizationMean = getFloatList("normalization_mean"),
                normalizationStd = getFloatList("normalization_std"),
                outputSoftmaxApplied = getBoolean("output_softmax_applied"),
                flops = if (isNull("flops")) null else getString("flops"),
                notes = optString("notes", "")
            )
        }

        private fun JSONObject.getFloatList(name: String): List<Float> {
            val array = getJSONArray(name)
            return buildList {
                for (i in 0 until array.length()) {
                    add(array.getDouble(i).toFloat())
                }
            }
        }
    }
}

enum class ImageSource(val label: String) {
    CAMERA("camera"),
    UPLOAD("upload")
}

data class Prediction(
    val index: Int,
    val label: String,
    val confidence: Float
)

data class BenchmarkMetrics(
    val timestamp: String,
    val deviceModel: String,
    val androidVersion: String,
    val appVersion: String,
    val modelId: String,
    val modelName: String,
    val format: String,
    val precision: String,
    val runtime: String,
    val delegate: String,
    val modelSizeMb: Double,
    val inputWidth: Int,
    val inputHeight: Int,
    val preprocessMs: Double,
    val inferenceMs: Double,
    val postprocessMs: Double,
    val totalMs: Double,
    val fps: Double,
    val flops: String?,
    val top1Label: String,
    val top1Confidence: Float,
    val topKPredictions: List<Prediction>,
    val groundTruthLabel: String?,
    val correct: Boolean?,
    val sessionAccuracy: Double?,
    val source: ImageSource
) {
    val inputSizeLabel: String
        get() = "${inputWidth}x${inputHeight}"

    val flopsLabel: String
        get() = flops ?: "N/A"

    val accuracyLabel: String
        get() = sessionAccuracy?.let { String.format(Locale.US, "%.2f%%", it * 100.0) } ?: "N/A"

    val correctnessLabel: String
        get() = correct?.toString() ?: "N/A"
}

data class ClassificationResult(
    val model: CvioModel,
    val predictions: List<Prediction>,
    val metrics: BenchmarkMetrics,
    val logFilePath: String?
)
