package rs.smobile.shrimpdisease.classifier

import android.graphics.Bitmap
import rs.smobile.shrimpdisease.data.PredictionItem

/** One model class prediction with a normalized confidence score. */
data class Prediction(
    val label: String,
    val confidence: Float,
)

/** Full inference and benchmark result returned by the classifier for one input image. */
data class ClassificationResult(
    val predictedClass: String,
    val confidence: Float,
    val top3Predictions: List<PredictionItem>,
    val inferenceTimeMs: Long,
    val speed: Float,
    val fps: Float,
    val modelName: String,
    val threshold: Float,
    val isAboveThreshold: Boolean,
    val timestamp: Long,
    val groundTruthLabel: String? = null,
    val isCorrect: Boolean? = null,
    val preprocessingTimeMs: Long = 0L,
    val modelInferenceTimeMs: Long = inferenceTimeMs,
    val postprocessingTimeMs: Long = 0L,
    val totalTimeMs: Long = inferenceTimeMs,
    val segmentation: SegmentationResult? = null,
) {
    val rawTop1Label: String
        get() = top3Predictions.firstOrNull()?.label ?: predictedClass

    val top1: Prediction
        get() = Prediction(rawTop1Label, confidence)

    val topK: List<Prediction>
        get() = top3Predictions.map { prediction ->
            Prediction(prediction.label, prediction.confidence)
        }
}

data class SegmentationResult(
    val maskBitmap: Bitmap,
    val label: String,
    val confidence: Float,
    val detectionCount: Int,
    val classCounts: Map<String, Int> = emptyMap(),
)
