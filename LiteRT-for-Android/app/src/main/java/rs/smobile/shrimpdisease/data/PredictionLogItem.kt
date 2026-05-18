package rs.smobile.shrimpdisease.data

import android.graphics.Bitmap

data class PredictionLogItem(
    val imageUri: String?,
    val predictedClass: String,
    val confidence: Float,
    val top3Predictions: String,
    val inferenceTimeMs: Long,
    val speed: Float,
    val fps: Float,
    val modelName: String,
    val threshold: Float,
    val isAboveThreshold: Boolean,
    val groundTruthLabel: String?,
    val isCorrect: Boolean?,
    val timestamp: Long,
    val thumbnail: Bitmap? = null,
    val averageInferenceTimeMs: Float = 0f,
    val averageSpeed: Float = 0f,
    val averageFps: Float = 0f,
    val runningAccuracy: Float? = null,
) {
    val id: Long = timestamp
}
