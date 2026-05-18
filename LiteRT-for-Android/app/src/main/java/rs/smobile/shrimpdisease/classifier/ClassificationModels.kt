package rs.smobile.shrimpdisease.classifier

/** One model class prediction with a normalized confidence score. */
data class Prediction(
    val label: String,
    val confidence: Float,
)

/** Full inference result returned by the classifier for one input image. */
data class ClassificationResult(
    val top1: Prediction,
    val topK: List<Prediction>,
    val inferenceTimeMs: Double,
)

/** Tensor metadata shown in the diagnostics panel and logs. */
data class TensorInfo(
    val name: String,
    val shape: List<Int>,
    val dataType: String,
    val quantizationScale: Float,
    val quantizationZeroPoint: Int,
)

/** Runtime model metadata used to validate label count, tensor layout, and warnings. */
data class ModelInfo(
    val modelFile: String,
    val labelFile: String,
    val modelFamily: String,
    val input: TensorInfo,
    val output: TensorInfo,
    val inputLayout: InputLayout,
    val outputClassCount: Int,
    val labelsCount: Int,
    val warnings: List<String>,
)
