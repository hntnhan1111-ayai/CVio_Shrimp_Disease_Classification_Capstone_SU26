package rs.smobile.shrimpdisease.classifier

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
