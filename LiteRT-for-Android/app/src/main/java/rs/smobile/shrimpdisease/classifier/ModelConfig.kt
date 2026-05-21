package rs.smobile.shrimpdisease.classifier

import org.tensorflow.lite.DataType

/**
 * Central runtime defaults for packaged shrimp disease classifiers.
 * Keep DEFAULT_MODEL_FILE and LABEL_FILE aligned with app/src/main/assets.
 */
object ModelDefaults {
    const val DEFAULT_MODEL_FILE = "efficientnet_b0_float16.tflite"
    const val BACKGROUND_REMOVER_MODEL_FILE = "u2net_dynamic_range_int8.tflite"
    const val LABEL_FILE = "labels.txt"

    const val INPUT_SIZE = 224
    const val NUM_THREADS = 4
    const val TOP_K = 3
    const val CONFIDENCE_THRESHOLD = 0.50f

    const val USE_NNAPI = false
    const val USE_XNNPACK = true
}

data class ModelConfig(
    val modelName: String,
    val inputSize: Int,
    val modelFamily: ModelFamily,
    val inputLayout: InputLayout,
    val inputDataType: DataType,
    val inputWidth: Int = inputSize,
    val inputHeight: Int = inputSize,
    val quantizationScale: Float = 0f,
    val quantizationZeroPoint: Int = 0,
    val inputShape: List<Int> = emptyList(),
)

/** Supported preprocessing profiles for the exported classification models. */
enum class ModelFamily {
    YOLO_ULTRALYTICS,
    PYTORCH_IMAGENET;

    companion object {
        fun fromConfig(value: String): ModelFamily = when (value.lowercase()) {
            "efficientnet_imagenet", "pytorch_imagenet", "mobilenet_imagenet" -> PYTORCH_IMAGENET
            "yolo_ultralytics", "ultralytics_yolo", "yolo" -> YOLO_ULTRALYTICS
            else -> error("Unsupported MODEL_FAMILY: $value")
        }
    }
}

/** Supported 4D image tensor layouts discovered from the LiteRT input tensor. */
enum class InputLayout {
    NHWC,
    NCHW,
}
