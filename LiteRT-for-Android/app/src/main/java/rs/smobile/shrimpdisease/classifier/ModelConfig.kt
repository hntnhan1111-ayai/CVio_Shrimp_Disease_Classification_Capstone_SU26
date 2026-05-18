package rs.smobile.shrimpdisease.classifier

/**
 * Central runtime configuration for the packaged shrimp disease classifier.
 * Keep MODEL_FILE and LABEL_FILE aligned with files under app/src/main/assets.
 */
object ModelConfig {
    const val MODEL_FILE = "efficientnet_b0_float16.tflite"
    const val LABEL_FILE = "labels.txt"

    // Change to "yolo_ultralytics" when using a YOLO classification export.
    const val MODEL_FAMILY = "efficientnet_imagenet"

    const val INPUT_SIZE = 224
    const val NUM_THREADS = 4
    const val TOP_K = 3
    const val CONFIDENCE_THRESHOLD = 0.60f

    const val USE_NNAPI = false
    const val USE_XNNPACK = true
}

/** Supported preprocessing profiles for the exported classification models. */
enum class ModelFamily {
    EFFICIENTNET_IMAGENET,
    YOLO_ULTRALYTICS;

    companion object {
        fun fromConfig(value: String): ModelFamily = when (value.lowercase()) {
            "efficientnet_imagenet", "pytorch_imagenet" -> EFFICIENTNET_IMAGENET
            "yolo_ultralytics" -> YOLO_ULTRALYTICS
            else -> error("Unsupported MODEL_FAMILY: $value")
        }
    }
}

/** Supported 4D image tensor layouts discovered from the LiteRT input tensor. */
enum class InputLayout {
    NHWC,
    NCHW
}
