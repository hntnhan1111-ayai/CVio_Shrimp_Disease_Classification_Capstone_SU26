package rs.smobile.shrimpdisease.classifier

import android.graphics.Bitmap
import android.util.Log
import androidx.core.graphics.scale
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Converts Android bitmaps into the exact tensor layout, dtype, and normalization
 * expected by the selected .tflite model.
 */
class ImagePreprocessor {
    var debugEnabled: Boolean = false

    fun preprocess(bitmap: Bitmap, config: ModelConfig): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(config.inputBufferByteSize())
            .order(ByteOrder.nativeOrder())
        return preprocess(bitmap, config, buffer)
    }

    fun preprocess(bitmap: Bitmap, config: ModelConfig, inputBuffer: ByteBuffer): ByteBuffer {
        inputBuffer.rewind()
        val stats = when (config.modelFamily) {
            ModelFamily.YOLO_ULTRALYTICS -> preprocessYolo(bitmap, config, inputBuffer)
            ModelFamily.PYTORCH_IMAGENET -> when (config.inputLayout) {
                InputLayout.NHWC -> preprocessPyTorchImageNetNhwc(bitmap, config, inputBuffer)
                InputLayout.NCHW -> preprocessPyTorchImageNetNchw(bitmap, config, inputBuffer)
            }
        }
        inputBuffer.rewind()

        if (debugEnabled) {
            logPreprocessDebug(config, inputBuffer, stats)
        }

        return inputBuffer
    }

    fun preprocessYolo(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        val config = ModelConfig(
            modelName = "debug_yolo",
            inputSize = inputSize,
            modelFamily = ModelFamily.YOLO_ULTRALYTICS,
            inputLayout = InputLayout.NHWC,
            inputDataType = DataType.FLOAT32,
        )
        return preprocess(bitmap, config)
    }

    fun preprocessPyTorchImageNetNHWC(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        val config = ModelConfig(
            modelName = "debug_pytorch_imagenet_nhwc",
            inputSize = inputSize,
            modelFamily = ModelFamily.PYTORCH_IMAGENET,
            inputLayout = InputLayout.NHWC,
            inputDataType = DataType.FLOAT32,
        )
        return preprocess(bitmap, config)
    }

    fun preprocessPyTorchImageNetNCHW(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        val config = ModelConfig(
            modelName = "debug_pytorch_imagenet_nchw",
            inputSize = inputSize,
            modelFamily = ModelFamily.PYTORCH_IMAGENET,
            inputLayout = InputLayout.NCHW,
            inputDataType = DataType.FLOAT32,
        )
        return preprocess(bitmap, config)
    }

    fun firstInputValues(inputBuffer: ByteBuffer, config: ModelConfig, count: Int = DEBUG_VALUE_COUNT): List<Float> {
        val duplicate = inputBuffer.duplicate().order(ByteOrder.nativeOrder())
        duplicate.rewind()
        val limit = count.coerceAtLeast(0)
        return buildList {
            repeat(limit) {
                if (duplicate.remaining() < config.inputDataType.elementByteSize()) return@buildList
                add(readValue(duplicate, config.inputDataType))
            }
        }
    }

    fun logFirstInputValues(inputBuffer: ByteBuffer, config: ModelConfig, count: Int = DEBUG_VALUE_COUNT) {
        Log.d(TAG, "first_${count}_input_values=${firstInputValues(inputBuffer, config, count)}")
    }

    private fun preprocessYolo(
        bitmap: Bitmap,
        config: ModelConfig,
        inputBuffer: ByteBuffer,
    ): PreprocessStats {
        val pixels = resizedPixels(bitmap, config.inputWidth, config.inputHeight)
        return when (config.inputLayout) {
            InputLayout.NHWC -> writeNhwc(pixels, inputBuffer, config, normalizer = ::normalizeYolo)
            InputLayout.NCHW -> writeNchw(pixels, inputBuffer, config, normalizer = ::normalizeYolo)
        }
    }

    private fun preprocessPyTorchImageNetNhwc(
        bitmap: Bitmap,
        config: ModelConfig,
        inputBuffer: ByteBuffer,
    ): PreprocessStats {
        val pixels = resizedPixels(bitmap, config.inputWidth, config.inputHeight)
        return writeNhwc(pixels, inputBuffer, config, normalizer = ::normalizeImageNet)
    }

    private fun preprocessPyTorchImageNetNchw(
        bitmap: Bitmap,
        config: ModelConfig,
        inputBuffer: ByteBuffer,
    ): PreprocessStats {
        val pixels = resizedPixels(bitmap, config.inputWidth, config.inputHeight)
        return writeNchw(pixels, inputBuffer, config, normalizer = ::normalizeImageNet)
    }

    private fun resizedPixels(bitmap: Bitmap, inputWidth: Int, inputHeight: Int): IntArray {
        val resizedBitmap = if (bitmap.width == inputWidth && bitmap.height == inputHeight) {
            bitmap
        } else {
            bitmap.scale(inputWidth, inputHeight, false)
        }

        val pixels = IntArray(inputWidth * inputHeight)
        resizedBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }
        return BackgroundRemover.removeBackground(pixels, inputWidth, inputHeight)
    }

    private fun writeNhwc(
        pixels: IntArray,
        buffer: ByteBuffer,
        config: ModelConfig,
        normalizer: (Int, Int) -> Float,
    ): PreprocessStats {
        val stats = PreprocessStats()
        for (pixel in pixels) {
            putValue(buffer, normalizer(pixel, RED_CHANNEL), config, stats)
            putValue(buffer, normalizer(pixel, GREEN_CHANNEL), config, stats)
            putValue(buffer, normalizer(pixel, BLUE_CHANNEL), config, stats)
        }
        return stats
    }

    private fun writeNchw(
        pixels: IntArray,
        buffer: ByteBuffer,
        config: ModelConfig,
        normalizer: (Int, Int) -> Float,
    ): PreprocessStats {
        val stats = PreprocessStats()
        for (channel in 0 until RGB_CHANNELS) {
            for (pixel in pixels) {
                putValue(buffer, normalizer(pixel, channel), config, stats)
            }
        }
        return stats
    }

    private fun normalizeYolo(pixel: Int, channel: Int): Float {
        return rawRgbChannel(pixel, channel)
    }

    private fun normalizeImageNet(pixel: Int, channel: Int): Float {
        val raw = rawRgbChannel(pixel, channel)
        return (raw - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
    }

    private fun rawRgbChannel(pixel: Int, channel: Int): Float {
        return when (channel) {
            RED_CHANNEL -> pixel shr 16 and 0xFF
            GREEN_CHANNEL -> pixel shr 8 and 0xFF
            else -> pixel and 0xFF
        } / 255.0f
    }

    private fun putValue(
        buffer: ByteBuffer,
        value: Float,
        config: ModelConfig,
        stats: PreprocessStats,
    ) {
        stats.add(value)
        when (config.inputDataType) {
            DataType.FLOAT32 -> buffer.putFloat(value)
            DataType.UINT8 -> buffer.put(quantize(value, config, 0, 255).toByte())
            DataType.INT8 -> buffer.put(quantize(value, config, -128, 127).toByte())
            else -> error("Unsupported input tensor dtype: ${config.inputDataType.name}")
        }
    }

    private fun quantize(value: Float, config: ModelConfig, min: Int, max: Int): Int {
        require(config.quantizationScale > 0f) {
            "Quantized input tensor requires a positive quantization scale."
        }
        return (value / config.quantizationScale + config.quantizationZeroPoint)
            .roundToInt()
            .coerceIn(min, max)
    }

    private fun logPreprocessDebug(
        config: ModelConfig,
        inputBuffer: ByteBuffer,
        stats: PreprocessStats,
    ) {
        val summary = String.format(
            Locale.US,
            "Preprocess model=%s family=%s layout=%s inputSize=%dx%d shape=%s dtype=%s bufferCapacity=%d min=%.6f max=%.6f mean=%.6f",
            config.modelName,
            config.modelFamily,
            config.inputLayout,
            config.inputWidth,
            config.inputHeight,
            config.inputShape,
            config.inputDataType,
            inputBuffer.capacity(),
            stats.min,
            stats.max,
            stats.mean(),
        )
        Log.d(TAG, summary)
        logFirstInputValues(inputBuffer, config)
    }

    private fun readValue(buffer: ByteBuffer, dataType: DataType): Float {
        return when (dataType) {
            DataType.FLOAT32 -> buffer.getFloat()
            DataType.UINT8 -> (buffer.get().toInt() and 0xFF).toFloat()
            DataType.INT8 -> buffer.get().toFloat()
            else -> error("Unsupported debug input tensor dtype: ${dataType.name}")
        }
    }

    private fun DataType.elementByteSize(): Int {
        return when (this) {
            DataType.FLOAT32 -> 4
            DataType.UINT8, DataType.INT8 -> 1
            else -> error("Unsupported input tensor dtype: $name")
        }
    }

    private fun ModelConfig.inputBufferByteSize(): Int {
        return 1 * inputWidth * inputHeight * RGB_CHANNELS * inputDataType.elementByteSize()
    }

    private class PreprocessStats {
        var min: Float = Float.POSITIVE_INFINITY
            private set
        var max: Float = Float.NEGATIVE_INFINITY
            private set
        private var sum = 0.0
        private var count = 0

        fun add(value: Float) {
            if (value < min) min = value
            if (value > max) max = value
            sum += value
            count += 1
        }

        fun mean(): Float = if (count == 0) 0f else (sum / count).toFloat()
    }

    private companion object {
        private const val TAG = "ImagePreprocessor"
        private const val RGB_CHANNELS = 3
        private const val RED_CHANNEL = 0
        private const val GREEN_CHANNEL = 1
        private const val BLUE_CHANNEL = 2
        private const val DEBUG_VALUE_COUNT = 10
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
