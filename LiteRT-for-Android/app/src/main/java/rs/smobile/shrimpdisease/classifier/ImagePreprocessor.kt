package rs.smobile.shrimpdisease.classifier

import android.graphics.Bitmap
import androidx.core.graphics.scale
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import kotlin.math.roundToInt

/**
 * Converts Android bitmaps into the exact tensor layout, dtype, and normalization
 * expected by the selected .tflite model.
 */
class ImagePreprocessor(
    private val inputWidth: Int,
    private val inputHeight: Int,
    private val inputLayout: InputLayout,
    private val inputDataType: DataType,
    private val quantizationScale: Float,
    private val quantizationZeroPoint: Int,
    private val modelFamily: ModelFamily,
) {
    private val pixels = IntArray(inputWidth * inputHeight)

    fun preprocess(bitmap: Bitmap, inputBuffer: ByteBuffer): ByteBuffer {
        val resizedBitmap = if (bitmap.width == inputWidth && bitmap.height == inputHeight) {
            bitmap
        } else {
            bitmap.scale(inputWidth, inputHeight, false)
        }

        resizedBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }
        inputBuffer.rewind()

        when (inputLayout) {
            InputLayout.NHWC -> putNhwc(inputBuffer)
            InputLayout.NCHW -> putNchw(inputBuffer)
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    private fun putNhwc(buffer: ByteBuffer) {
        for (pixel in pixels) {
            putValue(buffer, normalizedChannelValue(pixel, 0))
            putValue(buffer, normalizedChannelValue(pixel, 1))
            putValue(buffer, normalizedChannelValue(pixel, 2))
        }
    }

    private fun putNchw(buffer: ByteBuffer) {
        for (channel in 0 until RGB_CHANNELS) {
            for (pixel in pixels) {
                putValue(buffer, normalizedChannelValue(pixel, channel))
            }
        }
    }

    private fun normalizedChannelValue(pixel: Int, channel: Int): Float {
        val raw = when (channel) {
            0 -> pixel shr 16 and 0xFF
            1 -> pixel shr 8 and 0xFF
            else -> pixel and 0xFF
        } / 255.0f

        return when (modelFamily) {
            ModelFamily.EFFICIENTNET_IMAGENET -> (raw - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
            ModelFamily.YOLO_ULTRALYTICS -> raw
        }
    }

    private fun putValue(buffer: ByteBuffer, value: Float) {
        when (inputDataType) {
            DataType.FLOAT32 -> buffer.putFloat(value)
            DataType.UINT8 -> buffer.put(quantize(value, 0, 255).toByte())
            DataType.INT8 -> buffer.put(quantize(value, -128, 127).toByte())
            else -> error("Unsupported input tensor dtype: ${inputDataType.name}")
        }
    }

    private fun quantize(value: Float, min: Int, max: Int): Int {
        require(quantizationScale > 0f) {
            "Quantized input tensor requires a positive quantization scale."
        }
        return (value / quantizationScale + quantizationZeroPoint)
            .roundToInt()
            .coerceIn(min, max)
    }

    private companion object {
        private const val RGB_CHANNELS = 3
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
