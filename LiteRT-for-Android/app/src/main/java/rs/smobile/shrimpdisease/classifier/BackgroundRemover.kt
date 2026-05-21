package rs.smobile.shrimpdisease.classifier

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.scale
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import rs.smobile.shrimpdisease.loadModelFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Removes the image background with the packaged U2Net/rembg LiteRT model.
 * If the model cannot be loaded or invoked, the app falls back to the previous
 * edge-connected color heuristic so classification can still continue.
 */
class BackgroundRemover(
    private val assetManager: AssetManager,
    private val modelFile: String = ModelDefaults.BACKGROUND_REMOVER_MODEL_FILE,
) {
    private var segmenter: U2NetBackgroundSegmenter? = null
    private var modelLoadFailed = false

    @Synchronized
    fun removeBackground(
        bitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
    ): IntArray {
        val loadedSegmenter = segmenterOrNull()
        if (loadedSegmenter != null) {
            try {
                return loadedSegmenter.removeBackground(bitmap, targetWidth, targetHeight)
            } catch (e: Exception) {
                Log.e(TAG, "U2Net background removal failed, using color fallback.", e)
            }
        }

        val pixels = bitmap.resizedPixels(targetWidth, targetHeight, filter = false)
        return EdgeBackgroundRemover.removeBackground(pixels, targetWidth, targetHeight)
    }

    private fun segmenterOrNull(): U2NetBackgroundSegmenter? {
        if (modelLoadFailed) return null
        segmenter?.let { return it }

        return try {
            U2NetBackgroundSegmenter(assetManager, modelFile).also { segmenter = it }
        } catch (e: Exception) {
            modelLoadFailed = true
            Log.e(TAG, "Cannot load U2Net background remover model: $modelFile", e)
            null
        }
    }

    private companion object {
        private const val TAG = "BackgroundRemover"
    }
}

private class U2NetBackgroundSegmenter(
    assetManager: AssetManager,
    modelFile: String,
) {
    private val interpreter: InterpreterApi = InterpreterApi.create(
        assetManager.loadModelFile(modelFile),
        InterpreterApi.Options()
            .setNumThreads(ModelDefaults.NUM_THREADS)
            .setUseXNNPACK(ModelDefaults.USE_XNNPACK)
            .setUseNNAPI(ModelDefaults.USE_NNAPI),
    )
    private val inputBuffer: ByteBuffer
    private val outputBuffers: MutableMap<Int, Any>
    private val inputWidth: Int
    private val inputHeight: Int
    private val inputLayout: InputLayout
    private val inputDataType: DataType
    private val inputQuantizationScale: Float
    private val inputQuantizationZeroPoint: Int
    private val maskWidth: Int
    private val maskHeight: Int
    private val maskLayout: InputLayout
    private val maskDataType: DataType
    private val maskQuantizationScale: Float
    private val maskQuantizationZeroPoint: Int

    init {
        interpreter.allocateTensors()

        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        inputLayout = resolveImageLayout(inputShape)
        inputHeight = when (inputLayout) {
            InputLayout.NHWC -> inputShape[1]
            InputLayout.NCHW -> inputShape[2]
        }
        inputWidth = when (inputLayout) {
            InputLayout.NHWC -> inputShape[2]
            InputLayout.NCHW -> inputShape[3]
        }
        inputDataType = inputTensor.dataType()
        val inputQuantization = inputTensor.quantizationParams()
        inputQuantizationScale = inputQuantization.getScale()
        inputQuantizationZeroPoint = inputQuantization.getZeroPoint()
        inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())

        val firstOutputTensor = interpreter.getOutputTensor(PRIMARY_OUTPUT_INDEX)
        val outputShape = firstOutputTensor.shape()
        maskLayout = resolveMaskLayout(outputShape)
        maskHeight = when (maskLayout) {
            InputLayout.NHWC -> outputShape[1]
            InputLayout.NCHW -> outputShape[2]
        }
        maskWidth = when (maskLayout) {
            InputLayout.NHWC -> outputShape[2]
            InputLayout.NCHW -> outputShape[3]
        }
        maskDataType = firstOutputTensor.dataType()
        val maskQuantization = firstOutputTensor.quantizationParams()
        maskQuantizationScale = maskQuantization.getScale()
        maskQuantizationZeroPoint = maskQuantization.getZeroPoint()
        outputBuffers = (0 until interpreter.getOutputTensorCount())
            .associateWith { outputIndex ->
                ByteBuffer.allocateDirect(interpreter.getOutputTensor(outputIndex).numBytes())
                    .order(ByteOrder.nativeOrder()) as Any
            }
            .toMutableMap()
    }

    fun removeBackground(
        bitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
    ): IntArray {
        if (targetWidth <= 0 || targetHeight <= 0) return IntArray(0)

        val sourcePixels = bitmap.resizedPixels(targetWidth, targetHeight, filter = false)
        writeInput(bitmap)
        runModel()

        val mask = readNormalizedPrimaryMask()
        return applyMask(sourcePixels, targetWidth, targetHeight, mask)
    }

    private fun writeInput(bitmap: Bitmap) {
        val pixels = bitmap.resizedPixels(inputWidth, inputHeight, filter = true)
        val channelDivisor = pixels.maxRgbChannel().coerceAtLeast(1).toFloat()

        inputBuffer.rewind()
        when (inputLayout) {
            InputLayout.NHWC -> {
                for (pixel in pixels) {
                    putInputValue(normalizeRembg(pixel, RED_CHANNEL, channelDivisor))
                    putInputValue(normalizeRembg(pixel, GREEN_CHANNEL, channelDivisor))
                    putInputValue(normalizeRembg(pixel, BLUE_CHANNEL, channelDivisor))
                }
            }

            InputLayout.NCHW -> {
                for (channel in 0 until RGB_CHANNELS) {
                    for (pixel in pixels) {
                        putInputValue(normalizeRembg(pixel, channel, channelDivisor))
                    }
                }
            }
        }
        inputBuffer.rewind()
    }

    private fun runModel() {
        outputBuffers.values.forEach { output ->
            (output as ByteBuffer).rewind()
        }
        interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputBuffer), outputBuffers)
        outputBuffers.values.forEach { output ->
            (output as ByteBuffer).rewind()
        }
    }

    private fun readNormalizedPrimaryMask(): FloatArray {
        val primaryOutputBuffer = outputBuffers.getValue(PRIMARY_OUTPUT_INDEX) as ByteBuffer
        primaryOutputBuffer.rewind()

        val rawMask = FloatArray(maskWidth * maskHeight)
        var minValue = Float.POSITIVE_INFINITY
        var maxValue = Float.NEGATIVE_INFINITY

        fun remember(value: Float, index: Int) {
            rawMask[index] = value
            if (value < minValue) minValue = value
            if (value > maxValue) maxValue = value
        }

        when (maskLayout) {
            InputLayout.NHWC -> {
                for (index in rawMask.indices) {
                    remember(readMaskValue(primaryOutputBuffer), index)
                }
            }

            InputLayout.NCHW -> {
                for (index in rawMask.indices) {
                    remember(readMaskValue(primaryOutputBuffer), index)
                }
            }
        }

        val range = maxValue - minValue
        if (!range.isFinite() || range <= MIN_MASK_RANGE) {
            val fill = if (maxValue > 0f) 1f else 0f
            for (index in rawMask.indices) rawMask[index] = fill
            return rawMask
        }

        for (index in rawMask.indices) {
            rawMask[index] = ((rawMask[index] - minValue) / range).coerceIn(0f, 1f)
        }
        return rawMask
    }

    private fun applyMask(
        sourcePixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        mask: FloatArray,
    ): IntArray {
        val output = sourcePixels.copyOf()
        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val index = y * targetWidth + x
                val alpha = sampleMask(mask, x, y, targetWidth, targetHeight)
                output[index] = compositeOnBlack(sourcePixels[index], alpha)
            }
        }
        return output
    }

    private fun sampleMask(
        mask: FloatArray,
        x: Int,
        y: Int,
        targetWidth: Int,
        targetHeight: Int,
    ): Float {
        if (maskWidth <= 1 || maskHeight <= 1 || targetWidth <= 1 || targetHeight <= 1) {
            return mask.firstOrNull() ?: 0f
        }

        val sourceX = x * (maskWidth - 1f) / (targetWidth - 1f)
        val sourceY = y * (maskHeight - 1f) / (targetHeight - 1f)
        val x0 = sourceX.toInt().coerceIn(0, maskWidth - 1)
        val y0 = sourceY.toInt().coerceIn(0, maskHeight - 1)
        val x1 = (x0 + 1).coerceAtMost(maskWidth - 1)
        val y1 = (y0 + 1).coerceAtMost(maskHeight - 1)
        val dx = sourceX - x0
        val dy = sourceY - y0

        val top = lerp(mask[y0 * maskWidth + x0], mask[y0 * maskWidth + x1], dx)
        val bottom = lerp(mask[y1 * maskWidth + x0], mask[y1 * maskWidth + x1], dx)
        return lerp(top, bottom, dy).coerceIn(0f, 1f)
    }

    private fun putInputValue(value: Float) {
        when (inputDataType) {
            DataType.FLOAT32 -> inputBuffer.putFloat(value)
            DataType.UINT8 -> inputBuffer.put(quantize(value, inputQuantizationScale, inputQuantizationZeroPoint, 0, 255).toByte())
            DataType.INT8 -> inputBuffer.put(quantize(value, inputQuantizationScale, inputQuantizationZeroPoint, -128, 127).toByte())
            else -> error("Unsupported U2Net input tensor dtype: ${inputDataType.name}")
        }
    }

    private fun readMaskValue(buffer: ByteBuffer): Float {
        return when (maskDataType) {
            DataType.FLOAT32 -> buffer.getFloat()
            DataType.UINT8 -> dequantize(buffer.get().toInt() and 0xFF, maskQuantizationScale, maskQuantizationZeroPoint)
            DataType.INT8 -> dequantize(buffer.get().toInt(), maskQuantizationScale, maskQuantizationZeroPoint)
            else -> error("Unsupported U2Net output tensor dtype: ${maskDataType.name}")
        }
    }

    private fun normalizeRembg(pixel: Int, channel: Int, channelDivisor: Float): Float {
        val raw = rawRgbChannel(pixel, channel) / channelDivisor
        return (raw - REMBG_MEAN[channel]) / REMBG_STD[channel]
    }

    private fun compositeOnBlack(pixel: Int, alpha: Float): Int {
        val safeAlpha = alpha.coerceIn(0f, 1f)
        val red = (rawRgbChannel(pixel, RED_CHANNEL) * safeAlpha).roundToInt().coerceIn(0, 255)
        val green = (rawRgbChannel(pixel, GREEN_CHANNEL) * safeAlpha).roundToInt().coerceIn(0, 255)
        val blue = (rawRgbChannel(pixel, BLUE_CHANNEL) * safeAlpha).roundToInt().coerceIn(0, 255)
        return Color.rgb(red, green, blue)
    }

    private fun quantize(value: Float, scale: Float, zeroPoint: Int, min: Int, max: Int): Int {
        require(scale > 0f) {
            "Quantized U2Net tensor requires a positive quantization scale."
        }
        return (value / scale + zeroPoint).roundToInt().coerceIn(min, max)
    }

    private fun dequantize(value: Int, scale: Float, zeroPoint: Int): Float {
        return if (scale > 0f) {
            (value - zeroPoint) * scale
        } else {
            value.toFloat()
        }
    }

    private fun resolveImageLayout(shape: IntArray): InputLayout {
        require(shape.size == 4) {
            "Expected 4D U2Net image input tensor, got shape=${shape.contentToString()}"
        }
        return when {
            shape[3] == RGB_CHANNELS -> InputLayout.NHWC
            shape[1] == RGB_CHANNELS -> InputLayout.NCHW
            else -> error("Cannot infer U2Net RGB layout from shape=${shape.contentToString()}")
        }
    }

    private fun resolveMaskLayout(shape: IntArray): InputLayout {
        require(shape.size == 4) {
            "Expected 4D U2Net mask output tensor, got shape=${shape.contentToString()}"
        }
        return when {
            shape[3] == MASK_CHANNELS -> InputLayout.NHWC
            shape[1] == MASK_CHANNELS -> InputLayout.NCHW
            else -> error("Cannot infer U2Net mask layout from shape=${shape.contentToString()}")
        }
    }

    private companion object {
        private const val PRIMARY_OUTPUT_INDEX = 0
        private const val RGB_CHANNELS = 3
        private const val MASK_CHANNELS = 1
        private const val RED_CHANNEL = 0
        private const val GREEN_CHANNEL = 1
        private const val BLUE_CHANNEL = 2
        private const val MIN_MASK_RANGE = 1e-6f
        private val REMBG_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val REMBG_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}

internal object EdgeBackgroundRemover {
    fun removeBackground(
        pixels: IntArray,
        width: Int,
        height: Int,
    ): IntArray {
        if (pixels.isEmpty() || width <= 0 || height <= 0) return pixels

        val background = estimateBorderColor(pixels, width, height)
        val threshold = estimateThreshold(pixels, width, height, background)
        val backgroundMask = BooleanArray(pixels.size)
        val queue = IntArray(pixels.size)
        var head = 0
        var tail = 0

        fun enqueue(index: Int) {
            if (index !in pixels.indices || backgroundMask[index]) return
            if (!isBackgroundLike(pixels[index], background, threshold)) return
            backgroundMask[index] = true
            queue[tail++] = index
        }

        for (x in 0 until width) {
            enqueue(x)
            enqueue((height - 1) * width + x)
        }
        for (y in 0 until height) {
            enqueue(y * width)
            enqueue(y * width + width - 1)
        }

        while (head < tail) {
            val index = queue[head++]
            val x = index % width
            val y = index / width

            if (x > 0) enqueue(index - 1)
            if (x < width - 1) enqueue(index + 1)
            if (y > 0) enqueue(index - width)
            if (y < height - 1) enqueue(index + width)
        }

        val output = pixels.copyOf()
        for (index in output.indices) {
            if (backgroundMask[index]) {
                output[index] = BLACK_RGB
            }
        }
        return output
    }

    private fun estimateBorderColor(
        pixels: IntArray,
        width: Int,
        height: Int,
    ): RgbColor {
        var rSum = 0f
        var gSum = 0f
        var bSum = 0f
        var count = 0f

        fun add(pixel: Int) {
            rSum += red(pixel)
            gSum += green(pixel)
            bSum += blue(pixel)
            count += 1f
        }

        for (x in 0 until width) {
            add(pixels[x])
            add(pixels[(height - 1) * width + x])
        }
        for (y in 0 until height) {
            add(pixels[y * width])
            add(pixels[y * width + width - 1])
        }

        val safeCount = count.coerceAtLeast(1f)
        return RgbColor(
            r = rSum / safeCount,
            g = gSum / safeCount,
            b = bSum / safeCount,
        )
    }

    private fun estimateThreshold(
        pixels: IntArray,
        width: Int,
        height: Int,
        background: RgbColor,
    ): Float {
        val borderDistances = ArrayList<Float>((width + height) * 2)

        fun add(pixel: Int) {
            borderDistances.add(colorDistance(pixel, background))
        }

        for (x in 0 until width) {
            add(pixels[x])
            add(pixels[(height - 1) * width + x])
        }
        for (y in 0 until height) {
            add(pixels[y * width])
            add(pixels[y * width + width - 1])
        }

        val mean = borderDistances.average().toFloat()
        val variance = borderDistances
            .map { distance -> (distance - mean) * (distance - mean) }
            .average()
            .toFloat()
        val std = sqrt(variance)
        return (mean + 2.5f * std + 18f).coerceIn(MIN_THRESHOLD, MAX_THRESHOLD)
    }

    private fun isBackgroundLike(pixel: Int, background: RgbColor, threshold: Float): Boolean {
        return colorDistance(pixel, background) <= threshold
    }

    private fun colorDistance(pixel: Int, background: RgbColor): Float {
        val dr = red(pixel) - background.r
        val dg = green(pixel) - background.g
        val db = blue(pixel) - background.b
        return sqrt(dr * dr + dg * dg + db * db)
    }

    private fun red(pixel: Int): Float = ((pixel shr 16) and 0xFF).toFloat()

    private fun green(pixel: Int): Float = ((pixel shr 8) and 0xFF).toFloat()

    private fun blue(pixel: Int): Float = (pixel and 0xFF).toFloat()

    private data class RgbColor(
        val r: Float,
        val g: Float,
        val b: Float,
    )

    private const val BLACK_RGB = -0x1000000
    private const val MIN_THRESHOLD = 28f
    private const val MAX_THRESHOLD = 95f
}

private fun Bitmap.resizedPixels(width: Int, height: Int, filter: Boolean): IntArray {
    val resizedBitmap = if (this.width == width && this.height == height) {
        this
    } else {
        scale(width, height, filter)
    }

    val pixels = IntArray(width * height)
    resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
    if (resizedBitmap !== this) {
        resizedBitmap.recycle()
    }
    return pixels
}

private fun IntArray.maxRgbChannel(): Int {
    var maxValue = 0
    for (pixel in this) {
        maxValue = max(maxValue, rawRgbChannel(pixel, 0))
        maxValue = max(maxValue, rawRgbChannel(pixel, 1))
        maxValue = max(maxValue, rawRgbChannel(pixel, 2))
    }
    return maxValue
}

private fun rawRgbChannel(pixel: Int, channel: Int): Int {
    return when (channel) {
        0 -> pixel shr 16 and 0xFF
        1 -> pixel shr 8 and 0xFF
        else -> pixel and 0xFF
    }
}

private fun lerp(start: Float, end: Float, amount: Float): Float {
    return start + (end - start) * amount
}
