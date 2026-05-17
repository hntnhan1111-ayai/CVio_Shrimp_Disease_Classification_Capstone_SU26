package org.tensorflow.lite.examples.imageclassification.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class TfliteClassifierRunner(
    private val context: Context,
    val model: CvioModel,
    numThreads: Int,
    delegate: Int
) : Closeable {
    private var gpuDelegate: GpuDelegate? = null
    private val interpreter: Interpreter
    private val inputTensor: Tensor
    private val outputTensor: Tensor
    private val inputShape: IntArray
    private val outputShape: IntArray
    private val inputWidth: Int
    private val inputHeight: Int
    private val channelCount: Int
    private val isNchw: Boolean
    private var warmedUp = false

    init {
        val options = Interpreter.Options().setNumThreads(numThreads)
        when (delegate) {
            DELEGATE_GPU -> {
                val compatibilityList = CompatibilityList()
                if (!compatibilityList.isDelegateSupportedOnThisDevice) {
                    throw IllegalStateException("GPU delegate is not supported on this device")
                }
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            }
            DELEGATE_NNAPI -> options.setUseNNAPI(true)
        }
        interpreter = Interpreter(loadModelFile(context, model.assetPath), options)
        inputTensor = interpreter.getInputTensor(0)
        outputTensor = interpreter.getOutputTensor(0)
        inputShape = inputTensor.shape()
        outputShape = outputTensor.shape()

        isNchw = inputShape.size == 4 && inputShape[1] == 3
        inputHeight = if (isNchw) inputShape[2] else inputShape[1]
        inputWidth = if (isNchw) inputShape[3] else inputShape[2]
        channelCount = if (isNchw) inputShape[1] else inputShape[3]
        require(channelCount == 3) {
            "Expected 3-channel image input but ${model.displayName} has shape ${inputShape.contentToString()}"
        }
    }

    fun classify(
        bitmap: Bitmap,
        rotationDegrees: Int,
        labels: List<String>,
        threshold: Float,
        maxResults: Int,
        benchmarkRuns: Int
    ): RunnerOutput {
        val preprocessStart = SystemClock.elapsedRealtimeNanos()
        val inputBuffer = preprocess(bitmap, rotationDegrees)
        val preprocessMs = elapsedMs(preprocessStart)

        if (!warmedUp) {
            repeat(WARMUP_RUNS) {
                runInterpreter(inputBuffer, allocateOutputBuffer())
            }
            warmedUp = true
        }

        val outputBuffer = allocateOutputBuffer()
        val measuredRuns = max(1, benchmarkRuns)
        val inferenceStart = SystemClock.elapsedRealtimeNanos()
        repeat(measuredRuns) {
            runInterpreter(inputBuffer, outputBuffer)
        }
        val inferenceMs = elapsedMs(inferenceStart) / measuredRuns.toDouble()

        val postprocessStart = SystemClock.elapsedRealtimeNanos()
        val probabilities = readProbabilities(outputBuffer)
        val predictions = probabilities
            .withIndex()
            .sortedByDescending { it.value }
            .filter { it.value >= threshold }
            .take(maxResults)
            .map { indexedValue ->
                Prediction(
                    index = indexedValue.index,
                    label = labels.getOrElse(indexedValue.index) { "class_${indexedValue.index}" },
                    confidence = indexedValue.value
                )
            }
        val postprocessMs = elapsedMs(postprocessStart)

        return RunnerOutput(
            predictions = predictions,
            inputWidth = inputWidth,
            inputHeight = inputHeight,
            preprocessMs = preprocessMs,
            inferenceMs = inferenceMs,
            postprocessMs = postprocessMs
        )
    }

    fun inputSizeLabel(): String = "${inputWidth}x${inputHeight}"

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    private fun runInterpreter(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
    }

    private fun allocateOutputBuffer(): ByteBuffer {
        return ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
    }

    private fun preprocess(bitmap: Bitmap, rotationDegrees: Int): ByteBuffer {
        val rotated = rotateBitmap(bitmap, rotationDegrees)
        val cropped = centerCrop(rotated, inputWidth, inputHeight)
        val scaled = if (cropped.width == inputWidth && cropped.height == inputHeight) {
            cropped
        } else {
            Bitmap.createScaledBitmap(cropped, inputWidth, inputHeight, true)
        }

        val inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
        if (isNchw) {
            fillNchw(scaled, inputBuffer)
        } else {
            fillNhwc(scaled, inputBuffer)
        }

        if (scaled !== cropped) scaled.recycle()
        if (cropped !== rotated && cropped !== bitmap) cropped.recycle()
        if (rotated !== bitmap) rotated.recycle()

        inputBuffer.rewind()
        return inputBuffer
    }

    private fun fillNhwc(bitmap: Bitmap, inputBuffer: ByteBuffer) {
        val pixels = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        for (pixel in pixels) {
            putChannel(inputBuffer, Color.red(pixel), 0)
            putChannel(inputBuffer, Color.green(pixel), 1)
            putChannel(inputBuffer, Color.blue(pixel), 2)
        }
    }

    private fun fillNchw(bitmap: Bitmap, inputBuffer: ByteBuffer) {
        val pixels = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        for (channel in 0 until channelCount) {
            for (pixel in pixels) {
                val value = when (channel) {
                    0 -> Color.red(pixel)
                    1 -> Color.green(pixel)
                    else -> Color.blue(pixel)
                }
                putChannel(inputBuffer, value, channel)
            }
        }
    }

    private fun putChannel(buffer: ByteBuffer, channelValue: Int, channelIndex: Int) {
        val scaled = channelValue / 255.0f
        val normalized = (scaled - model.normalizationMean[channelIndex]) /
            model.normalizationStd[channelIndex]
        when (inputTensor.dataType()) {
            DataType.FLOAT32 -> buffer.putFloat(normalized)
            DataType.INT8 -> {
                val quant = inputTensor.quantizationParams()
                val value = (normalized / quant.scale + quant.zeroPoint).roundToInt()
                    .coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt())
                buffer.put(value.toByte())
            }
            DataType.UINT8 -> {
                val quant = inputTensor.quantizationParams()
                val value = (normalized / quant.scale + quant.zeroPoint).roundToInt()
                    .coerceIn(0, 255)
                buffer.put(value.toByte())
            }
            else -> throw IllegalStateException(
                "Unsupported input tensor type ${inputTensor.dataType()} for ${model.displayName}"
            )
        }
    }

    private fun readProbabilities(outputBuffer: ByteBuffer): FloatArray {
        val values = FloatArray(outputTensor.numElements())
        when (outputTensor.dataType()) {
            DataType.FLOAT32 -> outputBuffer.asFloatBuffer().get(values)
            DataType.INT8 -> {
                val quant = outputTensor.quantizationParams()
                for (i in values.indices) {
                    values[i] = (outputBuffer.get().toInt() - quant.zeroPoint) * quant.scale
                }
            }
            DataType.UINT8 -> {
                val quant = outputTensor.quantizationParams()
                for (i in values.indices) {
                    values[i] = ((outputBuffer.get().toInt() and 0xFF) - quant.zeroPoint) * quant.scale
                }
            }
            else -> throw IllegalStateException(
                "Unsupported output tensor type ${outputTensor.dataType()} for ${model.displayName}"
            )
        }
        return if (model.outputSoftmaxApplied) values else softmax(values)
    }

    private fun softmax(values: FloatArray): FloatArray {
        val maxValue = values.maxOrNull() ?: 0f
        val expValues = FloatArray(values.size)
        var sum = 0.0
        for (i in values.indices) {
            val expValue = exp((values[i] - maxValue).toDouble())
            expValues[i] = expValue.toFloat()
            sum += expValue
        }
        if (sum == 0.0) return expValues
        for (i in expValues.indices) {
            expValues[i] = (expValues[i] / sum).toFloat()
        }
        return expValues
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val normalizedDegrees = ((rotationDegrees % 360) + 360) % 360
        if (normalizedDegrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(normalizedDegrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun centerCrop(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val sourceAspect = bitmap.width.toFloat() / bitmap.height.toFloat()
        val targetAspect = targetWidth.toFloat() / targetHeight.toFloat()
        val cropWidth: Int
        val cropHeight: Int
        if (sourceAspect > targetAspect) {
            cropHeight = bitmap.height
            cropWidth = (bitmap.height * targetAspect).roundToInt()
        } else {
            cropWidth = bitmap.width
            cropHeight = (bitmap.width / targetAspect).roundToInt()
        }
        val left = ((bitmap.width - cropWidth) / 2).coerceAtLeast(0)
        val top = ((bitmap.height - cropHeight) / 2).coerceAtLeast(0)
        val safeCropWidth = min(cropWidth, bitmap.width - left)
        val safeCropHeight = min(cropHeight, bitmap.height - top)
        if (left == 0 && top == 0 && safeCropWidth == bitmap.width && safeCropHeight == bitmap.height) {
            return bitmap
        }
        return Bitmap.createBitmap(bitmap, left, top, safeCropWidth, safeCropHeight)
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(assetPath)
        return FileInputStream(assetFileDescriptor.fileDescriptor).use { inputStream ->
            inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFileDescriptor.startOffset,
                assetFileDescriptor.declaredLength
            )
        }
    }

    private fun elapsedMs(startNanos: Long): Double {
        return (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000.0
    }

    data class RunnerOutput(
        val predictions: List<Prediction>,
        val inputWidth: Int,
        val inputHeight: Int,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val postprocessMs: Double
    )

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        private const val WARMUP_RUNS = 2
    }
}
