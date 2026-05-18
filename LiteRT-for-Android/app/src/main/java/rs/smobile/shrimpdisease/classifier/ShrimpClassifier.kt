package rs.smobile.shrimpdisease.classifier

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.Tensor
import rs.smobile.shrimpdisease.loadModelFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.abs
import kotlin.math.exp

@Singleton
class ShrimpClassifier @Inject constructor(
    private val assetManager: AssetManager,
) {
    private var modelFamily = ModelFamily.fromConfig(ModelConfig.MODEL_FAMILY)
    var labels: List<String> = loadLabels(assetManager, ModelConfig.LABEL_FILE)
        private set

    private var interpreterApi: InterpreterApi
    private var inputTensor: Tensor
    private var outputTensor: Tensor
    private var inputBuffer: ByteBuffer
    private var outputBuffer: ByteBuffer
    private var preprocessor: ImagePreprocessor
    var modelInfo: ModelInfo
        private set

    init {
        // Load the default model eagerly so the first prediction can run immediately.
        val (api, inT, outT, inBuf, outBuf, prep, info, lbls) = initializeFromAssets(
            modelFile = ModelConfig.MODEL_FILE,
            labelFile = ModelConfig.LABEL_FILE,
            modelFamilyString = ModelConfig.MODEL_FAMILY,
        )

        interpreterApi = api
        inputTensor = inT
        outputTensor = outT
        inputBuffer = inBuf
        outputBuffer = outBuf
        preprocessor = prep
        modelInfo = info
        labels = lbls
        logModelInfo()
    }

    private data class InitResult(
        val api: InterpreterApi,
        val inT: Tensor,
        val outT: Tensor,
        val inBuf: ByteBuffer,
        val outBuf: ByteBuffer,
        val prep: ImagePreprocessor,
        val info: ModelInfo,
        val labels: List<String>,
    )

    private fun initializeFromAssets(
        modelFile: String,
        labelFile: String,
        modelFamilyString: String,
    ): InitResult {
        val options = InterpreterApi.Options()
            .setNumThreads(ModelConfig.NUM_THREADS)
            .setUseNNAPI(ModelConfig.USE_NNAPI)
            .setUseXNNPACK(ModelConfig.USE_XNNPACK)

        val api = InterpreterApi.create(
            assetManager.loadModelFile(modelFile),
            options
        )
        api.allocateTensors()

        val inT = api.getInputTensor(0)
        val outT = api.getOutputTensor(0)

        val inputShape = inT.shape()
        val inputLayout = resolveInputLayout(inputShape)
        val inputHeight = resolveInputHeight(inputShape, inputLayout)
        val inputWidth = resolveInputWidth(inputShape, inputLayout)

        val inBuf = ByteBuffer.allocateDirect(inT.numBytes()).order(ByteOrder.nativeOrder())
        val outBuf = ByteBuffer.allocateDirect(outT.numBytes()).order(ByteOrder.nativeOrder())

        val inputQuantization = inT.quantizationParams()
        val mf = ModelFamily.fromConfig(modelFamilyString)
        val prep = ImagePreprocessor(
            inputWidth = inputWidth,
            inputHeight = inputHeight,
            inputLayout = inputLayout,
            inputDataType = inT.dataType(),
            quantizationScale = inputQuantization.getScale(),
            quantizationZeroPoint = inputQuantization.getZeroPoint(),
            modelFamily = mf,
        )

        val lbls = loadLabels(assetManager, labelFile)

        val warnings = buildWarnings(
            labelsCount = lbls.size,
            outputClassCount = outT.numElements(),
            inputWidth = inputWidth,
            inputHeight = inputHeight,
        )
        require(lbls.size == outT.numElements()) {
            "$labelFile has ${lbls.size} labels but $modelFile outputs ${outT.numElements()} classes. " +
                "Fix labels.txt so its order/count matches the training class order."
        }

        val info = ModelInfo(
            modelFile = modelFile,
            labelFile = labelFile,
            modelFamily = modelFamilyString,
            input = inT.toTensorInfo(),
            output = outT.toTensorInfo(),
            inputLayout = inputLayout,
            outputClassCount = outT.numElements(),
            labelsCount = lbls.size,
            warnings = warnings,
        )

        // Store the resolved family so preprocessing stays aligned with the loaded model.
        modelFamily = mf

        return InitResult(api, inT, outT, inBuf, outBuf, prep, info, lbls)
    }

    /**
     * Load a different model packaged in assets at runtime. `modelFile` should be an asset path
     * relative to the `assets/` root (e.g. "efficientnet_b0.tflite").
     */
    @Synchronized
    fun loadModelFromAssets(modelFile: String, labelFile: String = ModelConfig.LABEL_FILE, modelFamilyString: String = ModelConfig.MODEL_FAMILY) {
        val (api, inT, outT, inBuf, outBuf, prep, info, lbls) = initializeFromAssets(modelFile, labelFile, modelFamilyString)
        val previousApi = interpreterApi

        // Replace the active model only after the new interpreter is ready.
        interpreterApi = api
        inputTensor = inT
        outputTensor = outT
        inputBuffer = inBuf
        outputBuffer = outBuf
        preprocessor = prep
        modelInfo = info
        labels = lbls
        previousApi.close()
        logModelInfo()
    }

    fun availableModelsInAssets(): List<String> {
        return try {
            assetManager.list("")
                ?.filter { it.lowercase().endsWith(".tflite") }
                ?.sorted()
                ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun modelFamilyForAsset(modelFile: String): String {
        val lowerName = modelFile.lowercase()
        return if (lowerName.contains("yolo")) {
            "yolo_ultralytics"
        } else {
            ModelConfig.MODEL_FAMILY
        }
    }

    @Synchronized
    fun classify(bitmap: Bitmap): ClassificationResult {
        preprocessor.preprocess(bitmap, inputBuffer)
        outputBuffer.rewind()

        val startNanos = SystemClock.elapsedRealtimeNanos()
        interpreterApi.run(inputBuffer, outputBuffer)
        val elapsedNanos = SystemClock.elapsedRealtimeNanos() - startNanos

        val probabilities = toProbabilities(readOutputValues())
        val topK = probabilities.indices
            .sortedByDescending { probabilities[it] }
            .take(ModelConfig.TOP_K)
            .map { index ->
                Prediction(
                    label = labels.getOrNull(index) ?: "Unknown class index: $index",
                    confidence = probabilities[index],
                )
            }

        require(topK.isNotEmpty()) { "Model output is empty." }

        return ClassificationResult(
            top1 = topK.first(),
            topK = topK,
            inferenceTimeMs = elapsedNanos / 1_000_000.0,
        )
    }

    private fun readOutputValues(): FloatArray {
        val outputValues = FloatArray(outputTensor.numElements())
        val outputQuantization = outputTensor.quantizationParams()
        val outputScale = outputQuantization.getScale()
        val outputZeroPoint = outputQuantization.getZeroPoint()

        outputBuffer.rewind()
        for (index in outputValues.indices) {
            outputValues[index] = when (outputTensor.dataType()) {
                DataType.FLOAT32 -> outputBuffer.getFloat()
                DataType.UINT8 -> dequantize(outputBuffer.get().toInt() and 0xFF, outputScale, outputZeroPoint)
                DataType.INT8 -> dequantize(outputBuffer.get().toInt(), outputScale, outputZeroPoint)
                DataType.INT16 -> dequantize(outputBuffer.getShort().toInt(), outputScale, outputZeroPoint)
                DataType.INT32 -> dequantize(outputBuffer.getInt(), outputScale, outputZeroPoint)
                DataType.INT64 -> outputBuffer.getLong().toFloat()
                else -> error("Unsupported output tensor dtype: ${outputTensor.dataType().name}")
            }
        }
        outputBuffer.rewind()
        return outputValues
    }

    private fun dequantize(value: Int, scale: Float, zeroPoint: Int): Float {
        return if (scale > 0f) {
            (value - zeroPoint) * scale
        } else {
            value.toFloat()
        }
    }

    private fun buildWarnings(
        labelsCount: Int,
        outputClassCount: Int,
        inputWidth: Int,
        inputHeight: Int,
    ): List<String> = buildList {
        if (labelsCount != outputClassCount) {
            add("labels.txt has $labelsCount labels but model output has $outputClassCount classes.")
        }
        if (inputWidth != ModelConfig.INPUT_SIZE || inputHeight != ModelConfig.INPUT_SIZE) {
            add("Model input is ${inputWidth}x$inputHeight, while ModelConfig.INPUT_SIZE is ${ModelConfig.INPUT_SIZE}. Runtime tensor shape is used.")
        }
    }

    private fun resolveInputLayout(shape: IntArray): InputLayout {
        require(shape.size == 4) {
            "Expected 4D image input tensor, got shape=${shape.contentToString()}"
        }
        return when {
            shape[3] == RGB_CHANNELS -> InputLayout.NHWC
            shape[1] == RGB_CHANNELS -> InputLayout.NCHW
            else -> error("Cannot infer RGB layout from input shape=${shape.contentToString()}")
        }
    }

    private fun resolveInputHeight(shape: IntArray, layout: InputLayout): Int {
        return when (layout) {
            InputLayout.NHWC -> shape[1]
            InputLayout.NCHW -> shape[2]
        }.also { require(it > 0) { "Invalid input height in shape=${shape.contentToString()}" } }
    }

    private fun resolveInputWidth(shape: IntArray, layout: InputLayout): Int {
        return when (layout) {
            InputLayout.NHWC -> shape[2]
            InputLayout.NCHW -> shape[3]
        }.also { require(it > 0) { "Invalid input width in shape=${shape.contentToString()}" } }
    }

    private fun Tensor.toTensorInfo(): TensorInfo {
        val quantization = quantizationParams()
        return TensorInfo(
            name = name(),
            shape = shape().toList(),
            dataType = dataType().name,
            quantizationScale = quantization.getScale(),
            quantizationZeroPoint = quantization.getZeroPoint(),
        )
    }

    private fun logModelInfo() {
        Log.i(TAG, "Model: ${modelInfo.modelFile}, family: ${modelInfo.modelFamily}")
        Log.i(TAG, "Input: ${modelInfo.input}")
        Log.i(TAG, "Output: ${modelInfo.output}")
        modelInfo.warnings.forEach { Log.w(TAG, it) }
    }

    private fun loadLabels(assetManager: AssetManager, labelPath: String): List<String> {
        return assetManager.open(labelPath).bufferedReader().useLines { lines ->
            lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }
    }

    private companion object {
        private const val TAG = "ShrimpClassifier"
        private const val RGB_CHANNELS = 3
    }
}

fun toProbabilities(values: FloatArray): FloatArray {
    val sum = values.sum()
    val looksLikeProbabilities = values.isNotEmpty() &&
        values.all { it.isFinite() && it >= -1e-5f && it <= 1.00001f } &&
        abs(sum - 1.0f) <= 0.001f

    if (looksLikeProbabilities) return values

    val maxLogit = values.maxOrNull() ?: return values
    val exps = FloatArray(values.size)
    var expSum = 0.0

    for (index in values.indices) {
        val expValue = exp((values[index] - maxLogit).toDouble())
        exps[index] = expValue.toFloat()
        expSum += expValue
    }

    if (expSum == 0.0) return values
    return FloatArray(values.size) { index -> (exps[index] / expSum).toFloat() }
}
