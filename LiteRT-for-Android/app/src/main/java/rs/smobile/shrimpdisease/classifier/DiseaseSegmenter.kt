package rs.smobile.shrimpdisease.classifier

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.scale
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.Tensor
import rs.smobile.shrimpdisease.loadModelFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class DiseaseSegmenter(
    private val assetManager: AssetManager,
    private val modelFile: String = ModelDefaults.DISEASE_SEGMENTATION_MODEL_FILE,
    private val labelFile: String = ModelDefaults.DISEASE_SEGMENTATION_LABEL_FILE,
) {
    private var runner: YoloSegmentationRunner? = null
    private var loadFailed = false

    @Synchronized
    fun segment(
        bitmap: Bitmap,
        predictedClass: String,
    ): SegmentationResult? {
        val loadedRunner = runnerOrNull() ?: return null
        return runCatching {
            loadedRunner.segment(bitmap, predictedClass)
        }.onFailure { error ->
            Log.e(TAG, "Disease segmentation failed.", error)
        }.getOrNull()
    }

    private fun runnerOrNull(): YoloSegmentationRunner? {
        if (loadFailed) return null
        runner?.let { return it }

        return try {
            YoloSegmentationRunner(assetManager, modelFile, labelFile).also { runner = it }
        } catch (error: Exception) {
            loadFailed = true
            Log.e(TAG, "Cannot load disease segmentation model: $modelFile", error)
            null
        }
    }

    private companion object {
        private const val TAG = "DiseaseSegmenter"
    }
}

private class YoloSegmentationRunner(
    private val assetManager: AssetManager,
    modelFile: String,
    labelFile: String,
) {
    private val labels = assetManager.open(labelFile).bufferedReader().useLines { lines ->
        lines.map { label -> label.trim() }.filter { label -> label.isNotEmpty() }.toList()
    }
    private val interpreter = InterpreterApi.create(
        assetManager.loadModelFile(modelFile),
        InterpreterApi.Options()
            .setNumThreads(ModelDefaults.NUM_THREADS)
            .setUseXNNPACK(ModelDefaults.USE_XNNPACK)
            .setUseNNAPI(ModelDefaults.USE_NNAPI),
    )
    private val inputTensor: Tensor
    private val detectionTensor: Tensor
    private val protoTensor: Tensor
    private val inputBuffer: ByteBuffer
    private val detectionBuffer: ByteBuffer
    private val protoBuffer: ByteBuffer
    private val inputWidth: Int
    private val inputHeight: Int
    private val inputLayout: InputLayout
    private val detectionChannels: Int
    private val detectionCount: Int
    private val protoWidth: Int
    private val protoHeight: Int
    private val protoChannels: Int

    init {
        require(labels.isNotEmpty()) { "Disease segmentation labels are empty." }

        interpreter.allocateTensors()
        inputTensor = interpreter.getInputTensor(0)
        detectionTensor = interpreter.getOutputTensor(DETECTION_OUTPUT_INDEX)
        protoTensor = interpreter.getOutputTensor(PROTO_OUTPUT_INDEX)

        val inputShape = inputTensor.shape()
        inputLayout = resolveImageLayout(inputShape)
        inputHeight = if (inputLayout == InputLayout.NHWC) inputShape[1] else inputShape[2]
        inputWidth = if (inputLayout == InputLayout.NHWC) inputShape[2] else inputShape[3]

        val detectionShape = detectionTensor.shape()
        require(detectionShape.size == 3) {
            "Expected YOLO segmentation detection output [1, channels, count], got ${detectionShape.contentToString()}"
        }
        detectionChannels = detectionShape[1]
        detectionCount = detectionShape[2]
        require(detectionChannels > BOX_CHANNELS + labels.size) {
            "Detection output channels=$detectionChannels cannot fit ${labels.size} classes and mask coefficients."
        }

        val protoShape = protoTensor.shape()
        require(protoShape.size == 4) {
            "Expected YOLO segmentation proto output [1, h, w, c], got ${protoShape.contentToString()}"
        }
        protoHeight = protoShape[1]
        protoWidth = protoShape[2]
        protoChannels = protoShape[3]
        require(maskCoefficientCount == protoChannels) {
            "Mask coeff count=$maskCoefficientCount but proto channels=$protoChannels."
        }

        inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
        detectionBuffer = ByteBuffer.allocateDirect(detectionTensor.numBytes()).order(ByteOrder.nativeOrder())
        protoBuffer = ByteBuffer.allocateDirect(protoTensor.numBytes()).order(ByteOrder.nativeOrder())
    }

    fun segment(
        bitmap: Bitmap,
        predictedClass: String,
    ): SegmentationResult? {
        if (bitmap.width <= 0 || bitmap.height <= 0) return null

        writeInput(bitmap)
        detectionBuffer.rewind()
        protoBuffer.rewind()
        val outputs = mutableMapOf<Int, Any>(
            DETECTION_OUTPUT_INDEX to detectionBuffer,
            PROTO_OUTPUT_INDEX to protoBuffer,
        )
        interpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(inputBuffer),
            outputs,
        )

        val detectionValues = readTensorValues(detectionBuffer, detectionTensor)
        val allowedClasses = allowedClassIndexes(predictedClass)
        val allDiseaseClasses = labels.indices.toSet()
        val shouldRequireEachAllowedClass = shouldRequireEachAllowedClass(predictedClass, allowedClasses)
        var detections = selectDetections(
            values = detectionValues,
            allowedClasses = allowedClasses,
            scoreThreshold = SCORE_THRESHOLD,
        )
        if (shouldRequireEachAllowedClass) {
            val missingClasses = allowedClasses - detections.map { detection -> detection.classIndex }.toSet()
            if (missingClasses.isNotEmpty()) {
                val relaxedMissingDetections = selectDetections(
                    values = detectionValues,
                    allowedClasses = missingClasses,
                    scoreThreshold = COMPOSITE_MISSING_CLASS_SCORE_THRESHOLD,
                )
                if (relaxedMissingDetections.isNotEmpty()) {
                    detections = suppressDetections(detections + relaxedMissingDetections)
                    Log.i(
                        TAG,
                        "Using composite disease segmentation fallback for predicted=$predictedClass. " +
                            "missing=${missingClasses.labelSummary()}, selected=${detections.summary()}",
                    )
                }
            }
        }
        if (detections.isEmpty()) {
            detections = selectDetections(
                values = detectionValues,
                allowedClasses = allDiseaseClasses,
                scoreThreshold = FALLBACK_SCORE_THRESHOLD,
            )
            if (detections.isNotEmpty()) {
                Log.i(
                    TAG,
                    "Using relaxed disease segmentation fallback for predicted=$predictedClass. " +
                        "maxAllowed=${maxScoreForClasses(detectionValues, allowedClasses).formatScore()}, " +
                        "maxAny=${maxScoreForClasses(detectionValues, allDiseaseClasses).formatScore()}, " +
                        "selected=${detections.summary()}",
                )
            }
        }
        if (detections.isEmpty()) {
            Log.i(
                TAG,
                "No disease segmentation candidate for predicted=$predictedClass. " +
                    "maxAllowed=${maxScoreForClasses(detectionValues, allowedClasses).formatScore()}, " +
                    "maxAny=${maxScoreForClasses(detectionValues, allDiseaseClasses).formatScore()}",
            )
            return null
        }

        val proto = readTensorValues(protoBuffer, protoTensor)
        var classMasks = buildClassMasks(
            detections = detections,
            proto = proto,
            maskThreshold = MASK_THRESHOLD,
        )
        if (classMasks.isEmpty()) {
            classMasks = buildClassMasks(
                detections = detections,
                proto = proto,
                maskThreshold = RELAXED_MASK_THRESHOLD,
            )
            if (classMasks.isNotEmpty()) {
                Log.i(
                    TAG,
                    "Using relaxed mask threshold for predicted=$predictedClass, selected=${detections.summary()}",
                )
            }
        }
        if (classMasks.isEmpty()) {
            classMasks = buildBoxFallbackMasks(detections)
            if (classMasks.isNotEmpty()) {
                Log.i(
                    TAG,
                    "Using box fallback mask for predicted=$predictedClass, selected=${detections.summary()}",
                )
            }
        }
        if (classMasks.isEmpty()) {
            Log.i(
                TAG,
                "Disease segmentation mask is empty for predicted=$predictedClass, selected=${detections.summary()}",
            )
            return null
        }
        val maskBitmap = classMasks.toOverlayBitmap(bitmap.width, bitmap.height)
        val visibleDetections = detections.filter { detection -> detection.classIndex in classMasks.keys }
        val label = classMasks.keys
            .sorted()
            .mapNotNull { classIndex -> labels.getOrNull(classIndex) }
            .joinToString(separator = "_")
            .ifBlank { predictedClass }
        val classCounts = visibleDetections
            .mapNotNull { detection -> labels.getOrNull(detection.classIndex) }
            .groupingBy { label -> label }
            .eachCount()

        return SegmentationResult(
            maskBitmap = maskBitmap,
            label = label,
            confidence = visibleDetections.maxOf { detection -> detection.score },
            detectionCount = visibleDetections.size,
            classCounts = classCounts,
        )
    }

    private val maskCoefficientCount: Int
        get() = detectionChannels - BOX_CHANNELS - labels.size

    private fun writeInput(bitmap: Bitmap) {
        val resized = if (bitmap.width == inputWidth && bitmap.height == inputHeight) {
            bitmap
        } else {
            bitmap.scale(inputWidth, inputHeight, true)
        }

        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (resized !== bitmap) resized.recycle()

        inputBuffer.rewind()
        when (inputLayout) {
            InputLayout.NHWC -> {
                for (pixel in pixels) {
                    putInput(pixel, RED_CHANNEL)
                    putInput(pixel, GREEN_CHANNEL)
                    putInput(pixel, BLUE_CHANNEL)
                }
            }

            InputLayout.NCHW -> {
                for (channel in 0 until RGB_CHANNELS) {
                    for (pixel in pixels) {
                        putInput(pixel, channel)
                    }
                }
            }
        }
        inputBuffer.rewind()
    }

    private fun putInput(pixel: Int, channel: Int) {
        val value = rawRgbChannel(pixel, channel) / 255f
        when (inputTensor.dataType()) {
            DataType.FLOAT32 -> inputBuffer.putFloat(value)
            else -> error("Unsupported YOLO segmentation input dtype: ${inputTensor.dataType().name}")
        }
    }

    private fun selectDetections(
        values: FloatArray,
        allowedClasses: Set<Int>,
        scoreThreshold: Float,
    ): List<SegCandidate> {
        val candidates = ArrayList<SegCandidate>()
        val usableClasses = allowedClasses.ifEmpty { labels.indices.toSet() }

        for (anchor in 0 until detectionCount) {
            val box = readBox(values, anchor) ?: continue
            val coeffStart = BOX_CHANNELS + labels.size
            val coeffs = FloatArray(maskCoefficientCount) { coeffIndex ->
                values[channelOffset(coeffStart + coeffIndex, anchor)]
            }
            for (classIndex in usableClasses) {
                val classScore = score(values[channelOffset(BOX_CHANNELS + classIndex, anchor)])
                if (classScore < scoreThreshold) continue

                candidates += SegCandidate(
                    box = box,
                    classIndex = classIndex,
                    score = classScore,
                    coefficients = coeffs,
                )
            }
        }

        return suppressDetections(candidates)
    }

    private fun suppressDetections(candidates: List<SegCandidate>): List<SegCandidate> {
        return candidates
            .sortedByDescending { candidate -> candidate.score }
            .fold(mutableListOf<SegCandidate>()) { selected, candidate ->
                val selectedForClass = selected.filter { other -> other.classIndex == candidate.classIndex }
                if (
                    selectedForClass.size < MAX_DETECTIONS_PER_CLASS &&
                    selectedForClass.none { other -> candidate.box.iou(other.box) > IOU_THRESHOLD }
                ) {
                    selected += candidate
                }
                selected
            }
    }

    private fun maxScoreForClasses(values: FloatArray, classes: Set<Int>): Float {
        val usableClasses = classes.ifEmpty { labels.indices.toSet() }
        var maxScore = 0f
        for (anchor in 0 until detectionCount) {
            for (classIndex in usableClasses) {
                maxScore = max(maxScore, score(values[channelOffset(BOX_CHANNELS + classIndex, anchor)]))
            }
        }
        return maxScore
    }

    private fun shouldRequireEachAllowedClass(predictedClass: String, allowedClasses: Set<Int>): Boolean {
        val normalizedPrediction = predictedClass.lowercase()
        val wantsCompositeDisease = "bg" in normalizedPrediction && "wssv" in normalizedPrediction
        return wantsCompositeDisease && allowedClasses.size > 1
    }

    private fun Set<Int>.labelSummary(): String {
        return sorted()
            .map { classIndex -> labels.getOrNull(classIndex) ?: classIndex.toString() }
            .joinToString(separator = "_")
    }

    private fun List<SegCandidate>.summary(): String {
        return joinToString(separator = ", ") { candidate ->
            val label = labels.getOrNull(candidate.classIndex) ?: candidate.classIndex.toString()
            "$label:${candidate.score.formatScore()}"
        }
    }

    private fun Float.formatScore(): String {
        return String.format(Locale.US, "%.3f", this)
    }

    private fun readBox(values: FloatArray, anchor: Int): BoxRatios? {
        val centerX = values[channelOffset(0, anchor)]
        val centerY = values[channelOffset(1, anchor)]
        val width = values[channelOffset(2, anchor)]
        val height = values[channelOffset(3, anchor)]
        val coordinateScale = if (max(max(centerX, centerY), max(width, height)) <= NORMALIZED_COORD_MAX) {
            1f
        } else {
            inputWidth.toFloat()
        }
        val x1 = ((centerX - width / 2f) / coordinateScale).coerceIn(0f, 1f)
        val y1 = ((centerY - height / 2f) / coordinateScale).coerceIn(0f, 1f)
        val x2 = ((centerX + width / 2f) / coordinateScale).coerceIn(0f, 1f)
        val y2 = ((centerY + height / 2f) / coordinateScale).coerceIn(0f, 1f)
        if (x2 <= x1 || y2 <= y1) return null
        return BoxRatios(x1, y1, x2, y2)
    }

    private fun buildClassMasks(
        detections: List<SegCandidate>,
        proto: FloatArray,
        maskThreshold: Float,
    ): Map<Int, FloatArray> {
        val outputs = linkedMapOf<Int, FloatArray>()
        for (detection in detections) {
            val output = outputs.getOrPut(detection.classIndex) {
                FloatArray(protoWidth * protoHeight)
            }
            for (y in 0 until protoHeight) {
                val yRatio = (y + 0.5f) / protoHeight
                if (yRatio < detection.box.y1 || yRatio > detection.box.y2) continue

                for (x in 0 until protoWidth) {
                    val xRatio = (x + 0.5f) / protoWidth
                    if (xRatio < detection.box.x1 || xRatio > detection.box.x2) continue

                    val pixelIndex = y * protoWidth + x
                    var logit = 0f
                    for (channel in 0 until protoChannels) {
                        logit += proto[(pixelIndex * protoChannels) + channel] * detection.coefficients[channel]
                    }
                    val probability = sigmoid(logit)
                    if (probability >= maskThreshold && probability > output[pixelIndex]) {
                        output[pixelIndex] = probability
                    }
                }
            }
        }
        return outputs.filterValues { mask -> mask.any { value -> value > 0f } }
    }

    private fun buildBoxFallbackMasks(
        detections: List<SegCandidate>,
    ): Map<Int, FloatArray> {
        val outputs = linkedMapOf<Int, FloatArray>()
        for (detection in detections) {
            val output = outputs.getOrPut(detection.classIndex) {
                FloatArray(protoWidth * protoHeight)
            }
            val xStart = (detection.box.x1 * protoWidth)
                .toInt()
                .coerceIn(0, protoWidth - 1)
            val yStart = (detection.box.y1 * protoHeight)
                .toInt()
                .coerceIn(0, protoHeight - 1)
            val xEnd = (detection.box.x2 * protoWidth)
                .roundToInt()
                .coerceIn(xStart + 1, protoWidth)
            val yEnd = (detection.box.y2 * protoHeight)
                .roundToInt()
                .coerceIn(yStart + 1, protoHeight)
            val fillValue = detection.score.coerceIn(BOX_FALLBACK_MASK_VALUE_MIN, 1f)

            for (y in yStart until yEnd) {
                for (x in xStart until xEnd) {
                    val pixelIndex = y * protoWidth + x
                    if (fillValue > output[pixelIndex]) {
                        output[pixelIndex] = fillValue
                    }
                }
            }
        }

        return outputs.filterValues { mask -> mask.any { value -> value > 0f } }
    }

    private fun Map<Int, FloatArray>.toOverlayBitmap(
        sourceWidth: Int,
        sourceHeight: Int,
    ): Bitmap {
        val scale = min(1f, OVERLAY_MAX_DIMENSION / max(sourceWidth, sourceHeight).toFloat())
        val overlayWidth = max(1, (sourceWidth * scale).roundToInt())
        val overlayHeight = max(1, (sourceHeight * scale).roundToInt())
        val pixels = IntArray(overlayWidth * overlayHeight)

        for (y in 0 until overlayHeight) {
            val protoY = ((y + 0.5f) * protoHeight / overlayHeight)
                .toInt()
                .coerceIn(0, protoHeight - 1)
            for (x in 0 until overlayWidth) {
                val protoX = ((x + 0.5f) * protoWidth / overlayWidth)
                    .toInt()
                    .coerceIn(0, protoWidth - 1)
                val pixelIndex = protoY * protoWidth + protoX
                val winningMask = asSequence()
                    .map { (classIndex, mask) -> classIndex to mask[pixelIndex].coerceIn(0f, 1f) }
                    .filter { (_, value) -> value > 0f }
                    .maxWithOrNull(
                        compareBy<Pair<Int, Float>> { (_, value) -> value }
                            .thenBy { (classIndex, _) -> if (isWssvClass(classIndex)) 1 else 0 },
                    )
                    ?: continue
                val value = winningMask.second

                val alpha = (value * MASK_ALPHA_MAX).roundToInt().coerceIn(0, MASK_ALPHA_MAX)
                val color = maskColorForClass(winningMask.first)
                pixels[y * overlayWidth + x] = Color.argb(alpha, color.red, color.green, color.blue)
            }
        }

        return Bitmap.createBitmap(pixels, overlayWidth, overlayHeight, Bitmap.Config.ARGB_8888)
    }

    private fun maskColorForClass(classIndex: Int): MaskColor {
        val normalizedLabel = labels.getOrNull(classIndex).orEmpty().lowercase()
        return when {
            "wssv" in normalizedLabel -> MaskColor(WSSV_MASK_RED, WSSV_MASK_GREEN, WSSV_MASK_BLUE)
            else -> MaskColor(BG_MASK_RED, BG_MASK_GREEN, BG_MASK_BLUE)
        }
    }

    private fun isWssvClass(classIndex: Int): Boolean {
        return "wssv" in labels.getOrNull(classIndex).orEmpty().lowercase()
    }

    private fun allowedClassIndexes(predictedClass: String): Set<Int> {
        val normalizedPrediction = predictedClass.lowercase()
        val wantsBg = "bg" in normalizedPrediction || "black" in normalizedPrediction
        val wantsWssv = "wssv" in normalizedPrediction
        if (!wantsBg && !wantsWssv) return labels.indices.toSet()

        return labels.withIndex()
            .filter { (_, label) ->
                val normalizedLabel = label.lowercase()
                (wantsBg && ("bg" in normalizedLabel || "black" in normalizedLabel)) ||
                    (wantsWssv && "wssv" in normalizedLabel)
            }
            .map { (index, _) -> index }
            .toSet()
    }

    private fun channelOffset(channel: Int, anchor: Int): Int {
        return channel * detectionCount + anchor
    }

    private fun readTensorValues(buffer: ByteBuffer, tensor: Tensor): FloatArray {
        val values = FloatArray(tensor.numElements())
        buffer.rewind()
        for (index in values.indices) {
            values[index] = when (tensor.dataType()) {
                DataType.FLOAT32 -> buffer.getFloat()
                DataType.UINT8 -> (buffer.get().toInt() and 0xFF).toFloat()
                DataType.INT8 -> buffer.get().toFloat()
                else -> error("Unsupported YOLO segmentation output dtype: ${tensor.dataType().name}")
            }
        }
        buffer.rewind()
        return values
    }

    private fun score(value: Float): Float {
        return if (value in 0f..1f) value else sigmoid(value)
    }

    private fun sigmoid(value: Float): Float {
        return (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
    }

    private fun resolveImageLayout(shape: IntArray): InputLayout {
        require(shape.size == 4) {
            "Expected 4D YOLO segmentation image input, got ${shape.contentToString()}"
        }
        return when {
            shape[3] == RGB_CHANNELS -> InputLayout.NHWC
            shape[1] == RGB_CHANNELS -> InputLayout.NCHW
            else -> error("Cannot infer YOLO segmentation RGB layout from ${shape.contentToString()}")
        }
    }

    private fun rawRgbChannel(pixel: Int, channel: Int): Int {
        return when (channel) {
            RED_CHANNEL -> pixel shr 16 and 0xFF
            GREEN_CHANNEL -> pixel shr 8 and 0xFF
            else -> pixel and 0xFF
        }
    }

    private data class SegCandidate(
        val box: BoxRatios,
        val classIndex: Int,
        val score: Float,
        val coefficients: FloatArray,
    )

    private data class MaskColor(
        val red: Int,
        val green: Int,
        val blue: Int,
    )

    private data class BoxRatios(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
    ) {
        fun iou(other: BoxRatios): Float {
            val intersectionWidth = (min(x2, other.x2) - max(x1, other.x1)).coerceAtLeast(0f)
            val intersectionHeight = (min(y2, other.y2) - max(y1, other.y1)).coerceAtLeast(0f)
            val intersection = intersectionWidth * intersectionHeight
            val union = area() + other.area() - intersection
            return if (union <= 0f) 0f else intersection / union
        }

        private fun area(): Float {
            return (x2 - x1).coerceAtLeast(0f) * (y2 - y1).coerceAtLeast(0f)
        }
    }

    private companion object {
        private const val TAG = "DiseaseSegmenter"
        private const val DETECTION_OUTPUT_INDEX = 0
        private const val PROTO_OUTPUT_INDEX = 1
        private const val BOX_CHANNELS = 4
        private const val RGB_CHANNELS = 3
        private const val RED_CHANNEL = 0
        private const val GREEN_CHANNEL = 1
        private const val BLUE_CHANNEL = 2
        private const val SCORE_THRESHOLD = 0.2f
        private const val FALLBACK_SCORE_THRESHOLD = 0.08f
        private const val COMPOSITE_MISSING_CLASS_SCORE_THRESHOLD = 0.03f
        private const val IOU_THRESHOLD = 0.5f
        private const val MASK_THRESHOLD = 0.3f
        private const val RELAXED_MASK_THRESHOLD = 0.18f
        private const val BOX_FALLBACK_MASK_VALUE_MIN = 0.42f
        private const val MAX_DETECTIONS_PER_CLASS = 4
        private const val NORMALIZED_COORD_MAX = 2f
        private const val OVERLAY_MAX_DIMENSION = 640f
        private const val MASK_ALPHA_MAX = 180
        private const val BG_MASK_RED = 0
        private const val BG_MASK_GREEN = 200
        private const val BG_MASK_BLUE = 83
        private const val WSSV_MASK_RED = 124
        private const val WSSV_MASK_GREEN = 77
        private const val WSSV_MASK_BLUE = 255
    }
}
