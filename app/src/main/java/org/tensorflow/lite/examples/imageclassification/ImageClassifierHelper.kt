/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import org.tensorflow.lite.examples.imageclassification.benchmark.BenchmarkMetrics
import org.tensorflow.lite.examples.imageclassification.benchmark.ClassificationResult
import org.tensorflow.lite.examples.imageclassification.benchmark.CvioModel
import org.tensorflow.lite.examples.imageclassification.benchmark.CvioModelRegistry
import org.tensorflow.lite.examples.imageclassification.benchmark.ImageSource
import org.tensorflow.lite.examples.imageclassification.benchmark.MetricsLogger
import org.tensorflow.lite.examples.imageclassification.benchmark.Prediction
import org.tensorflow.lite.examples.imageclassification.benchmark.TfliteClassifierRunner
import java.io.Closeable
import java.time.Instant
import java.util.Locale

class ImageClassifierHelper(
    var threshold: Float = 0.0f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = DELEGATE_CPU,
    var currentModel: Int = 0,
    val context: Context,
    val imageClassifierListener: ClassifierListener?
) : Closeable {
    private val registry = CvioModelRegistry.load(context)
    private val metricsLogger = MetricsLogger(context)
    private var runner: TfliteClassifierRunner? = null
    private var sessionCorrect = 0
    private var sessionLabeled = 0
    var lastResult: ClassificationResult? = null
        private set

    val models: List<CvioModel>
        get() = registry.models

    val labels: List<String>
        get() = registry.labels

    val maxAvailableResults: Int
        get() = registry.labels.size

    fun selectedModel(): CvioModel = registry.models[currentModel]

    fun logsDirectoryPath(): String = metricsLogger.logDirectoryPath()

    fun clearImageClassifier() {
        runner?.close()
        runner = null
    }

    fun resetSessionAccuracy() {
        sessionCorrect = 0
        sessionLabeled = 0
    }

    fun saveLastResult(): String? {
        val result = lastResult ?: return null
        return metricsLogger.append(result).absolutePath
    }

    fun classify(
        image: Bitmap,
        rotationDegrees: Int,
        source: ImageSource = ImageSource.CAMERA,
        groundTruthLabel: String? = null,
        autoLog: Boolean = false
    ) {
        val model = selectedModel()
        if (!model.supported) {
            imageClassifierListener?.onError(
                "${model.displayName} is registered but ONNX Runtime is not implemented in this build."
            )
            return
        }

        try {
            val activeRunner = runner ?: createRunner(model).also { runner = it }
            val benchmarkRuns = if (source == ImageSource.UPLOAD) UPLOAD_BENCHMARK_RUNS else CAMERA_BENCHMARK_RUNS
            val output = activeRunner.classify(
                bitmap = image,
                rotationDegrees = rotationDegrees,
                labels = registry.labels,
                threshold = threshold,
                maxResults = maxResults.coerceIn(1, maxAvailableResults),
                benchmarkRuns = benchmarkRuns
            )
            val predictions = output.predictions.ifEmpty {
                listOf(Prediction(index = -1, label = "No result above threshold", confidence = 0f))
            }
            val top1 = predictions.first()
            val correct = groundTruthLabel?.let { it == top1.label }
            if (correct != null) {
                sessionLabeled += 1
                if (correct) sessionCorrect += 1
            }
            val sessionAccuracy = if (sessionLabeled > 0) {
                sessionCorrect.toDouble() / sessionLabeled.toDouble()
            } else {
                null
            }
            val totalMs = output.preprocessMs + output.inferenceMs + output.postprocessMs
            val metrics = BenchmarkMetrics(
                timestamp = timestampNow(),
                deviceModel = Build.MODEL ?: "unknown",
                androidVersion = Build.VERSION.RELEASE ?: "unknown",
                appVersion = appVersionName(),
                modelId = model.modelId,
                modelName = model.displayName,
                format = model.formatLabel,
                precision = model.precision,
                runtime = model.runtime,
                delegate = delegateLabel(currentDelegate),
                modelSizeMb = model.modelSizeMb,
                inputWidth = output.inputWidth,
                inputHeight = output.inputHeight,
                preprocessMs = output.preprocessMs,
                inferenceMs = output.inferenceMs,
                postprocessMs = output.postprocessMs,
                totalMs = totalMs,
                fps = if (output.inferenceMs > 0.0) 1000.0 / output.inferenceMs else 0.0,
                flops = model.flops,
                top1Label = top1.label,
                top1Confidence = top1.confidence,
                topKPredictions = predictions,
                groundTruthLabel = groundTruthLabel,
                correct = correct,
                sessionAccuracy = sessionAccuracy,
                source = source
            )
            val result = ClassificationResult(
                model = model,
                predictions = predictions,
                metrics = metrics,
                logFilePath = null
            )
            val loggedResult = if (autoLog) {
                val logPath = metricsLogger.append(result).absolutePath
                result.copy(logFilePath = logPath)
            } else {
                result
            }
            lastResult = loggedResult
            imageClassifierListener?.onResults(loggedResult)
            Log.i(
                TAG,
                String.format(
                    Locale.US,
                    "model=%s source=%s inference=%.3fms fps=%.2f top1=%s %.4f",
                    model.modelId,
                    source.label,
                    metrics.inferenceMs,
                    metrics.fps,
                    metrics.top1Label,
                    metrics.top1Confidence
                )
            )
        } catch (e: Exception) {
            clearImageClassifier()
            imageClassifierListener?.onError(
                "Inference failed for ${model.displayName}: ${e.message ?: e.javaClass.simpleName}"
            )
            Log.e(TAG, "Inference failed", e)
        }
    }

    override fun close() {
        clearImageClassifier()
    }

    private fun createRunner(model: CvioModel): TfliteClassifierRunner {
        if (model.format != "tflite") {
            throw IllegalStateException("Only TFLite models are supported in this build")
        }
        return TfliteClassifierRunner(
            context = context,
            model = model,
            numThreads = numThreads,
            delegate = currentDelegate
        )
    }

    private fun timestampNow(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Instant.now().toString()
        } else {
            System.currentTimeMillis().toString()
        }
    }

    private fun appVersionName(): String {
        return try {
            val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
            packageInfo.versionName ?: "unknown"
        } catch (e: PackageManager.NameNotFoundException) {
            "unknown"
        }
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(result: ClassificationResult)
    }

    companion object {
        const val DELEGATE_CPU = TfliteClassifierRunner.DELEGATE_CPU
        const val DELEGATE_GPU = TfliteClassifierRunner.DELEGATE_GPU
        const val DELEGATE_NNAPI = TfliteClassifierRunner.DELEGATE_NNAPI

        private const val CAMERA_BENCHMARK_RUNS = 1
        private const val UPLOAD_BENCHMARK_RUNS = 5
        private const val TAG = "ImageClassifierHelper"

        fun delegateLabel(delegate: Int): String {
            return when (delegate) {
                DELEGATE_GPU -> "GPU"
                DELEGATE_NNAPI -> "NNAPI"
                else -> "CPU"
            }
        }
    }
}
