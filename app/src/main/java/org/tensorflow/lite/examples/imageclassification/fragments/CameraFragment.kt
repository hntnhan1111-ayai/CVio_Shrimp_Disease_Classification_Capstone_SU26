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

package org.tensorflow.lite.examples.imageclassification.fragments

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import androidx.recyclerview.widget.LinearLayoutManager
import org.tensorflow.lite.examples.imageclassification.ImageClassifierHelper
import org.tensorflow.lite.examples.imageclassification.R
import org.tensorflow.lite.examples.imageclassification.benchmark.ClassificationResult
import org.tensorflow.lite.examples.imageclassification.benchmark.ImageSource
import org.tensorflow.lite.examples.imageclassification.databinding.FragmentCameraBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(), ImageClassifierHelper.ClassifierListener {

    companion object {
        private const val TAG = "Image Classifier"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var imageClassifierHelper: ImageClassifierHelper
    private lateinit var bitmapBuffer: Bitmap
    private val classificationResultsAdapter by lazy {
        ClassificationResultsAdapter().apply {
            updateAdapterSize(imageClassifierHelper.maxResults)
        }
    }
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var selectedBitmap: Bitmap? = null
    private var selectedGroundTruthLabel: String? = null
    private var uploadedImageMode = false

    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService

    private val imagePickerLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    onImageSelected(uri, result.data)
                }
            }
        }

    override fun onResume() {
        super.onResume()

        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(CameraFragmentDirections.actionCameraToPermissions())
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        if (::imageClassifierHelper.isInitialized) {
            imageClassifierHelper.close()
        }
        selectedBitmap = null
        super.onDestroyView()

        // Shut down our background executor
        cameraExecutor.shutdown()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        imageClassifierHelper =
            ImageClassifierHelper(context = requireContext(), imageClassifierListener = this)

        with(fragmentCameraBinding.recyclerviewResults) {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = classificationResultsAdapter
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        applySystemBarInsets()
        initBenchmarkControls()
        initBottomSheetControls()
        updateControlsUi()
        updateModelStatus()
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    private fun applySystemBarInsets() {
        val bottomSheetRoot = fragmentCameraBinding.bottomSheetLayout.root
        val initialBottomPadding = bottomSheetRoot.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(fragmentCameraBinding.cameraContainer) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            bottomSheetRoot.setPadding(
                bottomSheetRoot.paddingLeft,
                bottomSheetRoot.paddingTop,
                bottomSheetRoot.paddingRight,
                initialBottomPadding + systemBars.bottom
            )
            insets
        }
    }

    private fun initBenchmarkControls() {
        val modelAdapter = ArrayAdapter(
            requireContext(),
            android.R.layout.simple_spinner_item,
            imageClassifierHelper.models.map { it.selectorLabel }
        )
        modelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        fragmentCameraBinding.spinnerModelSelector.adapter = modelAdapter
        fragmentCameraBinding.spinnerModelSelector.setSelection(imageClassifierHelper.currentModel, false)
        fragmentCameraBinding.spinnerModelSelector.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    if (imageClassifierHelper.currentModel == position) return
                    imageClassifierHelper.currentModel = position
                    imageClassifierHelper.clearImageClassifier()
                    imageClassifierHelper.resetSessionAccuracy()
                    classificationResultsAdapter.updateResults(null)
                    classificationResultsAdapter.notifyDataSetChanged()
                    updateModelStatus()
                    val model = imageClassifierHelper.selectedModel()
                    if (!model.supported) {
                        Toast.makeText(
                            requireContext(),
                            "ONNX is registered but not supported in this build",
                            Toast.LENGTH_SHORT
                        ).show()
                        return
                    }
                    if (uploadedImageMode) {
                        selectedBitmap?.let { runUploadedImageInference(it) }
                    }
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    /* no op */
                }
            }

        val groundTruthAdapter = ArrayAdapter(
            requireContext(),
            android.R.layout.simple_spinner_item,
            listOf(getString(R.string.ground_truth_none)) + imageClassifierHelper.labels
        )
        groundTruthAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        fragmentCameraBinding.spinnerGroundTruth.adapter = groundTruthAdapter
        fragmentCameraBinding.spinnerGroundTruth.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    selectedGroundTruthLabel = if (position == 0) {
                        null
                    } else {
                        imageClassifierHelper.labels[position - 1]
                    }
                    imageClassifierHelper.resetSessionAccuracy()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    /* no op */
                }
            }

        fragmentCameraBinding.buttonSelectImage.setOnClickListener {
            launchImagePicker()
        }

        fragmentCameraBinding.buttonResumeCamera.setOnClickListener {
            uploadedImageMode = false
            fragmentCameraBinding.imageSelectedPreview.visibility = View.GONE
            updateModelStatus()
        }

        fragmentCameraBinding.buttonSaveLog.setOnClickListener {
            val path = imageClassifierHelper.saveLastResult()
            if (path == null) {
                Toast.makeText(requireContext(), "No inference result to save", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(requireContext(), "Saved log: $path", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun initBottomSheetControls() {
        // When clicked, lower classification score threshold floor
        fragmentCameraBinding.bottomSheetLayout.thresholdMinus.setOnClickListener {
            if (imageClassifierHelper.threshold >= 0.1) {
                imageClassifierHelper.threshold -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise classification score threshold floor
        fragmentCameraBinding.bottomSheetLayout.thresholdPlus.setOnClickListener {
            if (imageClassifierHelper.threshold < 0.9) {
                imageClassifierHelper.threshold += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, reduce the number of objects that can be classified at a time
        fragmentCameraBinding.bottomSheetLayout.maxResultsMinus.setOnClickListener {
            if (imageClassifierHelper.maxResults > 1) {
                imageClassifierHelper.maxResults--
                updateControlsUi()
                classificationResultsAdapter.updateAdapterSize(size = imageClassifierHelper.maxResults)
            }
        }

        // When clicked, increase the number of objects that can be classified at a time
        fragmentCameraBinding.bottomSheetLayout.maxResultsPlus.setOnClickListener {
            if (imageClassifierHelper.maxResults < imageClassifierHelper.maxAvailableResults) {
                imageClassifierHelper.maxResults++
                updateControlsUi()
                classificationResultsAdapter.updateAdapterSize(size = imageClassifierHelper.maxResults)
            }
        }

        // When clicked, decrease the number of threads used for classification
        fragmentCameraBinding.bottomSheetLayout.threadsMinus.setOnClickListener {
            if (imageClassifierHelper.numThreads > 1) {
                imageClassifierHelper.numThreads--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of threads used for classification
        fragmentCameraBinding.bottomSheetLayout.threadsPlus.setOnClickListener {
            if (imageClassifierHelper.numThreads < 4) {
                imageClassifierHelper.numThreads++
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(0, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    imageClassifierHelper.currentDelegate = position
                    updateControlsUi()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset classifier.
    private fun updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxResultsValue.text =
            imageClassifierHelper.maxResults.toString()

        fragmentCameraBinding.bottomSheetLayout.thresholdValue.text =
            String.format(Locale.US, "%.2f", imageClassifierHelper.threshold)
        fragmentCameraBinding.bottomSheetLayout.threadsValue.text =
            imageClassifierHelper.numThreads.toString()
        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        imageClassifierHelper.clearImageClassifier()
        updateModelStatus()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider =
            cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        // CameraSelector - makes assumption that we're only using the back camera
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview =
            Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (!::bitmapBuffer.isInitialized) {
                            // The image rotation and RGB image buffer are initialized only once
                            // the analyzer has started running
                            bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,
                                Bitmap.Config.ARGB_8888
                            )
                        }

                        classifyImage(image)
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun classifyImage(image: ImageProxy) {
        if (uploadedImageMode) {
            image.close()
            return
        }
        if (!imageClassifierHelper.selectedModel().supported) {
            image.close()
            return
        }

        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

        imageClassifierHelper.classify(
            image = bitmapBuffer,
            rotationDegrees = image.imageInfo.rotationDegrees,
            source = ImageSource.CAMERA,
            groundTruthLabel = selectedGroundTruthLabel,
            autoLog = false
        )
    }

    private fun launchImagePicker() {
        val photoPickerIntent = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Intent(MediaStore.ACTION_PICK_IMAGES).apply {
                type = "image/*"
            }
        } else {
            null
        }
        val fallbackIntent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
        }
        val intent = if (
            photoPickerIntent != null &&
            photoPickerIntent.resolveActivity(requireActivity().packageManager) != null
        ) {
            photoPickerIntent
        } else {
            fallbackIntent
        }
        imagePickerLauncher.launch(intent)
    }

    private fun onImageSelected(uri: Uri, data: Intent?) {
        data?.let { intent ->
            val flags = intent.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
            if (flags != 0 && intent.action == Intent.ACTION_OPEN_DOCUMENT) {
                try {
                    requireContext().contentResolver.takePersistableUriPermission(uri, flags)
                } catch (e: SecurityException) {
                    Log.w(TAG, "Unable to persist URI permission", e)
                }
            }
        }

        val bitmap = decodeBitmap(uri)
        if (bitmap == null) {
            Toast.makeText(requireContext(), "Unable to decode selected image", Toast.LENGTH_SHORT).show()
            return
        }
        selectedBitmap = bitmap
        uploadedImageMode = true
        fragmentCameraBinding.imageSelectedPreview.setImageBitmap(bitmap)
        fragmentCameraBinding.imageSelectedPreview.visibility = View.VISIBLE
        updateModelStatus()
        runUploadedImageInference(bitmap)
    }

    private fun runUploadedImageInference(bitmap: Bitmap) {
        if (!imageClassifierHelper.selectedModel().supported) {
            Toast.makeText(
                requireContext(),
                "Selected ONNX model is not supported in this build",
                Toast.LENGTH_SHORT
            ).show()
            return
        }
        cameraExecutor.execute {
            imageClassifierHelper.classify(
                image = bitmap,
                rotationDegrees = 0,
                source = ImageSource.UPLOAD,
                groundTruthLabel = selectedGroundTruthLabel,
                autoLog = true
            )
        }
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        return try {
            val resolver = requireContext().contentResolver
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(resolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = false
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(resolver, uri)
            }
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to decode selected image", e)
            null
        }
    }

    private fun updateModelStatus() {
        val model = imageClassifierHelper.selectedModel()
        val supportStatus = if (model.supported) "Ready" else "ONNX registered; not supported"
        val mode = if (uploadedImageMode) "Upload image" else "Live camera"
        fragmentCameraBinding.textModelStatus.text = String.format(
            Locale.US,
            "%s\n%s | %s | %s | %.3f MB | %dx%d | FLOPs: %s\nLogs: %s",
            model.displayName,
            supportStatus,
            mode,
            ImageClassifierHelper.delegateLabel(imageClassifierHelper.currentDelegate),
            model.modelSizeMb,
            model.expectedInputWidth,
            model.expectedInputHeight,
            model.flops ?: "N/A",
            imageClassifierHelper.logsDirectoryPath()
        )
    }

    @SuppressLint("NotifyDataSetChanged")
    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            classificationResultsAdapter.updateResults(null)
            classificationResultsAdapter.notifyDataSetChanged()
        }
    }

    @SuppressLint("NotifyDataSetChanged")
    override fun onResults(result: ClassificationResult) {
        activity?.runOnUiThread {
            // Show result on bottom sheet
            classificationResultsAdapter.updateResults(result.predictions)
            classificationResultsAdapter.notifyDataSetChanged()
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format(Locale.US, "%.2f ms", result.metrics.inferenceMs)
            fragmentCameraBinding.bottomSheetLayout.benchmarkMetricsValue.text =
                formatMetrics(result)
            if (result.metrics.source == ImageSource.UPLOAD && result.logFilePath != null) {
                Toast.makeText(requireContext(), "Logged upload metrics", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun formatMetrics(result: ClassificationResult): String {
        val metrics = result.metrics
        return String.format(
            Locale.US,
            "Model: %s\nFormat: %s | Precision: %s | Runtime: %s\nDelegate: %s | Size: %.3f MB | Input: %s\nPreprocess: %.2f ms | Inference: %.2f ms | Postprocess: %.2f ms | Total: %.2f ms\nFPS: %.2f | FLOPs: %s\nTop-1: %s (%.2f%%)\nGround truth: %s | Correct: %s | Session accuracy: %s\nLog: %s",
            metrics.modelName,
            metrics.format,
            metrics.precision,
            metrics.runtime,
            metrics.delegate,
            metrics.modelSizeMb,
            metrics.inputSizeLabel,
            metrics.preprocessMs,
            metrics.inferenceMs,
            metrics.postprocessMs,
            metrics.totalMs,
            metrics.fps,
            metrics.flopsLabel,
            metrics.top1Label,
            metrics.top1Confidence * 100.0f,
            metrics.groundTruthLabel ?: "N/A",
            metrics.correctnessLabel,
            metrics.accuracyLabel,
            result.logFilePath ?: "not saved automatically"
        )
    }
}
