package rs.smobile.shrimpdisease

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Size
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview as CameraXPreview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.lifecycle.compose.LocalLifecycleOwner
import rs.smobile.shrimpdisease.classifier.ModelDefaults
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

@Composable
fun CameraCaptureCard(
    enabled: Boolean,
    capturedBitmap: Bitmap?,
    onSnapshot: (Bitmap) -> Unit,
    onRetake: () -> Unit,
    onFrameObserved: () -> Unit,
    onError: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val latestFrame = remember { AtomicReference<Bitmap?>(null) }

    DisposableEffect(Unit) {
        onDispose {
            latestFrame.getAndSet(null)?.recycle()
        }
    }

    DisposableEffect(capturedBitmap) {
        if (capturedBitmap == null) {
            latestFrame.getAndSet(null)?.recycle()
        }
        onDispose { }
    }

    Card(
        modifier = modifier
            .fillMaxWidth()
            .height(420.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (capturedBitmap != null) {
                Image(
                    bitmap = capturedBitmap.asImageBitmap(),
                    contentDescription = "Captured shrimp image",
                    modifier = Modifier
                        .fillMaxSize()
                        .weight(1f),
                    contentScale = ContentScale.Crop,
                )
            } else if (!enabled) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .weight(1f),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(text = "Camera is inactive")
                }
            } else {
                CameraPreview(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    enabled = enabled,
                    onFrameBitmap = { bitmap ->
                        latestFrame.getAndSet(bitmap)?.recycle()
                    },
                    onFrameObserved = onFrameObserved,
                    onError = onError,
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                if (capturedBitmap == null) {
                    Button(
                        enabled = enabled,
                        onClick = {
                            val snapshot = latestFrame.get()?.copy(Bitmap.Config.ARGB_8888, false)
                            if (snapshot != null) {
                                onSnapshot(snapshot)
                            } else {
                                onError("Camera preview is not ready yet.")
                            }
                        },
                        modifier = Modifier.weight(1f),
                    ) {
                        Text(text = "Take snapshot")
                    }
                    OutlinedButton(
                        enabled = enabled,
                        onClick = {
                            latestFrame.getAndSet(null)?.recycle()
                        },
                        modifier = Modifier.weight(1f),
                    ) {
                        Text(text = "Clear frame")
                    }
                } else {
                    OutlinedButton(
                        enabled = enabled,
                        onClick = {
                            latestFrame.getAndSet(null)?.recycle()
                            onRetake()
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(text = "Retake snapshot")
                    }
                }
            }
        }
    }
}

/**
 * CameraX preview plus an ImageAnalysis pipeline that keeps the latest RGBA frame
 * available for snapshot inference.
 */
@Composable
private fun CameraPreview(
    modifier: Modifier = Modifier,
    enabled: Boolean,
    onFrameBitmap: (Bitmap) -> Unit,
    onFrameObserved: () -> Unit,
    onError: (String) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }
    val cameraExecutor = remember(enabled) { Executors.newSingleThreadExecutor() }

    DisposableEffect(enabled, cameraExecutor) {
        var cameraProvider: ProcessCameraProvider? = null
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val listener = Runnable {
            try {
                cameraProvider = cameraProviderFuture.get()
                if (!enabled) return@Runnable

                val rotation = previewView.display?.rotation ?: Surface.ROTATION_0
                val preview = CameraXPreview.Builder()
                    .setTargetRotation(rotation)
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    Size(ModelDefaults.INPUT_SIZE, ModelDefaults.INPUT_SIZE),
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                )
                            )
                            .build()
                    )
                    .setTargetRotation(rotation)
                    .build()

                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    onFrameObserved()
                    try {
                        val bitmap = imageProxyToBitmap(imageProxy)
                        if (bitmap != null) {
                            onFrameBitmap(bitmap)
                        } else {
                            onError("Could not convert camera frame to bitmap.")
                        }
                    } catch (error: Throwable) {
                        onError(error.message ?: "Camera analyzer failed.")
                    } finally {
                        imageProxy.close()
                    }
                }

                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis,
                )
            } catch (error: Throwable) {
                onError(error.message ?: "Unable to start camera.")
            }
        }

        cameraProviderFuture.addListener(listener, ContextCompat.getMainExecutor(context))

        onDispose {
            cameraProvider?.unbindAll()
            cameraExecutor.shutdown()
        }
    }

    AndroidView(
        modifier = modifier,
        factory = { previewView },
    )
}

/**
 * Converts CameraX RGBA frames to an upright bitmap. CameraX may pad each row, so
 * the bitmap is cropped back to the real image width before rotation is applied.
 */
private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
    val plane = imageProxy.planes.firstOrNull() ?: return null
    val buffer = plane.buffer
    buffer.rewind()

    val width = imageProxy.width
    val height = imageProxy.height
    val pixelStride = plane.pixelStride.coerceAtLeast(RGBA_PIXEL_STRIDE)
    val rowStride = plane.rowStride.coerceAtLeast(width * pixelStride)
    val paddedWidth = rowStride / pixelStride

    val paddedBitmap = createBitmap(paddedWidth, height).apply {
        copyPixelsFromBuffer(buffer)
    }

    val croppedBitmap = if (paddedWidth == width) {
        paddedBitmap
    } else {
        Bitmap.createBitmap(paddedBitmap, 0, 0, width, height).also {
            paddedBitmap.recycle()
        }
    }

    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
    if (rotationDegrees == 0) return croppedBitmap

    val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(
        croppedBitmap,
        0,
        0,
        croppedBitmap.width,
        croppedBitmap.height,
        matrix,
        true,
    ).also {
        croppedBitmap.recycle()
    }
}

private const val RGBA_PIXEL_STRIDE = 4
