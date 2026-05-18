package rs.smobile.shrimpdisease

import android.content.res.AssetManager
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel.MapMode.READ_ONLY

/**
 * Memory-map a packaged .tflite asset so LiteRT can read it without copying the whole file.
 */
fun AssetManager.loadModelFile(modelPath: String): MappedByteBuffer {
    return openFd(modelPath).use { asset ->
        FileInputStream(asset.fileDescriptor).use { inputStream ->
            inputStream.channel.map(READ_ONLY, asset.startOffset, asset.declaredLength)
        }
    }
}
