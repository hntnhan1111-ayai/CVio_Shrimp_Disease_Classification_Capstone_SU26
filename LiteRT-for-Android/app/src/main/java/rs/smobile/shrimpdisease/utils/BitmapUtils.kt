package rs.smobile.shrimpdisease.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import kotlin.math.max
import kotlin.math.roundToInt

object BitmapUtils {
    fun decodeBitmapFromUri(context: Context, uri: Uri): Bitmap {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val width = info.size.width
            val height = info.size.height
            val longestSide = max(width, height)
            if (longestSide > MAX_DECODE_SIDE) {
                val scale = MAX_DECODE_SIDE / longestSide.toFloat()
                decoder.setTargetSize(
                    (width * scale).roundToInt().coerceAtLeast(1),
                    (height * scale).roundToInt().coerceAtLeast(1),
                )
            }
        }
    }

    private const val MAX_DECODE_SIDE = 1280
}
