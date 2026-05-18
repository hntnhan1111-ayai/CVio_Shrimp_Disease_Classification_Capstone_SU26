package rs.smobile.shrimpdisease.classifier

import kotlin.math.sqrt

/**
 * Lightweight on-device background removal fallback.
 *
 * The training notebook used rembg before resizing. The Android app does not
 * bundle a segmentation model yet, so this removes edge-connected background
 * regions that look similar to the image border and paints them black.
 */
object BackgroundRemover {
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
