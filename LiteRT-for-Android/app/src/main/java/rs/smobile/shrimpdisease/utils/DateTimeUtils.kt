package rs.smobile.shrimpdisease.utils

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object DateTimeUtils {
    fun formatTimestamp(timestampMillis: Long): String {
        return TIMESTAMP_FORMAT.format(Date(timestampMillis))
    }

    private val TIMESTAMP_FORMAT = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
}
