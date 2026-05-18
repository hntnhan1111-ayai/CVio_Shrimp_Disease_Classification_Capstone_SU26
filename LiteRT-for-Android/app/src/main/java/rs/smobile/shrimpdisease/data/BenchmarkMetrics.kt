package rs.smobile.shrimpdisease.data

data class BenchmarkMetrics(
    val totalRuns: Int,
    val evaluatedRuns: Int,
    val correctRuns: Int,
    val averageInferenceTimeMs: Float,
    val averageSpeed: Float,
    val averageFps: Float,
    val accuracy: Float?,
)
