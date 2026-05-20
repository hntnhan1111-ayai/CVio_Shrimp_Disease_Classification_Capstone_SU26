package rs.smobile.shrimpdisease.utils

object DiseaseTextUtils {
    fun displayLabel(label: String): String {
        val normalized = label.trim().lowercase()
        return when {
            normalized.contains("healthy") -> "Tôm khỏe (Healthy)"
            normalized == "bg" || normalized.contains("background") -> "Ảnh chưa rõ hoặc không có tôm (BG)"
            normalized.contains("wssv_bg") -> "Bệnh đốm trắng phức tạp (WSSV_BG)"
            normalized.contains("wssv") -> "Bệnh đốm trắng (WSSV)"
            normalized.contains("ahpnd") -> "Bệnh hoại tử gan tụy cấp (AHPND)"
            normalized.contains("ems") -> "Hội chứng chết sớm (EMS)"
            normalized.contains("vibrio") -> "Nhiễm khuẩn Vibrio (Vibrio)"
            normalized.isBlank() -> "Chưa có kết quả"
            else -> label
        }
    }

    fun isHealthy(label: String): Boolean {
        return label.contains("healthy", ignoreCase = true)
    }

    fun isUnclear(label: String): Boolean {
        val normalized = label.trim().lowercase()
        return normalized == "bg" || normalized.contains("background") || normalized.contains("unknown")
    }

    fun isDisease(label: String): Boolean {
        return !isHealthy(label) && !isUnclear(label)
    }

    fun diseaseInfo(label: String): String? {
        val normalized = label.trim().lowercase()
        return when {
            normalized.contains("wssv") -> "Bệnh đốm trắng do virus gây ra, có thể lây nhanh trong ao. Bà con nên theo dõi tôm yếu, giảm ăn, xuất hiện đốm trắng trên vỏ; kiểm tra chất lượng nước và hạn chế di chuyển tôm giữa các ao."
            normalized.contains("ahpnd") || normalized.contains("ems") -> "Bệnh thường liên quan gan tụy, tôm có thể yếu, bỏ ăn và chết sớm. Bà con nên kiểm tra gan tụy, thức ăn, môi trường nước và liên hệ kỹ thuật viên nếu dấu hiệu lan nhanh."
            normalized.contains("vibrio") -> "Nhiễm khuẩn Vibrio thường tăng khi môi trường ao xấu hoặc hữu cơ cao. Bà con nên kiểm tra đáy ao, màu nước, khí độc và tăng cường quản lý vệ sinh ao."
            else -> null
        }
    }
}
