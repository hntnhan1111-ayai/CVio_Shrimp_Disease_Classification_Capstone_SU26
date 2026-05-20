package rs.smobile.shrimpdisease.utils

object DiseaseTextUtils {

    fun displayLabel(label: String): String {
        val normalized = label.trim().lowercase()

        return when {
            normalized.isBlank() -> "Chưa có kết quả"

            normalized.contains("healthy") ->
                "Tôm khỏe (Healthy)"

            isBgWssvLabel(normalized) ->
                "Bệnh đen mang kết hợp đốm trắng (BG_WSSV)"

            isBgLabel(normalized) ->
                "Bệnh đen mang (Black Gill - BG)"

            normalized.contains("wssv") ->
                "Bệnh đốm trắng (White Spot Syndrome Virus - WSSV)"

            normalized.contains("unknown") || normalized.contains("background") ->
                "Không xác định"

            else -> label
        }
    }

    fun isHealthy(label: String): Boolean {
        val normalized = label.trim().lowercase()
        return normalized.contains("healthy")
    }

    fun isUnclear(label: String): Boolean {
        val normalized = label.trim().lowercase()
        return normalized.isBlank() ||
                normalized.contains("unknown") ||
                normalized.contains("background")
    }

    fun isDisease(label: String): Boolean {
        return !isHealthy(label) && !isUnclear(label)
    }

    fun diseaseInfo(label: String): String? {
        val normalized = label.trim().lowercase()

        return when {
            normalized.isBlank() -> null

            normalized.contains("healthy") ->
                "Tôm không có dấu hiệu bệnh rõ ràng trên ảnh. Thân và vỏ tương đối bình thường, không thấy đốm trắng bất thường hoặc vùng mang bị sẫm màu. Tuy nhiên, kết quả AI chỉ mang tính hỗ trợ sàng lọc, không thay thế kiểm tra thực tế trong ao."

            isBgWssvLabel(normalized) ->
                "Ảnh tôm có dấu hiệu kết hợp của bệnh đen mang và bệnh đốm trắng. Vùng mang có thể bị sẫm màu, đồng thời trên vỏ có thể xuất hiện các đốm trắng hoặc mảng trắng bất thường. Đây là class khó vì model phải nhận diện đồng thời hai nhóm đặc trưng: vùng mang và bề mặt vỏ."

            isBgLabel(normalized) ->
                "Bệnh đen mang, còn gọi là Black Gill, thường biểu hiện ở vùng mang gần đầu tôm bị sẫm màu, nâu hoặc đen. Dấu hiệu này có thể liên quan đến môi trường nước kém, chất hữu cơ cao, ký sinh trùng, nấm hoặc vi khuẩn. Khi phát hiện, nên kiểm tra chất lượng nước, đáy ao và tình trạng hoạt động của tôm."

            normalized.contains("wssv") ->
                "Bệnh đốm trắng, hay White Spot Syndrome Virus, là bệnh do virus gây ra trên tôm. Dấu hiệu thường gặp là các đốm trắng hoặc vùng trắng bất thường trên vỏ, đặc biệt ở phần đầu ngực và thân. Bệnh có thể lây lan nhanh, vì vậy cần theo dõi tôm yếu, giảm ăn, kiểm tra môi trường nước và hạn chế di chuyển tôm giữa các ao."

            normalized.contains("unknown") || normalized.contains("background") ->
                "Ảnh chưa đủ rõ hoặc không thuộc các nhóm bệnh mà mô hình đang hỗ trợ. Nên chụp lại ảnh với ánh sáng tốt hơn, lấy rõ toàn thân tôm, đặc biệt là vùng đầu, mang và vỏ."

            else -> null
        }
    }

    private fun isBgLabel(normalized: String): Boolean {
        return normalized == "bg" ||
                normalized == "black_gill" ||
                normalized == "blackgill" ||
                normalized.contains("black_gill") ||
                normalized.contains("blackgill")
    }

    private fun isBgWssvLabel(normalized: String): Boolean {
        return normalized.contains("bg_wssv") ||
                normalized.contains("wssv_bg") ||
                normalized.contains("bg-wssv") ||
                normalized.contains("wssv-bg") ||
                normalized.contains("black_gill_wssv") ||
                normalized.contains("wssv_black_gill")
    }
}