package rs.smobile.shrimpdisease.profile

data class FarmerProfileUiState(
    val userId: String? = null,
    val displayName: String = "Nông dân nuôi tôm",
    val farmLocation: String = "Ao nuôi ven biển",
    val phoneNumber: String = "Chưa cập nhật",
    val email: String = "Chưa cập nhật",
    val avatarUri: String? = null,
    val dataPermissionEnabled: Boolean = true,
) {
    val initials: String
        get() = displayName
            .split(" ")
            .filter { it.isNotBlank() }
            .take(2)
            .joinToString("") { it.take(1).uppercase() }
            .ifBlank { "N" }
}

data class FarmerProfileUpdate(
    val displayName: String,
    val farmLocation: String,
    val phoneNumber: String,
    val email: String,
    val avatarUri: String? = null,
)
