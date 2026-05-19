package rs.smobile.shrimpdisease.profile

data class FarmerProfileUiState(
    val userId: String? = null,
    val displayName: String = "Shrimp Farmer",
    val farmLocation: String = "Coastal pond",
    val phoneNumber: String = "Not set",
    val email: String = "Not set",
    val dataPermissionEnabled: Boolean = true,
) {
    val initials: String
        get() = displayName
            .split(" ")
            .filter { it.isNotBlank() }
            .take(2)
            .joinToString("") { it.take(1).uppercase() }
            .ifBlank { "F" }
}

data class FarmerProfileUpdate(
    val displayName: String,
    val farmLocation: String,
    val phoneNumber: String,
    val email: String,
)
