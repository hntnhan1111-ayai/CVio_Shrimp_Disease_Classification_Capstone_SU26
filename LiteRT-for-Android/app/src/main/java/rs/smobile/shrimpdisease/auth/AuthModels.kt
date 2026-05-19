package rs.smobile.shrimpdisease.auth

enum class AuthRole {
    Farmer,
    Admin,
}

data class AuthUser(
    val id: String,
    val account: String,
    val role: AuthRole,
    val displayName: String,
    val createdAt: Long,
)

data class AuthSession(
    val user: AuthUser?,
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
) {
    val isAuthenticated: Boolean
        get() = user != null
}

sealed class AuthResult {
    data class Success(val user: AuthUser) : AuthResult()
    data class Error(val message: String) : AuthResult()
}
