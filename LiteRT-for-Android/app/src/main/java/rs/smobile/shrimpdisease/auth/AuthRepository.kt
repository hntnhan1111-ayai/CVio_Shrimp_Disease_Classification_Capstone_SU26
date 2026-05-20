package rs.smobile.shrimpdisease.auth

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AuthRepository @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
    private val random = SecureRandom()

    private val _session = MutableStateFlow(
        AuthSession(user = readSessionUser())
    )
    val session: StateFlow<AuthSession> = _session

    init {
        seedDefaultAdminIfNeeded()
    }

    fun login(
        role: AuthRole,
        account: String,
        password: String,
    ): AuthResult {
        val normalizedAccount = account.normalizedAccount()
        val validationError = validateCredentials(normalizedAccount, password)
        if (validationError != null) return AuthResult.Error(validationError)

        val record = readUsers().firstOrNull {
            it.account == normalizedAccount && it.role == role
        } ?: return AuthResult.Error("Account not found for ${role.name.lowercase()} role.")

        if (record.passwordHash != hashPassword(password, record.salt)) {
            return AuthResult.Error("Invalid password.")
        }

        saveSession(record.id)
        val user = record.toAuthUser()
        _session.value = AuthSession(user = user)
        return AuthResult.Success(user)
    }

    fun register(
        role: AuthRole,
        account: String,
        password: String,
    ): AuthResult {
        if (role == AuthRole.Admin) {
            return AuthResult.Error("Admin accounts must be created by the system administrator.")
        }

        val normalizedAccount = account.normalizedAccount()
        val validationError = validateCredentials(normalizedAccount, password)
        if (validationError != null) return AuthResult.Error(validationError)

        val users = readUsers()
        if (users.any { it.account == normalizedAccount }) {
            return AuthResult.Error("This account is already registered.")
        }

        val salt = generateSalt()
        val record = StoredUser(
            id = UUID.randomUUID().toString(),
            account = normalizedAccount,
            role = role,
            displayName = displayNameFor(normalizedAccount),
            salt = salt,
            passwordHash = hashPassword(password, salt),
            createdAt = System.currentTimeMillis(),
        )
        writeUsers(users + record)
        saveSession(record.id)
        val user = record.toAuthUser()
        _session.value = AuthSession(user = user)
        return AuthResult.Success(user)
    }

    fun logout() {
        preferences.edit()
            .remove(KEY_SESSION_USER_ID)
            .apply()
        _session.value = AuthSession(user = null)
    }

    fun updateDisplayName(displayName: String): AuthResult {
        val currentUser = _session.value.user ?: return AuthResult.Error("No active session.")
        val cleanName = displayName.trim()
        if (cleanName.isBlank()) return AuthResult.Error("Display name is required.")

        val users = readUsers()
        val updatedUsers = users.map { user ->
            if (user.id == currentUser.id) user.copy(displayName = cleanName) else user
        }
        writeUsers(updatedUsers)

        val updatedUser = updatedUsers.firstOrNull { it.id == currentUser.id }
            ?: return AuthResult.Error("Account not found.")
        val authUser = updatedUser.toAuthUser()
        _session.value = AuthSession(user = authUser)
        return AuthResult.Success(authUser)
    }

    fun getUsers(): List<AuthUser> {
        return readUsers().map { user -> user.toAuthUser() }
    }

    fun clearError() {
        _session.value = _session.value.copy(errorMessage = null)
    }

    private fun seedDefaultAdminIfNeeded() {
        val users = readUsers()
        if (users.any { it.role == AuthRole.Admin }) return

        val salt = generateSalt()
        val admin = StoredUser(
            id = DEFAULT_ADMIN_ID,
            account = DEFAULT_ADMIN_ACCOUNT,
            role = AuthRole.Admin,
            displayName = "AquaPulse Admin",
            salt = salt,
            passwordHash = hashPassword(DEFAULT_ADMIN_PASSWORD, salt),
            createdAt = System.currentTimeMillis(),
        )
        writeUsers(users + admin)
    }

    private fun readSessionUser(): AuthUser? {
        val sessionUserId = preferences.getString(KEY_SESSION_USER_ID, null) ?: return null
        return readUsers().firstOrNull { it.id == sessionUserId }?.toAuthUser()
    }

    private fun saveSession(userId: String) {
        preferences.edit()
            .putString(KEY_SESSION_USER_ID, userId)
            .apply()
    }

    private fun readUsers(): List<StoredUser> {
        val json = preferences.getString(KEY_USERS, null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(json)
            List(array.length()) { index ->
                array.getJSONObject(index).toStoredUser()
            }
        }.getOrElse { emptyList() }
    }

    private fun writeUsers(users: List<StoredUser>) {
        val array = JSONArray()
        users.forEach { user ->
            array.put(user.toJson())
        }
        preferences.edit()
            .putString(KEY_USERS, array.toString())
            .apply()
    }

    private fun validateCredentials(account: String, password: String): String? {
        if (account.isBlank()) return "Email or phone number is required."
        if (password.length < MIN_PASSWORD_LENGTH) {
            return "Password must be at least $MIN_PASSWORD_LENGTH characters."
        }
        return null
    }

    private fun generateSalt(): String {
        val bytes = ByteArray(16)
        random.nextBytes(bytes)
        return bytes.base64()
    }

    private fun hashPassword(password: String, salt: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        return digest.digest("$salt:$password".toByteArray(Charsets.UTF_8)).base64()
    }

    private fun ByteArray.base64(): String {
        return Base64.getEncoder().encodeToString(this)
    }

    private fun String.normalizedAccount(): String {
        return trim().lowercase()
    }

    private fun displayNameFor(account: String): String {
        return account.substringBefore("@").ifBlank { "Shrimp Farmer" }
            .replaceFirstChar { char -> char.uppercase() }
    }

    private data class StoredUser(
        val id: String,
        val account: String,
        val role: AuthRole,
        val displayName: String,
        val salt: String,
        val passwordHash: String,
        val createdAt: Long,
    ) {
        fun toAuthUser(): AuthUser {
            return AuthUser(
                id = id,
                account = account,
                role = role,
                displayName = displayName,
                createdAt = createdAt,
            )
        }

        fun toJson(): JSONObject {
            return JSONObject()
                .put("id", id)
                .put("account", account)
                .put("role", role.name)
                .put("displayName", displayName)
                .put("salt", salt)
                .put("passwordHash", passwordHash)
                .put("createdAt", createdAt)
        }
    }

    private fun JSONObject.toStoredUser(): StoredUser {
        return StoredUser(
            id = getString("id"),
            account = getString("account"),
            role = AuthRole.valueOf(getString("role")),
            displayName = getString("displayName"),
            salt = getString("salt"),
            passwordHash = getString("passwordHash"),
            createdAt = getLong("createdAt"),
        )
    }

    companion object {
        const val DEFAULT_ADMIN_ACCOUNT = "admin@aquapulse.local"
        const val DEFAULT_ADMIN_PASSWORD = "Admin@123"

        private const val PREFERENCES_NAME = "aquapulse_auth"
        private const val KEY_USERS = "users"
        private const val KEY_SESSION_USER_ID = "session_user_id"
        private const val DEFAULT_ADMIN_ID = "default-admin"
        private const val MIN_PASSWORD_LENGTH = 6
    }
}
