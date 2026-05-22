package rs.smobile.shrimpdisease.auth

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import org.json.JSONObject
import rs.smobile.shrimpdisease.cloud.FirebaseCloudRepository
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AuthRepository @Inject constructor(
    @ApplicationContext context: Context,
    private val cloudRepository: FirebaseCloudRepository,
) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
    private val random = SecureRandom()

    private val _session = MutableStateFlow(
        AuthSession(user = readSessionUser())
    )
    val session: StateFlow<AuthSession> = _session

    init {
        seedDefaultAdminIfNeeded()
        refreshUsersFromCloud()
    }

    fun login(
        account: String,
        password: String,
    ): AuthResult {
        val normalizedAccount = account.normalizedAccount()
        val validationError = validateCredentials(normalizedAccount, password)
        if (validationError != null) return AuthResult.Error(validationError)

        refreshUsersFromCloud()
        cloudRepository.signIn(normalizedAccount, password)?.let { cloudUser ->
            cacheCloudUser(cloudUser, password)
            saveSession(cloudUser.id)
            _session.value = AuthSession(user = cloudUser)
            return AuthResult.Success(cloudUser)
        }

        val record = readUsers().firstOrNull {
            it.account == normalizedAccount
        } ?: return AuthResult.Error("Không tìm thấy tài khoản. Vui lòng kiểm tra email hoặc số điện thoại.")

        if (record.passwordHash != hashPassword(password, record.salt)) {
            return AuthResult.Error("Mật khẩu chưa đúng.")
        }

        saveSession(record.id)
        val user = record.toAuthUser()
        cloudRepository.upsertUser(user)
        _session.value = AuthSession(user = user)
        return AuthResult.Success(user)
    }

    fun register(
        account: String,
        password: String,
    ): AuthResult {
        val normalizedAccount = account.normalizedAccount()
        val validationError = validateCredentials(normalizedAccount, password)
        if (validationError != null) return AuthResult.Error(validationError)

        refreshUsersFromCloud()
        val users = readUsers()
        if (users.any { it.account == normalizedAccount }) {
            return AuthResult.Error("Tài khoản này đã được đăng ký.")
        }

        cloudRepository.registerFarmer(
            account = normalizedAccount,
            password = password,
            displayName = displayNameFor(normalizedAccount),
        )?.let { cloudUser ->
            val record = cloudUser.toStoredUser(password = password)
            writeUsers(users + record)
            saveSession(record.id)
            _session.value = AuthSession(user = cloudUser)
            return AuthResult.Success(cloudUser)
        }

        val salt = generateSalt()
        val record = StoredUser(
            id = UUID.randomUUID().toString(),
            account = normalizedAccount,
            role = AuthRole.Farmer,
            displayName = displayNameFor(normalizedAccount),
            salt = salt,
            passwordHash = hashPassword(password, salt),
            createdAt = System.currentTimeMillis(),
        )
        writeUsers(users + record)
        saveSession(record.id)
        val user = record.toAuthUser()
        cloudRepository.upsertUser(user)
        _session.value = AuthSession(user = user)
        return AuthResult.Success(user)
    }

    fun createManagedFarmer(
        account: String,
        password: String,
        displayName: String,
    ): AuthResult {
        val normalizedAccount = account.normalizedAccount()
        val validationError = validateCredentials(normalizedAccount, password)
        if (validationError != null) return AuthResult.Error(validationError)

        refreshUsersFromCloud()
        val users = readUsers()
        if (users.any { it.account == normalizedAccount }) {
            return AuthResult.Error("Tài khoản này đã được đăng ký.")
        }

        val cleanDisplayName = displayName.trim().ifBlank { displayNameFor(normalizedAccount) }
        val salt = generateSalt()
        val record = StoredUser(
            id = UUID.randomUUID().toString(),
            account = normalizedAccount,
            role = AuthRole.Farmer,
            displayName = cleanDisplayName,
            salt = salt,
            passwordHash = hashPassword(password, salt),
            createdAt = System.currentTimeMillis(),
        )
        writeUsers(users + record)
        cloudRepository.upsertUser(record.toAuthUser())
        return AuthResult.Success(record.toAuthUser())
    }

    fun logout() {
        cloudRepository.logout()
        preferences.edit()
            .remove(KEY_SESSION_USER_ID)
            .apply()
        _session.value = AuthSession(user = null)
    }

    fun updateDisplayName(displayName: String): AuthResult {
        val currentUser = _session.value.user ?: return AuthResult.Error("Chưa đăng nhập.")
        val cleanName = displayName.trim()
        if (cleanName.isBlank()) return AuthResult.Error("Cần nhập tên hiển thị.")

        val users = readUsers()
        val updatedUsers = users.map { user ->
            if (user.id == currentUser.id) user.copy(displayName = cleanName) else user
        }
        writeUsers(updatedUsers)

        val updatedUser = updatedUsers.firstOrNull { it.id == currentUser.id }
            ?: return AuthResult.Error("Không tìm thấy tài khoản.")
        val authUser = updatedUser.toAuthUser()
        cloudRepository.upsertUser(authUser)
        _session.value = AuthSession(user = authUser)
        return AuthResult.Success(authUser)
    }

    fun updateManagedFarmer(
        userId: String,
        account: String,
        displayName: String,
    ): AuthResult {
        val normalizedAccount = account.normalizedAccount()
        if (normalizedAccount.isBlank()) return AuthResult.Error("Cần nhập email hoặc số điện thoại.")

        val cleanDisplayName = displayName.trim()
        if (cleanDisplayName.isBlank()) return AuthResult.Error("Cần nhập tên hiển thị.")

        val users = readUsers()
        val existing = users.firstOrNull { user -> user.id == userId && user.role == AuthRole.Farmer }
            ?: return AuthResult.Error("Không tìm thấy người dùng.")
        if (users.any { user -> user.id != userId && user.account == normalizedAccount }) {
            return AuthResult.Error("Tài khoản này đã được đăng ký.")
        }

        val updatedUsers = users.map { user ->
            if (user.id == userId) {
                existing.copy(
                    account = normalizedAccount,
                    displayName = cleanDisplayName,
                )
            } else {
                user
            }
        }
        writeUsers(updatedUsers)

        val updatedUser = updatedUsers.first { user -> user.id == userId }.toAuthUser()
        cloudRepository.upsertUser(updatedUser)
        if (_session.value.user?.id == userId) {
            _session.value = AuthSession(user = updatedUser)
        }
        return AuthResult.Success(updatedUser)
    }

    fun deleteManagedUser(userId: String): Boolean {
        val users = readUsers()
        val updatedUsers = users.filterNot { user ->
            user.id == userId && user.role == AuthRole.Farmer
        }
        if (updatedUsers.size == users.size) return false

        writeUsers(updatedUsers)
        cloudRepository.deleteUser(userId)
        if (_session.value.user?.id == userId) {
            logout()
        }
        return true
    }

    fun getUsers(): List<AuthUser> {
        refreshUsersFromCloud()
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
            displayName = "CVio Admin",
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

    private fun refreshUsersFromCloud() {
        val cloudUsers = cloudRepository.fetchUsers() ?: return
        val localUsers = readUsers()
        val localById = localUsers.associateBy { user -> user.id }
        val cloudRecords = cloudUsers.map { cloudUser ->
            cloudUser.toStoredUser(existing = localById[cloudUser.id])
        }
        val cloudIds = cloudRecords.map { user -> user.id }.toSet()
        val mergedUsers = cloudRecords + localUsers.filterNot { user -> user.id in cloudIds }
        writeUsers(mergedUsers.distinctBy { user -> user.id })
    }

    private fun cacheCloudUser(
        user: AuthUser,
        password: String,
    ) {
        val users = readUsers()
        val existing = users.firstOrNull { record -> record.id == user.id }
        val record = user.toStoredUser(password = password, existing = existing)
        writeUsers(users.filterNot { item -> item.id == user.id } + record)
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
        if (account.isBlank()) return "Cần nhập email hoặc số điện thoại."
        if (password.length < MIN_PASSWORD_LENGTH) {
            return "Mật khẩu cần ít nhất $MIN_PASSWORD_LENGTH ký tự."
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
        return account.substringBefore("@").ifBlank { "Nông dân nuôi tôm" }
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

    private fun AuthUser.toStoredUser(
        password: String? = null,
        existing: StoredUser? = null,
    ): StoredUser {
        val nextSalt = if (password == null) {
            existing?.salt.orEmpty()
        } else {
            generateSalt()
        }
        val nextPasswordHash = if (password == null) {
            existing?.passwordHash.orEmpty()
        } else {
            hashPassword(password, nextSalt)
        }
        return StoredUser(
            id = id,
            account = account,
            role = role,
            displayName = displayName,
            salt = nextSalt,
            passwordHash = nextPasswordHash,
            createdAt = createdAt,
        )
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
        const val DEFAULT_ADMIN_ACCOUNT = "admin@cvio.local"
        const val DEFAULT_ADMIN_PASSWORD = "Admin@123"

        private const val PREFERENCES_NAME = "cvio_auth"
        private const val KEY_USERS = "users"
        private const val KEY_SESSION_USER_ID = "session_user_id"
        private const val DEFAULT_ADMIN_ID = "default-admin"
        private const val MIN_PASSWORD_LENGTH = 6
    }
}
