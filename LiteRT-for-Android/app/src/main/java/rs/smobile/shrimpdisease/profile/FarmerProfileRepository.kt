package rs.smobile.shrimpdisease.profile

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONObject
import rs.smobile.shrimpdisease.auth.AuthUser
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class FarmerProfileRepository @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    private val _profile = MutableStateFlow(FarmerProfileUiState())
    val profile: StateFlow<FarmerProfileUiState> = _profile

    private var currentUser: AuthUser? = null

    fun setUser(user: AuthUser?) {
        currentUser = user
        _profile.value = if (user == null) {
            FarmerProfileUiState()
        } else {
            readProfile(user)
        }
    }

    fun updateProfile(update: FarmerProfileUpdate): Boolean {
        val user = currentUser ?: return false
        val updated = _profile.value.copy(
            displayName = update.displayName.trim().ifBlank { user.displayName },
            farmLocation = update.farmLocation.trim().ifBlank { "Ao nuôi ven biển" },
            phoneNumber = update.phoneNumber.trim().ifBlank { "Chưa cập nhật" },
            email = update.email.trim().ifBlank { "Chưa cập nhật" },
            avatarUri = update.avatarUri,
        )
        writeProfile(user.id, updated)
        _profile.value = updated
        return true
    }

    fun setAvatarUri(avatarUri: String?): Boolean {
        val user = currentUser ?: return false
        val updated = _profile.value.copy(avatarUri = avatarUri)
        writeProfile(user.id, updated)
        _profile.value = updated
        return true
    }

    fun setDataPermissionEnabled(enabled: Boolean): Boolean {
        val user = currentUser ?: return false
        val updated = _profile.value.copy(dataPermissionEnabled = enabled)
        writeProfile(user.id, updated)
        _profile.value = updated
        return true
    }

    fun getProfileForUser(user: AuthUser): FarmerProfileUiState {
        return readProfile(user)
    }

    fun saveProfileForUser(
        user: AuthUser,
        profile: FarmerProfileUiState,
    ): Boolean {
        val normalizedProfile = profile.copy(
            userId = user.id,
            displayName = profile.displayName.trim().ifBlank { user.displayName },
            farmLocation = profile.farmLocation.trim().ifBlank { "Ao nuôi ven biển" },
            phoneNumber = profile.phoneNumber.trim().ifBlank { defaultPhone(user) },
            email = profile.email.trim().ifBlank { defaultEmail(user) },
        )
        writeProfile(user.id, normalizedProfile)
        if (currentUser?.id == user.id) {
            _profile.value = normalizedProfile
        }
        return true
    }

    private fun readProfile(user: AuthUser): FarmerProfileUiState {
        val json = preferences.getString(keyFor(user.id), null)
        if (json.isNullOrBlank()) return defaultProfile(user)

        return runCatching {
            val profile = JSONObject(json)
            FarmerProfileUiState(
                userId = user.id,
                displayName = profile.optString("displayName", user.displayName).ifBlank { user.displayName },
                farmLocation = profile.optString("farmLocation", "Ao nuôi ven biển").ifBlank { "Ao nuôi ven biển" },
                phoneNumber = profile.optString("phoneNumber", defaultPhone(user)).ifBlank { "Chưa cập nhật" },
                email = profile.optString("email", defaultEmail(user)).ifBlank { "Chưa cập nhật" },
                avatarUri = profile.optString("avatarUri", "").ifBlank { null },
                dataPermissionEnabled = profile.optBoolean("dataPermissionEnabled", true),
            )
        }.getOrElse {
            defaultProfile(user)
        }
    }

    private fun writeProfile(userId: String, profile: FarmerProfileUiState) {
        val json = JSONObject()
            .put("displayName", profile.displayName)
            .put("farmLocation", profile.farmLocation)
            .put("phoneNumber", profile.phoneNumber)
            .put("email", profile.email)
            .put("avatarUri", profile.avatarUri.orEmpty())
            .put("dataPermissionEnabled", profile.dataPermissionEnabled)

        preferences.edit()
            .putString(keyFor(userId), json.toString())
            .apply()
    }

    private fun defaultProfile(user: AuthUser): FarmerProfileUiState {
        return FarmerProfileUiState(
            userId = user.id,
            displayName = user.displayName,
            farmLocation = "Ao nuôi ven biển, khu A",
            phoneNumber = defaultPhone(user),
            email = defaultEmail(user),
            avatarUri = null,
            dataPermissionEnabled = true,
        )
    }

    private fun defaultPhone(user: AuthUser): String {
        return if (user.account.contains("@")) "Chưa cập nhật" else user.account
    }

    private fun defaultEmail(user: AuthUser): String {
        return if (user.account.contains("@")) user.account else "Chưa cập nhật"
    }

    private fun keyFor(userId: String): String {
        return "$KEY_PROFILE_PREFIX$userId"
    }

    private companion object {
        private const val PREFERENCES_NAME = "aquapulse_farmer_profiles"
        private const val KEY_PROFILE_PREFIX = "profile_"
    }
}
