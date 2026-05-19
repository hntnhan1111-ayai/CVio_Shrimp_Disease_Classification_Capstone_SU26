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
            farmLocation = update.farmLocation.trim().ifBlank { "Coastal pond" },
            phoneNumber = update.phoneNumber.trim().ifBlank { "Not set" },
            email = update.email.trim().ifBlank { "Not set" },
        )
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

    private fun readProfile(user: AuthUser): FarmerProfileUiState {
        val json = preferences.getString(keyFor(user.id), null)
        if (json.isNullOrBlank()) return defaultProfile(user)

        return runCatching {
            val profile = JSONObject(json)
            FarmerProfileUiState(
                userId = user.id,
                displayName = profile.optString("displayName", user.displayName).ifBlank { user.displayName },
                farmLocation = profile.optString("farmLocation", "Coastal pond").ifBlank { "Coastal pond" },
                phoneNumber = profile.optString("phoneNumber", defaultPhone(user)).ifBlank { "Not set" },
                email = profile.optString("email", defaultEmail(user)).ifBlank { "Not set" },
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
            .put("dataPermissionEnabled", profile.dataPermissionEnabled)

        preferences.edit()
            .putString(keyFor(userId), json.toString())
            .apply()
    }

    private fun defaultProfile(user: AuthUser): FarmerProfileUiState {
        return FarmerProfileUiState(
            userId = user.id,
            displayName = user.displayName,
            farmLocation = "Coastal View Farms, Block A",
            phoneNumber = defaultPhone(user),
            email = defaultEmail(user),
            dataPermissionEnabled = true,
        )
    }

    private fun defaultPhone(user: AuthUser): String {
        return if (user.account.contains("@")) "Not set" else user.account
    }

    private fun defaultEmail(user: AuthUser): String {
        return if (user.account.contains("@")) user.account else "Not set"
    }

    private fun keyFor(userId: String): String {
        return "$KEY_PROFILE_PREFIX$userId"
    }

    private companion object {
        private const val PREFERENCES_NAME = "aquapulse_farmer_profiles"
        private const val KEY_PROFILE_PREFIX = "profile_"
    }
}
