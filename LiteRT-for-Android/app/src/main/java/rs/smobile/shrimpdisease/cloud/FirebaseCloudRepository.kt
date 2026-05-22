package rs.smobile.shrimpdisease.cloud

import android.content.Context
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.firebase.FirebaseApp
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.DocumentSnapshot
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.SetOptions
import dagger.hilt.android.qualifiers.ApplicationContext
import rs.smobile.shrimpdisease.auth.AuthRole
import rs.smobile.shrimpdisease.auth.AuthUser
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.profile.FarmerProfileUiState
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class FirebaseCloudRepository @Inject constructor(
    @param:ApplicationContext private val context: Context,
) {
    val isEnabled: Boolean
        get() = runCatching { FirebaseApp.getApps(context).isNotEmpty() }.getOrDefault(false)

    fun signIn(
        account: String,
        password: String,
    ): AuthUser? {
        if (!canUseEmailAuth(account)) return null
        return withFirebase("signIn") {
            val authResult = auth().signInWithEmailAndPassword(account, password).awaitBlocking()
            val uid = authResult.user?.uid ?: return@withFirebase null
            fetchUser(uid) ?: AuthUser(
                id = uid,
                account = account,
                role = AuthRole.Farmer,
                displayName = displayNameFor(account),
                createdAt = System.currentTimeMillis(),
            ).also { user -> upsertUser(user) }
        }
    }

    fun registerFarmer(
        account: String,
        password: String,
        displayName: String,
    ): AuthUser? {
        if (!canUseEmailAuth(account)) return null
        return withFirebase("registerFarmer") {
            val authResult = auth().createUserWithEmailAndPassword(account, password).awaitBlocking()
            val uid = authResult.user?.uid ?: return@withFirebase null
            AuthUser(
                id = uid,
                account = account,
                role = AuthRole.Farmer,
                displayName = displayName.trim().ifBlank { displayNameFor(account) },
                createdAt = System.currentTimeMillis(),
            ).also { user -> upsertUser(user) }
        }
    }

    fun logout() {
        if (!isEnabled) return
        runCatching { auth().signOut() }
    }

    fun fetchUser(userId: String): AuthUser? {
        return withFirebase("fetchUser") {
            db().collection(USERS).document(userId).get().awaitBlocking().toAuthUser()
        }
    }

    fun fetchUsers(): List<AuthUser>? {
        return withFirebase("fetchUsers") {
            db().collection(USERS)
                .get()
                .awaitBlocking()
                .documents
                .mapNotNull { document -> document.toAuthUser() }
                .sortedByDescending { user -> user.createdAt }
        }
    }

    fun upsertUser(user: AuthUser): Boolean {
        return withFirebase("upsertUser") {
            db().collection(USERS)
                .document(user.id)
                .set(user.toFirestoreMap(), SetOptions.merge())
                .awaitBlocking()
            true
        } ?: false
    }

    fun deleteUser(userId: String): Boolean {
        return withFirebase("deleteUser") {
            db().collection(USERS).document(userId).delete().awaitBlocking()
            db().collection(FARMER_PROFILES).document(userId).delete().awaitBlocking()
            clearPredictionLogs(userId)
            true
        } ?: false
    }

    fun fetchProfile(userId: String): FarmerProfileUiState? {
        return withFirebase("fetchProfile") {
            db().collection(FARMER_PROFILES).document(userId).get().awaitBlocking().toFarmerProfile()
        }
    }

    fun upsertProfile(profile: FarmerProfileUiState): Boolean {
        val userId = profile.userId ?: return false
        return withFirebase("upsertProfile") {
            db().collection(FARMER_PROFILES)
                .document(userId)
                .set(profile.toFirestoreMap(), SetOptions.merge())
                .awaitBlocking()
            true
        } ?: false
    }

    fun deleteProfile(userId: String): Boolean {
        return withFirebase("deleteProfile") {
            db().collection(FARMER_PROFILES).document(userId).delete().awaitBlocking()
            true
        } ?: false
    }

    fun fetchPredictionLogs(ownerId: String): List<PredictionLogItem>? {
        return withFirebase("fetchPredictionLogs") {
            db().collection(PREDICTION_LOGS)
                .whereEqualTo("ownerId", ownerId)
                .get()
                .awaitBlocking()
                .documents
                .mapNotNull { document -> document.toPredictionLogItem() }
                .sortedByDescending { log -> log.timestamp }
        }
    }

    fun upsertPredictionLog(
        ownerId: String,
        logItem: PredictionLogItem,
    ): Boolean {
        return withFirebase("upsertPredictionLog") {
            db().collection(PREDICTION_LOGS)
                .document(predictionLogDocumentId(ownerId, logItem.id))
                .set(logItem.toFirestoreMap(ownerId), SetOptions.merge())
                .awaitBlocking()
            true
        } ?: false
    }

    fun updatePredictionLog(
        ownerId: String,
        logId: Long,
        predictedClass: String,
        confidence: Float,
    ): Boolean {
        return withFirebase("updatePredictionLog") {
            db().collection(PREDICTION_LOGS)
                .document(predictionLogDocumentId(ownerId, logId))
                .set(
                    mapOf(
                        "predictedClass" to predictedClass,
                        "confidence" to confidence.coerceIn(0f, 1f),
                    ),
                    SetOptions.merge(),
                )
                .awaitBlocking()
            true
        } ?: false
    }

    fun deletePredictionLog(
        ownerId: String,
        logId: Long,
    ): Boolean {
        return withFirebase("deletePredictionLog") {
            db().collection(PREDICTION_LOGS)
                .document(predictionLogDocumentId(ownerId, logId))
                .delete()
                .awaitBlocking()
            true
        } ?: false
    }

    fun clearPredictionLogs(ownerId: String): Boolean {
        return withFirebase("clearPredictionLogs") {
            val documents = db().collection(PREDICTION_LOGS)
                .whereEqualTo("ownerId", ownerId)
                .get()
                .awaitBlocking()
                .documents
            documents.forEach { document ->
                document.reference.delete().awaitBlocking()
            }
            true
        } ?: false
    }

    private fun auth(): FirebaseAuth = FirebaseAuth.getInstance()

    private fun db(): FirebaseFirestore = FirebaseFirestore.getInstance()

    private inline fun <T> withFirebase(
        operation: String,
        block: () -> T,
    ): T? {
        if (!isEnabled) return null
        return runCatching(block)
            .onFailure { error -> Log.w(TAG, "Firebase $operation failed.", error) }
            .getOrNull()
    }

    private fun <T> Task<T>.awaitBlocking(): T {
        return Tasks.await(this, FIREBASE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
    }

    private fun canUseEmailAuth(account: String): Boolean {
        return isEnabled && account.contains("@")
    }

    private fun AuthUser.toFirestoreMap(): Map<String, Any?> {
        return mapOf(
            "id" to id,
            "account" to account,
            "role" to role.name,
            "displayName" to displayName,
            "createdAt" to createdAt,
            "updatedAt" to System.currentTimeMillis(),
        )
    }

    private fun DocumentSnapshot.toAuthUser(): AuthUser? {
        if (!exists()) return null
        val account = getString("account").orEmpty()
        if (account.isBlank()) return null
        return AuthUser(
            id = getString("id").orEmpty().ifBlank { id },
            account = account,
            role = parseRole(getString("role")),
            displayName = getString("displayName").orEmpty().ifBlank { displayNameFor(account) },
            createdAt = getLong("createdAt") ?: System.currentTimeMillis(),
        )
    }

    private fun FarmerProfileUiState.toFirestoreMap(): Map<String, Any?> {
        return mapOf(
            "userId" to userId,
            "displayName" to displayName,
            "farmLocation" to farmLocation,
            "phoneNumber" to phoneNumber,
            "email" to email,
            "avatarUri" to avatarUri,
            "dataPermissionEnabled" to dataPermissionEnabled,
            "updatedAt" to System.currentTimeMillis(),
        )
    }

    private fun DocumentSnapshot.toFarmerProfile(): FarmerProfileUiState? {
        if (!exists()) return null
        return FarmerProfileUiState(
            userId = getString("userId").orEmpty().ifBlank { id },
            displayName = getString("displayName").orEmpty().ifBlank { "Shrimp Farmer" },
            farmLocation = getString("farmLocation").orEmpty().ifBlank { "Coastal pond" },
            phoneNumber = getString("phoneNumber").orEmpty().ifBlank { "Not updated" },
            email = getString("email").orEmpty().ifBlank { "Not updated" },
            avatarUri = getString("avatarUri").orEmpty().ifBlank { null },
            dataPermissionEnabled = getBoolean("dataPermissionEnabled") ?: true,
        )
    }

    private fun PredictionLogItem.toFirestoreMap(ownerId: String): Map<String, Any?> {
        return mapOf(
            "ownerId" to ownerId,
            "imageUri" to imageUri,
            "predictedClass" to predictedClass,
            "confidence" to confidence,
            "top3Predictions" to top3Predictions,
            "inferenceTimeMs" to inferenceTimeMs,
            "speed" to speed,
            "fps" to fps,
            "modelName" to modelName,
            "threshold" to threshold,
            "isAboveThreshold" to isAboveThreshold,
            "groundTruthLabel" to groundTruthLabel,
            "isCorrect" to isCorrect,
            "timestamp" to timestamp,
            "averageInferenceTimeMs" to averageInferenceTimeMs,
            "averageSpeed" to averageSpeed,
            "averageFps" to averageFps,
            "runningAccuracy" to runningAccuracy,
            "updatedAt" to System.currentTimeMillis(),
        )
    }

    private fun DocumentSnapshot.toPredictionLogItem(): PredictionLogItem? {
        if (!exists()) return null
        val timestamp = getLong("timestamp") ?: id.substringAfterLast("-").toLongOrNull() ?: return null
        return PredictionLogItem(
            imageUri = getString("imageUri").orEmpty().ifBlank { null },
            predictedClass = getString("predictedClass").orEmpty().ifBlank { "Unknown / Low confidence" },
            confidence = getFloat("confidence"),
            top3Predictions = getString("top3Predictions").orEmpty(),
            inferenceTimeMs = getLong("inferenceTimeMs") ?: 0L,
            speed = getFloat("speed"),
            fps = getFloat("fps"),
            modelName = getString("modelName").orEmpty(),
            threshold = getFloat("threshold"),
            isAboveThreshold = getBoolean("isAboveThreshold") ?: false,
            groundTruthLabel = getString("groundTruthLabel").orEmpty().ifBlank { null },
            isCorrect = getBoolean("isCorrect"),
            timestamp = timestamp,
            thumbnail = null,
            averageInferenceTimeMs = getFloat("averageInferenceTimeMs"),
            averageSpeed = getFloat("averageSpeed"),
            averageFps = getFloat("averageFps"),
            runningAccuracy = getNullableFloat("runningAccuracy"),
        )
    }

    private fun DocumentSnapshot.getFloat(field: String): Float {
        return getDouble(field)?.toFloat() ?: 0f
    }

    private fun DocumentSnapshot.getNullableFloat(field: String): Float? {
        return getDouble(field)?.toFloat()
    }

    private fun parseRole(value: String?): AuthRole {
        return AuthRole.entries.firstOrNull { role ->
            role.name.equals(value, ignoreCase = true)
        } ?: AuthRole.Farmer
    }

    private fun displayNameFor(account: String): String {
        return account.substringBefore("@").ifBlank { "Shrimp Farmer" }
            .replaceFirstChar { char -> char.uppercase() }
    }

    private fun predictionLogDocumentId(ownerId: String, logId: Long): String {
        return "$ownerId-$logId"
    }

    private companion object {
        private const val TAG = "FirebaseCloudRepo"
        private const val FIREBASE_TIMEOUT_SECONDS = 6L
        private const val USERS = "users"
        private const val FARMER_PROFILES = "farmer_profiles"
        private const val PREDICTION_LOGS = "prediction_logs"
    }
}
