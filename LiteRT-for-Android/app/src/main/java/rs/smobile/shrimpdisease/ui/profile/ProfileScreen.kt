package rs.smobile.shrimpdisease.ui.profile

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.profile.FarmerProfileUiState
import rs.smobile.shrimpdisease.profile.FarmerProfileUpdate
import rs.smobile.shrimpdisease.ui.components.AppBrandLogo
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioIconBubble
import rs.smobile.shrimpdisease.ui.components.DataPermissionToggle
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerHigh
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerHighest
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest

@Composable
fun ProfileScreen(
    profileUiState: FarmerProfileUiState,
    onDataPermissionChanged: (Boolean) -> Unit,
    onSaveProfile: (FarmerProfileUpdate) -> Unit,
    onLogout: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var showEditProfile by remember { mutableStateOf(false) }

    if (showEditProfile) {
        EditProfileDialog(
            profileUiState = profileUiState,
            onDismiss = { showEditProfile = false },
            onSave = { update ->
                onSaveProfile(update)
                showEditProfile = false
            },
        )
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
    ) {
        ProfileTopBar(profileUiState = profileUiState)

        Column(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 18.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            ProfileIdentityCard(
                profileUiState = profileUiState,
                onEditProfile = { showEditProfile = true },
            )
            DataPermissionsCard(
                checked = profileUiState.dataPermissionEnabled,
                onCheckedChange = onDataPermissionChanged,
                onLogout = onLogout,
            )
        }
    }
}

@Composable
private fun ProfileTopBar(profileUiState: FarmerProfileUiState) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .padding(horizontal = 20.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        CVioIconBubble(
            label = profileUiState.initials.take(2),
            size = 40.dp,
            containerColor = CVioSurfaceContainerHighest,
            contentColor = MaterialTheme.colorScheme.primary,
        )
        AppBrandLogo()
        CVioIconBubble(
            label = "N",
            size = 40.dp,
            containerColor = CVioSurfaceContainerLowest,
            contentColor = MaterialTheme.colorScheme.primary,
        )
    }
}

@Composable
private fun ProfileIdentityCard(
    profileUiState: FarmerProfileUiState,
    onEditProfile: () -> Unit,
) {
    CVioCard(tonalElevation = 3.dp) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Box(contentAlignment = Alignment.BottomEnd) {
                CVioIconBubble(
                    label = profileUiState.initials.take(2),
                    size = 96.dp,
                    containerColor = CVioSurfaceContainerHighest,
                    contentColor = MaterialTheme.colorScheme.primary,
                )
                Box(
                    modifier = Modifier
                        .size(34.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(MaterialTheme.colorScheme.primaryContainer)
                        .border(
                            width = 2.dp,
                            color = CVioSurfaceContainerLowest,
                            shape = RoundedCornerShape(999.dp),
                        ),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "E",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }

            Text(
                text = profileUiState.displayName,
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
            )
            Text(
                text = profileUiState.farmLocation,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(CVioSurfaceContainerLow)
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                ProfileContactRow(label = "TEL", value = profileUiState.phoneNumber)
                ProfileContactRow(label = "MAIL", value = profileUiState.email)
            }

            Button(
                onClick = onEditProfile,
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer,
                    contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
                ),
            ) {
                Text(
                    text = "Edit Profile",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                )
            }
        }
    }
}

@Composable
private fun ProfileContactRow(
    label: String,
    value: String,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        CVioIconBubble(
            label = label,
            size = 34.dp,
            containerColor = CVioSurfaceContainerHigh,
            contentColor = MaterialTheme.colorScheme.outline,
        )
        Text(
            text = value,
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun DataPermissionsCard(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    onLogout: () -> Unit,
) {
    CVioCard(tonalElevation = 3.dp) {
        Text(
            text = "Data Permissions",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
        )
        DataPermissionToggle(
            title = "AI Model Improvement",
            message = "Allow my anonymized images to help improve the diagnostic accuracy of the aquaculture AI model.",
            checked = checked,
            onCheckedChange = onCheckedChange,
        )
        HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
        OutlinedButton(
            onClick = onLogout,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = MaterialTheme.colorScheme.error,
            ),
        ) {
            Text(
                text = "Logout",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

@Composable
private fun EditProfileDialog(
    profileUiState: FarmerProfileUiState,
    onDismiss: () -> Unit,
    onSave: (FarmerProfileUpdate) -> Unit,
) {
    var displayName by remember(profileUiState.userId) { mutableStateOf(profileUiState.displayName) }
    var farmLocation by remember(profileUiState.userId) { mutableStateOf(profileUiState.farmLocation) }
    var phoneNumber by remember(profileUiState.userId) { mutableStateOf(profileUiState.phoneNumber) }
    var email by remember(profileUiState.userId) { mutableStateOf(profileUiState.email) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = "Edit Profile",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = displayName,
                    onValueChange = { displayName = it },
                    label = { Text("Name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = farmLocation,
                    onValueChange = { farmLocation = it },
                    label = { Text("Farm / location") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = phoneNumber,
                    onValueChange = { phoneNumber = it },
                    label = { Text("Phone") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = email,
                    onValueChange = { email = it },
                    label = { Text("Email") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    onSave(
                        FarmerProfileUpdate(
                            displayName = displayName,
                            farmLocation = farmLocation,
                            phoneNumber = phoneNumber,
                            email = email,
                        )
                    )
                },
            ) {
                Text("Save")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
    )
}
