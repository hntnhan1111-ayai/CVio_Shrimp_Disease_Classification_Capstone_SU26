package rs.smobile.shrimpdisease.ui.profile

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsFocusedAsState
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import rs.smobile.shrimpdisease.profile.FarmerProfileUiState
import rs.smobile.shrimpdisease.profile.FarmerProfileUpdate
import rs.smobile.shrimpdisease.ui.components.AppBrandLogo
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioIconBubble
import rs.smobile.shrimpdisease.ui.components.DataPermissionToggle
import rs.smobile.shrimpdisease.ui.components.ShrimpIconButton
import rs.smobile.shrimpdisease.ui.components.ShrimpLineIcon
import rs.smobile.shrimpdisease.ui.components.ShrimpNavIcon
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerHigh
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerHighest
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest

@Composable
fun ProfileScreen(
    profileUiState: FarmerProfileUiState,
    onDataPermissionChanged: (Boolean) -> Unit,
    onBackHome: () -> Unit,
    onSettings: () -> Unit,
    onUpdateAvatar: () -> Unit,
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
        ProfileTopBar(
            onBackHome = onBackHome,
            onSettings = onSettings,
        )

        Column(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 18.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            ProfileIdentityCard(
                profileUiState = profileUiState,
                onEditProfile = { showEditProfile = true },
                onUpdateAvatar = onUpdateAvatar,
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
private fun ProfileTopBar(
    onBackHome: () -> Unit,
    onSettings: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .padding(horizontal = 20.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        ShrimpIconButton(
            icon = ShrimpNavIcon.Back,
            contentDescription = "Về trang chủ",
            onClick = onBackHome,
        )
        AppBrandLogo()
        ShrimpIconButton(
            icon = ShrimpNavIcon.Settings,
            contentDescription = "Mở cài đặt",
            onClick = onSettings,
        )
    }
}

@Composable
private fun ProfileIdentityCard(
    profileUiState: FarmerProfileUiState,
    onEditProfile: () -> Unit,
    onUpdateAvatar: () -> Unit,
) {
    CVioCard(tonalElevation = 3.dp) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Box(contentAlignment = Alignment.BottomEnd) {
                val avatarModifier = Modifier
                    .size(96.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .clickable(onClick = onUpdateAvatar)

                if (profileUiState.avatarUri != null) {
                    AsyncImage(
                        model = profileUiState.avatarUri,
                        contentDescription = "Ảnh đại diện",
                        modifier = avatarModifier,
                        contentScale = ContentScale.Crop,
                    )
                } else {
                    CVioIconBubble(
                        label = profileUiState.initials.take(2),
                        modifier = avatarModifier,
                        size = 96.dp,
                        containerColor = CVioSurfaceContainerHighest,
                        contentColor = MaterialTheme.colorScheme.primary,
                    )
                }

                Box(
                    modifier = Modifier
                        .size(34.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(MaterialTheme.colorScheme.primaryContainer)
                        .border(
                            width = 2.dp,
                            color = CVioSurfaceContainerLowest,
                            shape = RoundedCornerShape(999.dp),
                        )
                        .clickable(onClick = onUpdateAvatar),
                    contentAlignment = Alignment.Center,
                ) {
                    ShrimpLineIcon(
                        icon = ShrimpNavIcon.Camera,
                        modifier = Modifier.size(18.dp),
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                }
            }

            Text(
                text = profileValueText(profileUiState.displayName),
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
            )
            Text(
                text = profileValueText(profileUiState.farmLocation),
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
                ProfileContactRow(icon = ShrimpNavIcon.Phone, value = profileUiState.phoneNumber)
                ProfileContactRow(icon = ShrimpNavIcon.Email, value = profileUiState.email)
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
                    text = "Sửa hồ sơ",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                )
            }
        }
    }
}

@Composable
private fun ProfileContactRow(
    icon: ShrimpNavIcon,
    value: String,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        ProfileIconBubble(
            icon = icon,
            size = 34.dp,
            containerColor = CVioSurfaceContainerHigh,
            contentColor = MaterialTheme.colorScheme.outline,
        )
        Text(
            text = profileValueText(value),
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
            text = "Quyền dữ liệu",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
        )
        DataPermissionToggle(
            title = "Góp ảnh để cải thiện AI",
            message = "Cho phép dùng ảnh đã ẩn danh để AI nhận biết bệnh tôm tốt hơn.",
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
                text = "Đăng xuất",
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
    var displayName by remember(profileUiState.userId) { mutableStateOf("") }
    var farmLocation by remember(profileUiState.userId) { mutableStateOf("") }
    var phoneNumber by remember(profileUiState.userId) { mutableStateOf("") }
    var email by remember(profileUiState.userId) { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = "Sửa hồ sơ",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                EditableProfileField(
                    value = displayName,
                    onValueChange = { displayName = it },
                    label = "Tên",
                    holder = profileValueText(profileUiState.displayName).ifBlank { "Nhập họ tên" },
                )
                EditableProfileField(
                    value = farmLocation,
                    onValueChange = { farmLocation = it },
                    label = "Ao/trại nuôi",
                    holder = profileValueText(profileUiState.farmLocation).ifBlank { "Nhập vị trí ao hoặc trại nuôi" },
                )
                EditableProfileField(
                    value = phoneNumber,
                    onValueChange = { phoneNumber = it },
                    label = "Số điện thoại",
                    holder = profileValueText(profileUiState.phoneNumber).ifBlank { "Nhập số điện thoại" },
                )
                EditableProfileField(
                    value = email,
                    onValueChange = { email = it },
                    label = "Email",
                    holder = profileValueText(profileUiState.email).ifBlank { "Nhập email" },
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    onSave(
                        FarmerProfileUpdate(
                            displayName = displayName.ifBlank { profileUiState.displayName },
                            farmLocation = farmLocation.ifBlank { profileUiState.farmLocation },
                            phoneNumber = phoneNumber.ifBlank { profileUiState.phoneNumber },
                            email = email.ifBlank { profileUiState.email },
                            avatarUri = profileUiState.avatarUri,
                        )
                    )
                },
            ) {
                Text("Lưu")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Hủy")
            }
        },
    )
}

@Composable
private fun EditableProfileField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    holder: String,
) {
    val interactionSource = remember { MutableInteractionSource() }
    val focused by interactionSource.collectIsFocusedAsState()

    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        placeholder = {
            if (!focused && value.isBlank()) {
                Text(holder)
            }
        },
        interactionSource = interactionSource,
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun ProfileIconBubble(
    icon: ShrimpNavIcon,
    size: Dp,
    containerColor: Color,
    contentColor: Color,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .size(size)
            .clip(RoundedCornerShape(999.dp))
            .background(containerColor),
        contentAlignment = Alignment.Center,
    ) {
        ShrimpLineIcon(
            icon = icon,
            modifier = Modifier.size(size * 0.55f),
            color = contentColor,
        )
    }
}

private fun profileValueText(value: String): String {
    return when (value) {
        "Not set" -> "Chưa cập nhật"
        "Coastal pond" -> "Ao nuôi ven biển"
        "Coastal View Farms, Block A" -> "Ao nuôi ven biển, khu A"
        "Shrimp Farmer" -> "Nông dân nuôi tôm"
        else -> value
    }
}
