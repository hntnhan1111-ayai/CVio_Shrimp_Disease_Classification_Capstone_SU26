package rs.smobile.shrimpdisease.ui.admin

import androidx.compose.foundation.background
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.data.AdminActivityItem
import rs.smobile.shrimpdisease.data.AdminActivityKind
import rs.smobile.shrimpdisease.data.AdminChartPoint
import rs.smobile.shrimpdisease.data.AdminCreateUserInput
import rs.smobile.shrimpdisease.data.AdminDashboardUiState
import rs.smobile.shrimpdisease.data.AdminDataReviewItem
import rs.smobile.shrimpdisease.data.AdminUserSummary
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.components.EmptyState
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import java.util.Locale
import kotlin.math.max

@Composable
fun AdminDashboardScreen(
    uiState: AdminDashboardUiState,
    currentUserName: String?,
    onRefresh: () -> Unit,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            AdminTopBar(
                currentUserName = currentUserName,
                onTrailingClick = onRefresh,
            )
        }

        item {
            Column(
                modifier = Modifier.padding(horizontal = 20.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = "Admin Dashboard",
                    style = MaterialTheme.typography.displayLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "System Overview & Analytics",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        item {
            Column(
                modifier = Modifier.padding(horizontal = 20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    DashboardStatCard(
                        label = "Total Farmers",
                        value = compactNumber(uiState.totalFarmers),
                        icon = "U",
                        trend = uiState.farmersTrend,
                        modifier = Modifier.weight(1f),
                    )
                    DashboardStatCard(
                        label = "Total Checks",
                        value = compactNumber(uiState.totalDiseaseChecks),
                        icon = "C",
                        accent = MaterialTheme.colorScheme.secondary,
                        trend = uiState.checksTrend,
                        modifier = Modifier.weight(1f),
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    DashboardStatCard(
                        label = "Images Processed",
                        value = compactNumber(uiState.imagesProcessed),
                        icon = "IMG",
                        accent = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.weight(1f),
                    )
                    DashboardStatCard(
                        label = "Active Alerts",
                        value = compactNumber(uiState.activeAlerts),
                        icon = "!",
                        accent = MaterialTheme.colorScheme.error,
                        trend = if (uiState.activeAlerts > 0) "Requires Action" else "Clear",
                        trendIsWarning = uiState.activeAlerts > 0,
                        modifier = Modifier.weight(1f),
                    )
                }
            }
        }

        item {
            DiagnosisVolumeCard(
                points = uiState.diagnosisVolume,
                modifier = Modifier.padding(horizontal = 20.dp),
            )
        }

        item {
            RecentActivityCard(
                activities = uiState.recentActivities,
                modifier = Modifier.padding(horizontal = 20.dp),
            )
        }
    }
}

@Composable
fun AdminUsersScreen(
    users: List<AdminUserSummary>,
    currentUserName: String?,
    onCreateUser: (AdminCreateUserInput) -> Boolean,
    modifier: Modifier = Modifier,
) {
    var searchQuery by rememberSaveable { mutableStateOf("") }
    var showCreateDialog by rememberSaveable { mutableStateOf(false) }
    var selectedUser by remember { mutableStateOf<AdminUserSummary?>(null) }
    val filteredUsers = remember(users, searchQuery) {
        val query = searchQuery.trim()
        if (query.isBlank()) {
            users
        } else {
            users.filter { user ->
                user.name.contains(query, ignoreCase = true) ||
                    user.account.contains(query, ignoreCase = true) ||
                    user.farmLocation.contains(query, ignoreCase = true) ||
                    user.status.contains(query, ignoreCase = true)
            }
        }
    }

    BoxWithConstraints(modifier = modifier.fillMaxSize()) {
        val columns = if (maxWidth >= 720.dp) 2 else 1
        val cardRows = (filteredUsers.map<AdminUserSummary, AdminUserSummary?> { user -> user } + null)
            .chunked(columns)

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            item {
                AdminTopBar(currentUserName = currentUserName)
            }
            item {
                Column(
                    modifier = Modifier.padding(horizontal = 20.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Bottom,
                    ) {
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(4.dp),
                        ) {
                            Text(
                                text = "User Management",
                                style = MaterialTheme.typography.displayLarge,
                                color = MaterialTheme.colorScheme.onSurface,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(
                                text = "Manage aquaculture farmers and field technicians.",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                    SearchUsersField(
                        value = searchQuery,
                        onValueChange = { searchQuery = it },
                    )
                }
            }

            if (users.isEmpty()) {
                item {
                    EmptyUsersCard(
                        onAddUser = { showCreateDialog = true },
                        modifier = Modifier.padding(horizontal = 20.dp),
                    )
                }
            } else if (filteredUsers.isEmpty()) {
                item {
                    CVioCard(
                        modifier = Modifier.padding(horizontal = 20.dp),
                        containerColor = CVioSurfaceContainerLow,
                    ) {
                        EmptyState(
                            title = "No matching users",
                            message = "Try another name, email, pond, or status.",
                        )
                    }
                }
            } else {
                items(cardRows) { rowUsers ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp),
                        horizontalArrangement = Arrangement.spacedBy(16.dp),
                    ) {
                        rowUsers.forEach { user ->
                            if (user == null) {
                                AddUserCard(
                                    onClick = { showCreateDialog = true },
                                    modifier = Modifier.weight(1f),
                                )
                            } else {
                                UserBentoCard(
                                    user = user,
                                    onClick = { selectedUser = user },
                                    modifier = Modifier.weight(1f),
                                )
                            }
                        }
                        repeat(columns - rowUsers.size) {
                            Box(modifier = Modifier.weight(1f))
                        }
                    }
                }
            }
        }
    }

    if (showCreateDialog) {
        AddUserDialog(
            onDismiss = { showCreateDialog = false },
            onCreateUser = { input ->
                if (onCreateUser(input)) {
                    showCreateDialog = false
                }
            },
        )
    }

    selectedUser?.let { user ->
        UserDetailsDialog(
            user = user,
            onDismiss = { selectedUser = null },
        )
    }
}

@Composable
fun AdminDataControlScreen(
    dataItems: List<AdminDataReviewItem>,
    currentUserName: String?,
    onMarkReviewed: (AdminDataReviewItem) -> Unit,
    onExcludeFromTraining: (AdminDataReviewItem) -> Unit,
    onExportMetadata: () -> Unit,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            AdminTopBar(currentUserName = currentUserName)
        }
        item {
            CVioSectionHeader(
                title = "Data Control",
                subtitle = "Review permitted image data before model improvement workflows.",
                modifier = Modifier.padding(horizontal = 20.dp),
            )
        }
        item {
            PrimaryActionButton(
                text = "Export metadata",
                onClick = onExportMetadata,
                enabled = dataItems.isNotEmpty(),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp),
            )
        }

        if (dataItems.isEmpty()) {
            item {
                CVioCard(
                    modifier = Modifier.padding(horizontal = 20.dp),
                    containerColor = CVioSurfaceContainerLow,
                ) {
                    EmptyState(
                        title = "No permitted data yet",
                        message = "Farmer diagnosis logs with data permission enabled will appear here.",
                    )
                }
            }
        } else {
            items(items = dataItems, key = { item -> item.id }) { item ->
                DataReviewCard(
                    item = item,
                    onMarkReviewed = onMarkReviewed,
                    onExcludeFromTraining = onExcludeFromTraining,
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }
        }
    }
}

@Composable
private fun AdminTopBar(
    currentUserName: String?,
    modifier: Modifier = Modifier,
    onTrailingClick: (() -> Unit)? = null,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .padding(horizontal = 20.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(CVioSurfaceContainerLow),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = initialsFor(currentUserName ?: "Admin"),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                )
            }
            Text(
                text = "AquaPulse",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }

        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(CVioSurfaceContainerLowest)
                .then(
                    if (onTrailingClick == null) {
                        Modifier
                    } else {
                        Modifier.clickable(onClick = onTrailingClick)
                    }
                ),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = if (onTrailingClick == null) "N" else "R",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

@Composable
private fun SearchUsersField(
    value: String,
    onValueChange: (String) -> Unit,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
        shape = RoundedCornerShape(999.dp),
        leadingIcon = {
            Text(
                text = "S",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.outline,
                fontWeight = FontWeight.Bold,
            )
        },
        placeholder = {
            Text(
                text = "Search users...",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.outline,
            )
        },
        textStyle = MaterialTheme.typography.bodyMedium,
    )
}

@Composable
private fun UserBentoCard(
    user: AdminUserSummary,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val isActive = user.status == "Active"
    Card(
        modifier = modifier
            .heightIn(min = 220.dp)
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(32.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
    ) {
        Column {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .background(
                        if (isActive) {
                            MaterialTheme.colorScheme.secondary.copy(alpha = 0.2f)
                        } else {
                            MaterialTheme.colorScheme.error.copy(alpha = 0.2f)
                        }
                    )
            )
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.Top,
                ) {
                    Box(
                        modifier = Modifier
                            .size(48.dp)
                            .clip(RoundedCornerShape(999.dp))
                            .background(CVioSurfaceContainerLow),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = initialsFor(user.name),
                            style = MaterialTheme.typography.headlineSmall,
                            color = if (isActive) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.onSurfaceVariant
                            },
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp),
                    ) {
                        Text(
                            text = user.name,
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.onSurface,
                            fontWeight = FontWeight.Bold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Text(
                            text = user.account,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                    CVioStatusChip(
                        text = user.status,
                        containerColor = if (isActive) {
                            MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.35f)
                        } else {
                            MaterialTheme.colorScheme.errorContainer
                        },
                        contentColor = if (isActive) {
                            MaterialTheme.colorScheme.onSecondaryContainer
                        } else {
                            MaterialTheme.colorScheme.onErrorContainer
                        },
                    )
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(CVioSurfaceContainerLow)
                        .padding(12.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    UserMetricBlock(
                        label = "Checks Count",
                        value = user.diseaseCheckCount.toString(),
                        active = isActive,
                        modifier = Modifier.weight(1f),
                    )
                    UserMetricBlock(
                        label = "Last Active",
                        value = user.lastActive,
                        active = isActive,
                        modifier = Modifier.weight(1f),
                    )
                }

                HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "View Details",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.secondary,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = " ->",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.secondary,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }
    }
}

@Composable
private fun UserMetricBlock(
    label: String,
    value: String,
    active: Boolean,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.outline,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        Text(
            text = value,
            style = if (label == "Checks Count") {
                MaterialTheme.typography.headlineSmall
            } else {
                MaterialTheme.typography.bodyMedium
            },
            color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = if (label == "Checks Count") FontWeight.Bold else FontWeight.Medium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun AddUserCard(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier
            .heightIn(min = 220.dp)
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(32.dp),
        colors = CardDefaults.cardColors(containerColor = CVioSurfaceContainerLow),
        border = BorderStroke(2.dp, MaterialTheme.colorScheme.outlineVariant),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "+",
                    style = MaterialTheme.typography.displayLarge,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                )
            }
            Text(
                text = "Add New User",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Register a new farmer or technician to the system.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun EmptyUsersCard(
    onAddUser: () -> Unit,
    modifier: Modifier = Modifier,
) {
    CVioCard(
        modifier = modifier,
        containerColor = CVioSurfaceContainerLow,
    ) {
        EmptyState(
            title = "No farmers yet",
            message = "Create the first farmer account for this admin console.",
        )
        PrimaryActionButton(
            text = "Add New User",
            onClick = onAddUser,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun AddUserDialog(
    onDismiss: () -> Unit,
    onCreateUser: (AdminCreateUserInput) -> Unit,
) {
    var displayName by rememberSaveable { mutableStateOf("") }
    var account by rememberSaveable { mutableStateOf("") }
    var password by rememberSaveable { mutableStateOf("") }
    var farmLocation by rememberSaveable { mutableStateOf("") }
    var phoneNumber by rememberSaveable { mutableStateOf("") }
    var email by rememberSaveable { mutableStateOf("") }
    val canSubmit = account.isNotBlank() && password.length >= 6

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = "Add New User",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = displayName,
                    onValueChange = { displayName = it },
                    label = { Text("Full name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = account,
                    onValueChange = {
                        account = it
                        if (email.isBlank() && it.contains("@")) email = it
                    },
                    label = { Text("Login email or phone") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("Temporary password") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = farmLocation,
                    onValueChange = { farmLocation = it },
                    label = { Text("Farm location") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedTextField(
                        value = phoneNumber,
                        onValueChange = { phoneNumber = it },
                        label = { Text("Phone") },
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone),
                        modifier = Modifier.weight(1f),
                    )
                    OutlinedTextField(
                        value = email,
                        onValueChange = { email = it },
                        label = { Text("Email") },
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                        modifier = Modifier.weight(1f),
                    )
                }
                Text(
                    text = "Password must be at least 6 characters.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        confirmButton = {
            TextButton(
                enabled = canSubmit,
                onClick = {
                    onCreateUser(
                        AdminCreateUserInput(
                            displayName = displayName,
                            account = account,
                            password = password,
                            farmLocation = farmLocation,
                            phoneNumber = phoneNumber,
                            email = email,
                        )
                    )
                },
            ) {
                Text("Create")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
    )
}

@Composable
private fun UserDetailsDialog(
    user: AdminUserSummary,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(CVioSurfaceContainerLow),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = initialsFor(user.name),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Bold,
                    )
                }
                Column {
                    Text(
                        text = user.name,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = user.account,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                DetailLine(label = "Status", value = user.status)
                DetailLine(label = "Checks Count", value = user.diseaseCheckCount.toString())
                DetailLine(label = "Last Active", value = user.lastActive)
                DetailLine(label = "Farm Location", value = user.farmLocation)
                DetailLine(
                    label = "Data Permission",
                    value = if (user.dataPermissionEnabled) "Allowed" else "Disabled",
                )
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        },
    )
}

@Composable
private fun DetailLine(
    label: String,
    value: String,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(0.9f),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            modifier = Modifier
                .weight(1.1f)
                .padding(start = 12.dp),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.SemiBold,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun DashboardStatCard(
    label: String,
    value: String,
    icon: String,
    modifier: Modifier = Modifier,
    accent: Color = MaterialTheme.colorScheme.primary,
    trend: String? = null,
    trendIsWarning: Boolean = false,
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(32.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Box(
                modifier = Modifier
                    .size(34.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(accent.copy(alpha = 0.12f)),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = icon,
                    style = MaterialTheme.typography.labelSmall,
                    color = accent,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                )
            }

            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = label,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = value,
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                )
            }

            if (trend != null) {
                Text(
                    text = trend,
                    style = MaterialTheme.typography.labelSmall,
                    color = if (trendIsWarning) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.secondary,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

@Composable
private fun DiagnosisVolumeCard(
    points: List<AdminChartPoint>,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Diagnosis Volume",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "...",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontWeight = FontWeight.Bold,
            )
        }
        HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
        WeeklyBarChart(points = points)
    }
}

@Composable
private fun WeeklyBarChart(points: List<AdminChartPoint>) {
    val maxValue = max(1, points.maxOfOrNull { point -> point.value } ?: 1)
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(150.dp),
        ) {
            Column(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.SpaceBetween,
            ) {
                repeat(4) {
                    HorizontalDivider(
                        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.24f),
                    )
                }
            }
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.Bottom,
            ) {
                points.forEach { point ->
                    val fraction = if (point.value == 0) {
                        0.12f
                    } else {
                        (point.value.toFloat() / maxValue).coerceIn(0.18f, 1f)
                    }
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxHeight(fraction)
                            .clip(RoundedCornerShape(topStart = 8.dp, topEnd = 8.dp))
                            .background(
                                if (point.highlighted) {
                                    MaterialTheme.colorScheme.secondary
                                } else {
                                    MaterialTheme.colorScheme.primaryContainer
                                }
                            ),
                    )
                }
            }
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            points.forEach { point ->
                Text(
                    text = point.label,
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.labelSmall,
                    color = if (point.highlighted) {
                        MaterialTheme.colorScheme.secondary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    fontWeight = if (point.highlighted) FontWeight.Bold else FontWeight.Medium,
                )
            }
        }
    }
}

@Composable
private fun RecentActivityCard(
    activities: List<AdminActivityItem>,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(32.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(CVioSurfaceContainerLow)
                    .padding(horizontal = 20.dp, vertical = 14.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Recent System Activity",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "View All",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                )
            }
            activities.forEachIndexed { index, activity ->
                ActivityRow(activity = activity)
                if (index != activities.lastIndex) {
                    HorizontalDivider(
                        modifier = Modifier.padding(start = 72.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f),
                    )
                }
            }
        }
    }
}

@Composable
private fun ActivityRow(activity: AdminActivityItem) {
    val accent = activityColor(activity.kind)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(18.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(accent.copy(alpha = 0.16f)),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = activityIcon(activity.kind),
                style = MaterialTheme.typography.labelMedium,
                color = accent,
                fontWeight = FontWeight.Bold,
            )
        }
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(
                text = activity.title,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = activity.subtitle,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        Text(
            text = activity.relativeTime,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.outline,
            maxLines = 1,
        )
    }
}

@Composable
private fun UserSummaryCard(
    user: AdminUserSummary,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(CVioSurfaceContainerLow),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = initialsFor(user.name),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = user.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = "${user.diseaseCheckCount} checks | Last active ${user.lastActive}",
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = user.farmLocation,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline,
                )
            }
            CVioStatusChip(
                text = user.status,
                containerColor = if (user.status == "Active") {
                    MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f)
                } else {
                    MaterialTheme.colorScheme.surfaceVariant
                },
                contentColor = if (user.status == "Active") HealthyGreen else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun DataReviewCard(
    item: AdminDataReviewItem,
    onMarkReviewed: (AdminDataReviewItem) -> Unit,
    onExcludeFromTraining: (AdminDataReviewItem) -> Unit,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = item.pond,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = item.label,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = "${item.farmerName} | ${String.format(Locale.US, "%.1f%%", item.confidence * 100f)} confidence",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Column(
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                CVioStatusChip(
                    text = item.permissionStatus,
                    containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f),
                    contentColor = HealthyGreen,
                )
                if (item.excluded) {
                    CVioStatusChip(
                        text = "Excluded",
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                        contentColor = DiseaseRed,
                    )
                } else if (item.reviewed) {
                    CVioStatusChip(
                        text = "Reviewed",
                        containerColor = CVioSurfaceContainerLow,
                        contentColor = MaterialTheme.colorScheme.primary,
                    )
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            SecondaryActionButton(
                text = if (item.reviewed) "Reviewed" else "Mark reviewed",
                onClick = { onMarkReviewed(item) },
                enabled = !item.reviewed,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = if (item.excluded) "Excluded" else "Exclude",
                onClick = { onExcludeFromTraining(item) },
                enabled = !item.excluded,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@Composable
private fun activityColor(kind: AdminActivityKind): Color {
    return when (kind) {
        AdminActivityKind.Alert -> MaterialTheme.colorScheme.error
        AdminActivityKind.Diagnosis -> MaterialTheme.colorScheme.primary
        AdminActivityKind.User -> MaterialTheme.colorScheme.secondary
        AdminActivityKind.Sync -> MaterialTheme.colorScheme.secondary
    }
}

private fun activityIcon(kind: AdminActivityKind): String {
    return when (kind) {
        AdminActivityKind.Alert -> "!"
        AdminActivityKind.Diagnosis -> "AI"
        AdminActivityKind.User -> "+"
        AdminActivityKind.Sync -> "OK"
    }
}

private fun compactNumber(value: Int): String {
    return when {
        value >= 1_000_000 -> compactNumber(value / 1_000_000f, "m")
        value >= 1_000 -> compactNumber(value / 1_000f, "k")
        else -> value.toString()
    }
}

private fun compactNumber(value: Float, suffix: String): String {
    val formatted = String.format(Locale.US, "%.1f", value).removeSuffix(".0")
    return "$formatted$suffix"
}

private fun initialsFor(name: String): String {
    return name
        .split(" ")
        .filter { part -> part.isNotBlank() }
        .take(2)
        .joinToString("") { part -> part.take(1).uppercase() }
        .ifBlank { "A" }
}
