package rs.smobile.shrimpdisease.ui.admin

import androidx.compose.foundation.background
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
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
import rs.smobile.shrimpdisease.data.AdminDataMutationInput
import rs.smobile.shrimpdisease.data.AdminDashboardUiState
import rs.smobile.shrimpdisease.data.AdminDataReviewItem
import rs.smobile.shrimpdisease.data.AdminUpdateUserInput
import rs.smobile.shrimpdisease.data.AdminUserSummary
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.components.EmptyState
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.components.ShrimpIconButton
import rs.smobile.shrimpdisease.ui.components.ShrimpLineIcon
import rs.smobile.shrimpdisease.ui.components.ShrimpNavIcon
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.ui.theme.WarningOrange
import rs.smobile.shrimpdisease.utils.DiseaseTextUtils
import java.util.Locale
import kotlin.math.max

private val AdminDataLabelOptions = listOf("Healthy", "BG", "WSSV", "WSSV_BG")

@Composable
fun AdminDashboardScreen(
    uiState: AdminDashboardUiState,
    currentUserName: String?,
    onRefresh: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenUsers: () -> Unit,
    onOpenData: () -> Unit,
    onOpenInference: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var showAllActivities by rememberSaveable { mutableStateOf(false) }

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            AdminTopBar(
                currentUserName = currentUserName,
                navigationIcon = ShrimpNavIcon.Settings,
                navigationContentDescription = "Mở cài đặt quản trị",
                onNavigationClick = onOpenSettings,
                trailingIcon = ShrimpNavIcon.Refresh,
                trailingContentDescription = "Làm mới bảng điều khiển",
                onTrailingClick = onRefresh,
            )
        }

        item {
            Column(
                modifier = Modifier.padding(horizontal = 20.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = "Bảng điều khiển",
                    style = MaterialTheme.typography.displayLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Tổng quan hệ thống và số liệu vận hành",
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
                        label = "Người dùng",
                        value = compactNumber(uiState.totalFarmers),
                        icon = ShrimpNavIcon.Users,
                        trend = uiState.farmersTrend,
                        onClick = onOpenUsers,
                        modifier = Modifier.weight(1f),
                    )
                    DashboardStatCard(
                        label = "Lượt kiểm tra",
                        value = compactNumber(uiState.totalDiseaseChecks),
                        icon = ShrimpNavIcon.Inference,
                        accent = MaterialTheme.colorScheme.secondary,
                        trend = uiState.checksTrend,
                        onClick = onOpenData,
                        modifier = Modifier.weight(1f),
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    DashboardStatCard(
                        label = "Ảnh đã xử lý",
                        value = compactNumber(uiState.imagesProcessed),
                        icon = ShrimpNavIcon.Data,
                        accent = MaterialTheme.colorScheme.primary,
                        onClick = onOpenData,
                        modifier = Modifier.weight(1f),
                    )
                    DashboardStatCard(
                        label = "Cảnh báo",
                        value = compactNumber(uiState.activeAlerts),
                        icon = ShrimpNavIcon.Diagnose,
                        accent = if (uiState.activeAlerts > 0) WarningOrange else MaterialTheme.colorScheme.secondary,
                        trend = if (uiState.activeAlerts > 0) "Cần xử lý" else "Ổn định",
                        trendIsWarning = uiState.activeAlerts > 0,
                        onClick = onOpenInference,
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
                activities = if (showAllActivities) uiState.recentActivities else uiState.recentActivities.take(3),
                totalActivityCount = uiState.recentActivities.size,
                showAll = showAllActivities,
                onViewAllClick = { showAllActivities = !showAllActivities },
                modifier = Modifier.padding(horizontal = 20.dp),
            )
        }
    }
}

@Composable
fun AdminUsersScreen(
    users: List<AdminUserSummary>,
    currentUserName: String?,
    onCreateUser: (AdminCreateUserInput, (Boolean) -> Unit) -> Unit,
    onUpdateUser: (String, AdminUpdateUserInput, (Boolean) -> Unit) -> Unit,
    onDeleteUser: (AdminUserSummary, (Boolean) -> Unit) -> Unit,
    onHome: () -> Unit,
    onSettings: () -> Unit,
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
                AdminTopBar(
                    currentUserName = currentUserName,
                    navigationIcon = ShrimpNavIcon.Home,
                    navigationContentDescription = "Về bảng điều khiển",
                    onNavigationClick = onHome,
                    trailingIcon = ShrimpNavIcon.Settings,
                    trailingContentDescription = "Mở cài đặt",
                    onTrailingClick = onSettings,
                )
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
                                text = "Quản lý người dùng",
                                style = MaterialTheme.typography.displayLarge,
                                color = MaterialTheme.colorScheme.onSurface,
                                fontWeight = FontWeight.Bold,
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
                            title = "Không tìm thấy người dùng",
                            message = "Thử tìm theo tên, tài khoản, ao nuôi hoặc trạng thái khác.",
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
                onCreateUser(input) { created ->
                    if (created) showCreateDialog = false
                }
            },
        )
    }

    selectedUser?.let { user ->
        UserDetailsDialog(
            user = user,
            onDismiss = { selectedUser = null },
            onUpdateUser = { input ->
                onUpdateUser(user.id, input) { updated ->
                    if (updated) selectedUser = null
                }
            },
            onDeleteUser = {
                onDeleteUser(user) { deleted ->
                    if (deleted) selectedUser = null
                }
            },
        )
    }
}

@Composable
fun AdminDataControlScreen(
    dataItems: List<AdminDataReviewItem>,
    currentUserName: String?,
    onCreateData: (AdminDataMutationInput, (Boolean) -> Unit) -> Unit,
    onUpdateData: (AdminDataReviewItem, AdminDataMutationInput, (Boolean) -> Unit) -> Unit,
    onDeleteData: (AdminDataReviewItem, (Boolean) -> Unit) -> Unit,
    onMarkReviewed: (AdminDataReviewItem) -> Unit,
    onExcludeFromTraining: (AdminDataReviewItem) -> Unit,
    onExportMetadata: () -> Unit,
    onHome: () -> Unit,
    onSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var showCreateDialog by rememberSaveable { mutableStateOf(false) }
    var editingItem by remember { mutableStateOf<AdminDataReviewItem?>(null) }
    var deletingItem by remember { mutableStateOf<AdminDataReviewItem?>(null) }

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            AdminTopBar(
                currentUserName = currentUserName,
                navigationIcon = ShrimpNavIcon.Home,
                navigationContentDescription = "Về bảng điều khiển",
                onNavigationClick = onHome,
                trailingIcon = ShrimpNavIcon.Settings,
                trailingContentDescription = "Mở cài đặt",
                onTrailingClick = onSettings,
            )
        }
        item {
            CVioSectionHeader(
                title = "Quản lý dữ liệu",
                modifier = Modifier.padding(horizontal = 20.dp),
            )
        }
        item {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                PrimaryActionButton(
                    text = "Thêm dữ liệu",
                    onClick = { showCreateDialog = true },
                    modifier = Modifier.weight(1f),
                )
                SecondaryActionButton(
                    text = "Xuất metadata",
                    onClick = onExportMetadata,
                    enabled = dataItems.isNotEmpty(),
                    modifier = Modifier.weight(1f),
                )
            }
        }

        if (dataItems.isEmpty()) {
            item {
                CVioCard(
                    modifier = Modifier.padding(horizontal = 20.dp),
                    containerColor = CVioSurfaceContainerLow,
                ) {
                    EmptyState(
                        title = "Chưa có dữ liệu",
                        message = "Dữ liệu được nông dân cho phép hoặc dữ liệu Admin thêm mới sẽ xuất hiện tại đây.",
                    )
                }
            }
        } else {
            items(items = dataItems, key = { item -> item.id }) { item ->
                DataReviewCard(
                    item = item,
                    onMarkReviewed = onMarkReviewed,
                    onExcludeFromTraining = onExcludeFromTraining,
                    onEdit = { editingItem = item },
                    onDelete = { deletingItem = item },
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }
        }
    }

    if (showCreateDialog) {
        DataMutationDialog(
            title = "Thêm dữ liệu",
            confirmText = "Tạo",
            onDismiss = { showCreateDialog = false },
            onSubmit = { input ->
                onCreateData(input) { created ->
                    if (created) showCreateDialog = false
                }
            },
        )
    }

    editingItem?.let { item ->
        DataMutationDialog(
            title = "Cập nhật dữ liệu",
            confirmText = "Lưu",
            item = item,
            onDismiss = { editingItem = null },
            onSubmit = { input ->
                onUpdateData(item, input) { updated ->
                    if (updated) editingItem = null
                }
            },
        )
    }

    deletingItem?.let { item ->
        ConfirmDeleteDialog(
            title = "Xóa dữ liệu",
            message = "Bạn có chắc muốn xóa dữ liệu của ${item.farmerName} tại ${item.pond}?",
            onDismiss = { deletingItem = null },
            onConfirm = {
                onDeleteData(item) { deleted ->
                    if (deleted) deletingItem = null
                }
            },
        )
    }
}

@Composable
private fun AdminTopBar(
    currentUserName: String?,
    modifier: Modifier = Modifier,
    navigationIcon: ShrimpNavIcon? = null,
    navigationContentDescription: String = "",
    onNavigationClick: (() -> Unit)? = null,
    trailingIcon: ShrimpNavIcon? = null,
    trailingContentDescription: String = "",
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
            if (navigationIcon != null && onNavigationClick != null) {
                ShrimpIconButton(
                    icon = navigationIcon,
                    contentDescription = navigationContentDescription,
                    onClick = onNavigationClick,
                    modifier = Modifier
                        .size(40.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(CVioSurfaceContainerLow),
                )
            } else {
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
            }
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = "CVio",
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                )
                if (!currentUserName.isNullOrBlank()) {
                    Text(
                        text = currentUserName,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
        }

        if (trailingIcon != null && onTrailingClick != null) {
            ShrimpIconButton(
                icon = trailingIcon,
                contentDescription = trailingContentDescription,
                onClick = onTrailingClick,
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(CVioSurfaceContainerLowest),
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
            ShrimpLineIcon(
                icon = ShrimpNavIcon.Search,
                modifier = Modifier.size(18.dp),
                color = MaterialTheme.colorScheme.outline,
            )
        },
        placeholder = {
            Text(
                text = "Tìm người dùng...",
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
                        text = displayUserStatus(user.status),
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
                        label = "Lượt kiểm tra",
                        value = user.diseaseCheckCount.toString(),
                        active = isActive,
                        modifier = Modifier.weight(1f),
                    )
                    UserMetricBlock(
                        label = "Hoạt động cuối",
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
                        text = "Xem chi tiết",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.secondary,
                        fontWeight = FontWeight.Bold,
                    )
                    ShrimpLineIcon(
                        icon = ShrimpNavIcon.ArrowRight,
                        modifier = Modifier.size(18.dp),
                        color = MaterialTheme.colorScheme.secondary,
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
            style = if (label == "Lượt kiểm tra") {
                MaterialTheme.typography.headlineSmall
            } else {
                MaterialTheme.typography.bodyMedium
            },
            color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = if (label == "Lượt kiểm tra") FontWeight.Bold else FontWeight.Medium,
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
                ShrimpLineIcon(
                    icon = ShrimpNavIcon.Users,
                    modifier = Modifier.size(30.dp),
                    color = MaterialTheme.colorScheme.primary,
                )
            }
            Text(
                text = "Thêm người dùng",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Tạo tài khoản nông dân hoặc kỹ thuật viên mới.",
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
            title = "Chưa có nông dân",
            message = "Tạo tài khoản đầu tiên cho trang quản trị.",
        )
        PrimaryActionButton(
            text = "Thêm người dùng",
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
                text = "Thêm người dùng",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = displayName,
                    onValueChange = { displayName = it },
                    label = { Text("Họ và tên") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = account,
                    onValueChange = {
                        account = it
                        if (email.isBlank() && it.contains("@")) email = it
                    },
                    label = { Text("Email hoặc số điện thoại đăng nhập") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("Mật khẩu tạm thời") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = farmLocation,
                    onValueChange = { farmLocation = it },
                    label = { Text("Vị trí ao nuôi") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedTextField(
                        value = phoneNumber,
                        onValueChange = { phoneNumber = it },
                        label = { Text("Số điện thoại") },
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
                    text = "Mật khẩu cần ít nhất 6 ký tự.",
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
                Text("Tạo")
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
private fun EditUserDialog(
    user: AdminUserSummary,
    onDismiss: () -> Unit,
    onUpdateUser: (AdminUpdateUserInput) -> Unit,
) {
    var displayName by rememberSaveable(user.id) { mutableStateOf(user.name) }
    var account by rememberSaveable(user.id) { mutableStateOf(user.account) }
    var farmLocation by rememberSaveable(user.id) { mutableStateOf(user.farmLocation) }
    var phoneNumber by rememberSaveable(user.id) { mutableStateOf(user.phoneNumber) }
    var email by rememberSaveable(user.id) { mutableStateOf(user.email) }
    var dataPermissionEnabled by rememberSaveable(user.id) { mutableStateOf(user.dataPermissionEnabled) }
    val canSubmit = displayName.isNotBlank() && account.isNotBlank()

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = "Cập nhật người dùng",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = displayName,
                    onValueChange = { displayName = it },
                    label = { Text("Họ và tên") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = account,
                    onValueChange = {
                        account = it
                        if (email.isBlank() && it.contains("@")) email = it
                    },
                    label = { Text("Email hoặc số điện thoại đăng nhập") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = farmLocation,
                    onValueChange = { farmLocation = it },
                    label = { Text("Vị trí ao nuôi") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedTextField(
                        value = phoneNumber,
                        onValueChange = { phoneNumber = it },
                        label = { Text("Số điện thoại") },
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
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "Cho phép dùng dữ liệu",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Switch(
                        checked = dataPermissionEnabled,
                        onCheckedChange = { dataPermissionEnabled = it },
                    )
                }
            }
        },
        confirmButton = {
            TextButton(
                enabled = canSubmit,
                onClick = {
                    onUpdateUser(
                        AdminUpdateUserInput(
                            displayName = displayName,
                            account = account,
                            farmLocation = farmLocation,
                            phoneNumber = phoneNumber,
                            email = email,
                            dataPermissionEnabled = dataPermissionEnabled,
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
private fun DataMutationDialog(
    title: String,
    confirmText: String,
    onDismiss: () -> Unit,
    onSubmit: (AdminDataMutationInput) -> Unit,
    item: AdminDataReviewItem? = null,
) {
    var farmerName by rememberSaveable(item?.id) { mutableStateOf(item?.farmerName.orEmpty()) }
    var pond by rememberSaveable(item?.id) { mutableStateOf(item?.pond.orEmpty()) }
    var label by rememberSaveable(item?.id) { mutableStateOf(item?.label.orEmpty()) }
    var permissionStatus by rememberSaveable(item?.id) {
        mutableStateOf(item?.permissionStatus ?: "Allowed")
    }
    var confidenceText by rememberSaveable(item?.id) {
        mutableStateOf(item?.confidence?.let { value -> String.format(Locale.US, "%.2f", value) } ?: "")
    }
    val labelOptions = remember(item?.id) {
        (AdminDataLabelOptions + item?.label.orEmpty())
            .filter { option -> option.isNotBlank() }
            .distinct()
    }
    val confidence = confidenceText.trim().toFloatOrNull()?.let { value ->
        if (value > 1f) value / 100f else value
    }?.coerceIn(0f, 1f) ?: 0f
    val canSubmit = farmerName.isNotBlank() && pond.isNotBlank() && label.isNotBlank()

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = title,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = farmerName,
                    onValueChange = { farmerName = it },
                    label = { Text("Tên nông dân") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = pond,
                    onValueChange = { pond = it },
                    label = { Text("Ao nuôi / khu vực") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = "Nhãn bệnh",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontWeight = FontWeight.Bold,
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        labelOptions.forEach { option ->
                            FilterChip(
                                selected = label == option,
                                onClick = { label = option },
                                label = {
                                    Text(
                                        text = DiseaseTextUtils.displayLabel(option),
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                },
                            )
                        }
                    }
                }
                OutlinedTextField(
                    value = confidenceText,
                    onValueChange = { confidenceText = it },
                    label = { Text("Độ tin cậy (0-1 hoặc %)") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "Cho phép dùng dữ liệu",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Switch(
                        checked = permissionStatus != "Disabled",
                        onCheckedChange = { checked ->
                            permissionStatus = if (checked) "Allowed" else "Disabled"
                        },
                    )
                }
            }
        },
        confirmButton = {
            TextButton(
                enabled = canSubmit,
                onClick = {
                    onSubmit(
                        AdminDataMutationInput(
                            farmerName = farmerName,
                            pond = pond,
                            label = label,
                            permissionStatus = permissionStatus,
                            confidence = confidence,
                        )
                    )
                },
            ) {
                Text(confirmText)
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
private fun ConfirmDeleteDialog(
    title: String,
    message: String,
    onDismiss: () -> Unit,
    onConfirm: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = title,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        },
        confirmButton = {
            TextButton(onClick = onConfirm) {
                Text("Xóa")
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
private fun UserDetailsDialog(
    user: AdminUserSummary,
    onDismiss: () -> Unit,
    onUpdateUser: (AdminUpdateUserInput) -> Unit,
    onDeleteUser: () -> Unit,
) {
    var showEditDialog by rememberSaveable { mutableStateOf(false) }
    var showDeleteDialog by rememberSaveable { mutableStateOf(false) }

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
                DetailLine(label = "Trạng thái", value = displayUserStatus(user.status))
                DetailLine(label = "Lượt kiểm tra", value = user.diseaseCheckCount.toString())
                DetailLine(label = "Hoạt động cuối", value = user.lastActive)
                DetailLine(label = "Vị trí ao nuôi", value = user.farmLocation)
                DetailLine(label = "Số điện thoại", value = user.phoneNumber.ifBlank { "Chưa cập nhật" })
                DetailLine(label = "Email", value = user.email.ifBlank { "Chưa cập nhật" })
                DetailLine(
                    label = "Quyền dữ liệu",
                    value = if (user.dataPermissionEnabled) "Được phép" else "Tắt",
                )
            }
        },
        confirmButton = {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                TextButton(onClick = { showDeleteDialog = true }) {
                    Text("Xóa")
                }
                TextButton(onClick = { showEditDialog = true }) {
                    Text("Sửa")
                }
                TextButton(onClick = onDismiss) {
                    Text("Đóng")
                }
            }
        },
    )

    if (showEditDialog) {
        EditUserDialog(
            user = user,
            onDismiss = { showEditDialog = false },
            onUpdateUser = onUpdateUser,
        )
    }

    if (showDeleteDialog) {
        ConfirmDeleteDialog(
            title = "Xóa người dùng",
            message = "Bạn có chắc muốn xóa ${user.name}? Lịch sử kiểm tra của người dùng này cũng sẽ bị xóa khỏi máy.",
            onDismiss = { showDeleteDialog = false },
            onConfirm = onDeleteUser,
        )
    }
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
    icon: ShrimpNavIcon,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    accent: Color = MaterialTheme.colorScheme.primary,
    trend: String? = null,
    trendIsWarning: Boolean = false,
) {
    Card(
        modifier = modifier.clickable(onClick = onClick),
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
                ShrimpLineIcon(
                    icon = icon,
                    modifier = Modifier.size(20.dp),
                    color = accent,
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
                text = "Lượt kiểm tra theo ngày",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
            )
            ShrimpLineIcon(
                icon = ShrimpNavIcon.MoreHorizontal,
                modifier = Modifier.size(24.dp),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
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
    totalActivityCount: Int,
    showAll: Boolean,
    onViewAllClick: () -> Unit,
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
                    text = "Hoạt động hệ thống gần đây",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = if (showAll) "Thu gọn" else "Xem tất cả ($totalActivityCount)",
                    modifier = Modifier.clickable(onClick = onViewAllClick),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                )
            }
            if (activities.isEmpty()) {
                EmptyState(
                    title = "Chưa có hoạt động",
                    message = "Các lượt kiểm tra và thay đổi người dùng sẽ hiển thị tại đây.",
                )
            } else {
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
            ShrimpLineIcon(
                icon = activityIcon(activity.kind),
                modifier = Modifier.size(20.dp),
                color = accent,
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
private fun DataReviewCard(
    item: AdminDataReviewItem,
    onMarkReviewed: (AdminDataReviewItem) -> Unit,
    onExcludeFromTraining: (AdminDataReviewItem) -> Unit,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
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
                    text = "${item.farmerName} | Độ tin cậy ${String.format(Locale.US, "%.1f%%", item.confidence * 100f)}",
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
                    text = displayPermissionStatus(item.permissionStatus),
                    containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f),
                    contentColor = HealthyGreen,
                )
                if (item.excluded) {
                    CVioStatusChip(
                        text = "Đã loại",
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                        contentColor = DiseaseRed,
                    )
                } else if (item.reviewed) {
                    CVioStatusChip(
                        text = "Đã duyệt",
                        containerColor = CVioSurfaceContainerLow,
                        contentColor = MaterialTheme.colorScheme.primary,
                    )
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            SecondaryActionButton(
                text = if (item.reviewed) "Đã duyệt" else "Đánh dấu duyệt",
                onClick = { onMarkReviewed(item) },
                enabled = !item.reviewed,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = if (item.excluded) "Đã loại" else "Loại khỏi train",
                onClick = { onExcludeFromTraining(item) },
                enabled = !item.excluded,
                modifier = Modifier.weight(1f),
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            SecondaryActionButton(
                text = "Sửa",
                onClick = onEdit,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = "Xóa",
                onClick = onDelete,
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

private fun activityIcon(kind: AdminActivityKind): ShrimpNavIcon {
    return when (kind) {
        AdminActivityKind.Alert -> ShrimpNavIcon.Diagnose
        AdminActivityKind.Diagnosis -> ShrimpNavIcon.Inference
        AdminActivityKind.User -> ShrimpNavIcon.Users
        AdminActivityKind.Sync -> ShrimpNavIcon.Data
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

private fun displayUserStatus(status: String): String {
    return when (status) {
        "Active" -> "Đang hoạt động"
        "Inactive" -> "Tạm ngưng"
        else -> status
    }
}

private fun displayPermissionStatus(status: String): String {
    return when (status) {
        "Allowed" -> "Được phép"
        "Disabled" -> "Tắt"
        else -> status
    }
}

private fun initialsFor(name: String): String {
    return name
        .split(" ")
        .filter { part -> part.isNotBlank() }
        .take(2)
        .joinToString("") { part -> part.take(1).uppercase() }
        .ifBlank { "A" }
}
