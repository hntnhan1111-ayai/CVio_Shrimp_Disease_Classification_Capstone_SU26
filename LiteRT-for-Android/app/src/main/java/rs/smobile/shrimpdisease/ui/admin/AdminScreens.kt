package rs.smobile.shrimpdisease.ui.admin

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.ui.components.AdminStatCard
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.components.SoftChartPlaceholder
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen

data class AdminUserItem(
    val name: String,
    val status: String,
    val diseaseCheckCount: Int,
    val lastActive: String,
)

data class AdminDataItem(
    val pond: String,
    val label: String,
    val permissionStatus: String,
    val reviewed: Boolean,
)

@Composable
fun AdminDashboardScreen(
    totalFarmers: Int = 128,
    totalDiseaseChecks: Int = 1024,
    permittedImages: Int = 318,
    diseaseDetectedCount: Int = 86,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            CVioSectionHeader(
                title = "Admin Dashboard",
                subtitle = "Aquaculture diagnosis operations summary.",
            )
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                AdminStatCard(
                    label = "Total farmers",
                    value = totalFarmers.toString(),
                    modifier = Modifier.weight(1f),
                )
                AdminStatCard(
                    label = "Disease checks",
                    value = totalDiseaseChecks.toString(),
                    modifier = Modifier.weight(1f),
                    accent = MaterialTheme.colorScheme.secondary,
                )
            }
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                AdminStatCard(
                    label = "Allowed images",
                    value = permittedImages.toString(),
                    modifier = Modifier.weight(1f),
                    accent = HealthyGreen,
                )
                AdminStatCard(
                    label = "Detected",
                    value = diseaseDetectedCount.toString(),
                    modifier = Modifier.weight(1f),
                    accent = DiseaseRed,
                )
            }
        }
        item {
            CVioCard(containerColor = CVioSurfaceContainerLow) {
                CVioSectionHeader(title = "Disease Trend", subtitle = "Sample weekly activity")
                SoftChartPlaceholder(color = MaterialTheme.colorScheme.primary)
            }
        }
        item {
            CVioCard {
                CVioSectionHeader(title = "Recent Activity")
                listOf(
                    "Pond B submitted WSSV image review",
                    "Farmer Nguyen allowed dataset usage",
                    "Admin excluded low-quality image batch",
                ).forEach { activity ->
                    Text(
                        text = activity,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
fun AdminUsersScreen(
    users: List<AdminUserItem> = sampleUsers,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            CVioSectionHeader(
                title = "User Management",
                subtitle = "Farmers connected to shrimp diagnosis monitoring.",
            )
        }
        items(users) { user ->
            CVioCard {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Text(
                            text = user.name,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = "${user.diseaseCheckCount} checks | Last active ${user.lastActive}",
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
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
    }
}

@Composable
fun AdminDataControlScreen(
    dataItems: List<AdminDataItem> = sampleDataItems,
    onMarkReviewed: (AdminDataItem) -> Unit,
    onExcludeFromTraining: (AdminDataItem) -> Unit,
    onExportMetadata: () -> Unit,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            CVioSectionHeader(
                title = "Data Control",
                subtitle = "Review permitted image data before model improvement workflows.",
            )
        }
        item {
            PrimaryActionButton(
                text = "Export metadata",
                onClick = onExportMetadata,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        items(dataItems) { item ->
            CVioCard {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Text(
                            text = item.pond,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = item.label,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    CVioStatusChip(
                        text = item.permissionStatus,
                        containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f),
                        contentColor = HealthyGreen,
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    SecondaryActionButton(
                        text = if (item.reviewed) "Reviewed" else "Mark reviewed",
                        onClick = { onMarkReviewed(item) },
                        modifier = Modifier.weight(1f),
                    )
                    SecondaryActionButton(
                        text = "Exclude",
                        onClick = { onExcludeFromTraining(item) },
                        modifier = Modifier.weight(1f),
                    )
                }
            }
        }
    }
}

private val sampleUsers = listOf(
    AdminUserItem("Pond A Farmer", "Active", 42, "Today"),
    AdminUserItem("Pond B Intake", "Active", 35, "Yesterday"),
    AdminUserItem("Pond C Nursery", "Inactive", 11, "Last week"),
)

private val sampleDataItems = listOf(
    AdminDataItem("Pond A - Zone 1", "Healthy shrimp image set", "Allowed", reviewed = true),
    AdminDataItem("Pond B - Intake", "WSSV detected sample", "Allowed", reviewed = false),
    AdminDataItem("Pond C - Center", "Low confidence retake", "Allowed", reviewed = false),
)
