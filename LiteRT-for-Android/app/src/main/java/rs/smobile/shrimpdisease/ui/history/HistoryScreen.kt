package rs.smobile.shrimpdisease.ui.history

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.data.HistoryFilter
import rs.smobile.shrimpdisease.data.HistoryUiState
import rs.smobile.shrimpdisease.ui.components.AppBrandLogo
import rs.smobile.shrimpdisease.ui.components.BenchmarkSummaryCard
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.EmptyState
import rs.smobile.shrimpdisease.ui.components.PredictionLogItemCard
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest

@Composable
fun HistoryScreen(
    historyUiState: HistoryUiState,
    onFilterSelected: (HistoryFilter) -> Unit,
    onExportCsv: () -> Unit,
    onClearLogs: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val logs = historyUiState.logs
    val filteredLogs = historyUiState.filteredLogs

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            HistoryTopBar()
        }

        item {
            Column(
                modifier = Modifier.padding(horizontal = 20.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "Diagnosis History",
                    style = MaterialTheme.typography.displayLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Review past health assessments and track pond trends.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        item {
            FilterRow(
                selectedFilter = historyUiState.selectedFilter,
                historyUiState = historyUiState,
                onFilterSelected = onFilterSelected,
            )
        }

        item {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                SecondaryActionButton(
                    text = "Export CSV",
                    enabled = logs.isNotEmpty(),
                    onClick = onExportCsv,
                    modifier = Modifier.weight(1f),
                )
                SecondaryActionButton(
                    text = "Clear logs",
                    enabled = logs.isNotEmpty(),
                    onClick = onClearLogs,
                    modifier = Modifier.weight(1f),
                )
            }
        }

        if (logs.isEmpty()) {
            item {
                CVioCard(
                    modifier = Modifier.padding(horizontal = 20.dp),
                    containerColor = CVioSurfaceContainerLow,
                ) {
                    EmptyState(
                        title = "No diagnosis history yet",
                        message = "Start your first shrimp health check.",
                    )
                }
            }
        } else if (filteredLogs.isEmpty()) {
            item {
                CVioCard(
                    modifier = Modifier.padding(horizontal = 20.dp),
                    containerColor = CVioSurfaceContainerLow,
                ) {
                    EmptyState(
                        title = "No matching records",
                        message = "Try another filter or save a new diagnosis result.",
                    )
                }
            }
        } else {
            items(
                items = filteredLogs,
                key = { item -> item.id },
            ) { item ->
                PredictionLogItemCard(
                    item = item,
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }
        }

        if (logs.isNotEmpty()) {
            item {
                BenchmarkSummaryCard(
                    metrics = historyUiState.benchmarkMetrics,
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }
        }
    }
}

@Composable
private fun HistoryTopBar() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .padding(horizontal = 20.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = "F",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.outline,
                fontWeight = FontWeight.Bold,
            )
        }
        AppBrandLogo()
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(CVioSurfaceContainerLowest),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = "N",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

@Composable
private fun FilterRow(
    selectedFilter: HistoryFilter,
    historyUiState: HistoryUiState,
    onFilterSelected: (HistoryFilter) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        HistoryFilter.values().forEach { filter ->
            val count = historyUiState.filterCounts.countFor(filter)
            FilterPill(
                label = if (count > 0) "${filter.label} $count" else filter.label,
                selected = selectedFilter == filter,
                onClick = { onFilterSelected(filter) },
            )
        }
    }
}

@Composable
private fun FilterPill(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        modifier = Modifier.clickable(onClick = onClick),
        shape = RoundedCornerShape(999.dp),
        color = if (selected) {
            MaterialTheme.colorScheme.primaryContainer
        } else {
            CVioSurfaceContainerLowest
        },
        contentColor = if (selected) {
            MaterialTheme.colorScheme.onPrimaryContainer
        } else {
            MaterialTheme.colorScheme.onSurfaceVariant
        },
        shadowElevation = 2.dp,
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 12.dp),
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
        )
    }
}
