package rs.smobile.shrimpdisease.ui.home

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioIconBubble
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.components.ShrimpIllustration
import rs.smobile.shrimpdisease.ui.components.SoftChartPlaceholder
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainer
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import rs.smobile.shrimpdisease.utils.DateTimeUtils

@Composable
fun HomeScreen(
    modelInfo: ModelInfo,
    benchmarkMetrics: BenchmarkMetrics,
    logs: List<PredictionLogItem>,
    onSelectImage: () -> Unit,
    onOpenCamera: () -> Unit,
    onChooseModel: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
    ) {
        FarmerHeader()

        Column(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            DiagnosisCard(
                onOpenCamera = onOpenCamera,
                onSelectImage = onSelectImage,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                ModelReadyCard(
                    modelInfo = modelInfo,
                    onChooseModel = onChooseModel,
                    modifier = Modifier.weight(1f),
                )
                WeeklyScansCard(
                    totalRuns = benchmarkMetrics.weeklyRuns,
                    modifier = Modifier.weight(1f),
                )
            }

            RecentCheckups(logs = logs.take(2))
        }
    }
}

@Composable
private fun FarmerHeader() {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(bottomStart = 32.dp, bottomEnd = 32.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
    ) {
        Box(modifier = Modifier.fillMaxWidth()) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(top = 12.dp)
                    .size(156.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.05f)),
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 32.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = "Hello, Farmer",
                        style = MaterialTheme.typography.headlineMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "Check shrimp health today",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                CVioIconBubble(
                    label = "F",
                    size = 48.dp,
                    containerColor = CVioSurfaceContainerLowest,
                    contentColor = MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}

@Composable
private fun DiagnosisCard(
    onOpenCamera: () -> Unit,
    onSelectImage: () -> Unit,
) {
    CVioCard(containerColor = CVioSurfaceContainerLowest, tonalElevation = 4.dp) {
        Box(modifier = Modifier.fillMaxWidth()) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .size(180.dp)
                    .clip(RoundedCornerShape(topStart = 999.dp))
                    .background(CVioSurfaceContainerLow.copy(alpha = 0.62f)),
            )
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CVioIconBubble(
                        label = "AI",
                        size = 44.dp,
                        containerColor = MaterialTheme.colorScheme.secondaryContainer,
                        contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                    Text(
                        text = "Start shrimp diagnosis",
                        style = MaterialTheme.typography.headlineSmall,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.Bold,
                    )
                }

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(160.dp)
                        .clip(RoundedCornerShape(24.dp))
                        .background(CVioSurfaceContainer),
                    contentAlignment = Alignment.Center,
                ) {
                    ShrimpIllustration(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 18.dp, vertical = 10.dp),
                    )
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    PrimaryActionButton(
                        text = "Take Photo",
                        onClick = onOpenCamera,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    SecondaryActionButton(
                        text = "Choose from Gallery",
                        onClick = onSelectImage,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelReadyCard(
    modelInfo: ModelInfo,
    onChooseModel: () -> Unit,
    modifier: Modifier = Modifier,
) {
    CVioCard(
        modifier = modifier,
        containerColor = CVioSurfaceContainerLowest,
        tonalElevation = 3.dp,
    ) {
        CVioIconBubble(
            label = "M",
            size = 48.dp,
            containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.22f),
            contentColor = MaterialTheme.colorScheme.tertiary,
        )
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = "AI model ready",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = "On-device detection",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "${modelInfo.labelsCount} classes",
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }
        SecondaryActionButton(
            text = "Choose model",
            onClick = onChooseModel,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun WeeklyScansCard(
    totalRuns: Int,
    modifier: Modifier = Modifier,
) {
    CVioCard(
        modifier = modifier,
        containerColor = CVioSurfaceContainerLowest,
        tonalElevation = 3.dp,
    ) {
        Text(
            text = "Weekly Scans",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = totalRuns.toString(),
            style = MaterialTheme.typography.displayLarge,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
        )
        SoftChartPlaceholder(
            modifier = Modifier.height(72.dp),
            color = MaterialTheme.colorScheme.primary,
        )
    }
}

@Composable
private fun RecentCheckups(logs: List<PredictionLogItem>) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Bottom,
        ) {
            Text(
                text = "Recent checkups",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "View all",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }

        if (logs.isEmpty()) {
            CVioCard(containerColor = CVioSurfaceContainerLow) {
                Text(
                    text = "No saved checkups yet",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Take a photo or choose from gallery to start your first shrimp health check.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            logs.forEach { item ->
                RecentCheckupCard(item = item)
            }
        }
    }
}

@Composable
private fun RecentCheckupCard(item: PredictionLogItem) {
    CVioCard(containerColor = MaterialTheme.colorScheme.surface, tonalElevation = 3.dp) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CheckupThumbnail(item = item)
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.Top,
                ) {
                    CVioStatusChip(
                        text = if (item.isAboveThreshold) item.predictedClass else "Warning",
                        containerColor = if (item.isAboveThreshold) {
                            MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f)
                        } else {
                            MaterialTheme.colorScheme.errorContainer
                        },
                        contentColor = if (item.isAboveThreshold) {
                            HealthyGreen
                        } else {
                            MaterialTheme.colorScheme.onErrorContainer
                        },
                    )
                    Text(
                        text = DateTimeUtils.formatTimestamp(item.timestamp),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    text = item.predictedClass,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "${BenchmarkUtils.confidenceText(item.confidence)} Confidence",
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (item.isAboveThreshold) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.error
                    },
                )
            }
        }
    }
}

@Composable
private fun CheckupThumbnail(item: PredictionLogItem) {
    val modifier = Modifier
        .size(64.dp)
        .clip(RoundedCornerShape(16.dp))
        .background(CVioSurfaceContainer)
    when {
        item.thumbnail != null -> {
            Image(
                bitmap = item.thumbnail.asImageBitmap(),
                contentDescription = "Checkup thumbnail",
                modifier = modifier,
                contentScale = ContentScale.Crop,
            )
        }

        item.imageUri != null -> {
            AsyncImage(
                model = item.imageUri,
                contentDescription = "Checkup thumbnail",
                modifier = modifier,
                contentScale = ContentScale.Crop,
            )
        }

        else -> {
            Box(modifier = modifier, contentAlignment = Alignment.Center) {
                ShrimpIllustration(modifier = Modifier.fillMaxSize().padding(8.dp))
            }
        }
    }
}
