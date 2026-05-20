package rs.smobile.shrimpdisease.ui.home

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.components.SecondaryActionButton
import rs.smobile.shrimpdisease.ui.components.ShrimpIllustration
import rs.smobile.shrimpdisease.ui.components.ShrimpIconButton
import rs.smobile.shrimpdisease.ui.components.ShrimpLineIcon
import rs.smobile.shrimpdisease.ui.components.ShrimpNavIcon
import rs.smobile.shrimpdisease.ui.components.SoftChartPlaceholder
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainer
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.utils.BenchmarkUtils
import rs.smobile.shrimpdisease.utils.DateTimeUtils
import rs.smobile.shrimpdisease.utils.DiseaseTextUtils

@Composable
fun HomeScreen(
    modelInfo: ModelInfo,
    benchmarkMetrics: BenchmarkMetrics,
    logs: List<PredictionLogItem>,
    onSelectImage: () -> Unit,
    onOpenCamera: () -> Unit,
    onChooseModel: () -> Unit,
    onOpenProfile: () -> Unit,
    onViewAllHistory: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
    ) {
        FarmerHeader(onOpenProfile = onOpenProfile)

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

            RecentCheckups(
                logs = logs.take(2),
                onViewAllHistory = onViewAllHistory,
            )
        }
    }
}

@Composable
private fun FarmerHeader(onOpenProfile: () -> Unit) {
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
                        text = "Chào bà con",
                        style = MaterialTheme.typography.headlineMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "Kiểm tra sức khỏe tôm hôm nay ngay nào",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                ShrimpIconButton(
                    icon = ShrimpNavIcon.Profile,
                    contentDescription = "Mở hồ sơ",
                    onClick = onOpenProfile,
                    modifier = Modifier
                        .size(48.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(CVioSurfaceContainerLowest),
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
                    HomeIconBubble(
                        icon = ShrimpNavIcon.Diagnose,
                        size = 44.dp,
                        containerColor = MaterialTheme.colorScheme.secondaryContainer,
                        contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                    Text(
                        text = "Bắt đầu kiểm tra tôm",
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
                        text = "Chụp ảnh",
                        onClick = onOpenCamera,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    SecondaryActionButton(
                        text = "Chọn ảnh có sẵn",
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
        HomeIconBubble(
            icon = ShrimpNavIcon.Models,
            size = 48.dp,
            containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.22f),
            contentColor = MaterialTheme.colorScheme.tertiary,
        )
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = "AI đã sẵn sàng",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = "Kiểm tra ngay trên máy",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "${modelInfo.labelsCount} loại nhận biết",
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }
        SecondaryActionButton(
            text = "Chọn mô hình",
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
            text = "Lượt kiểm tra tuần này",
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
private fun RecentCheckups(
    logs: List<PredictionLogItem>,
    onViewAllHistory: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Bottom,
        ) {
            Text(
                text = "Lần kiểm tra gần đây",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Xem tất cả",
                modifier = Modifier.clickable(onClick = onViewAllHistory),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }

        if (logs.isEmpty()) {
            CVioCard(containerColor = CVioSurfaceContainerLow) {
                Text(
                    text = "Chưa có lần kiểm tra nào",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Chụp ảnh hoặc chọn ảnh có sẵn để bắt đầu kiểm tra sức khỏe tôm.",
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
    val displayLabel = DiseaseTextUtils.displayLabel(item.predictedClass)
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
                        text = if (item.isAboveThreshold) displayLabel else "Cần chụp lại",
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
                    text = displayLabel,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "${BenchmarkUtils.confidenceText(item.confidence)} độ tin cậy",
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
                contentDescription = "Ảnh lần kiểm tra",
                modifier = modifier,
                contentScale = ContentScale.Crop,
            )
        }

        item.imageUri != null -> {
            AsyncImage(
                model = item.imageUri,
                contentDescription = "Ảnh lần kiểm tra",
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

@Composable
private fun HomeIconBubble(
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
            modifier = Modifier.size(size * 0.54f),
            color = contentColor,
        )
    }
}
