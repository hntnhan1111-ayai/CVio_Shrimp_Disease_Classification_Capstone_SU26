package rs.smobile.shrimpdisease.ui.home

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
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.classifier.ModelInfo
import rs.smobile.shrimpdisease.data.BenchmarkMetrics
import rs.smobile.shrimpdisease.data.PredictionLogItem
import rs.smobile.shrimpdisease.ui.components.CVioCard
import rs.smobile.shrimpdisease.ui.components.CVioIconBubble
import rs.smobile.shrimpdisease.ui.components.CVioMetricTile
import rs.smobile.shrimpdisease.ui.components.CVioSectionHeader
import rs.smobile.shrimpdisease.ui.components.CVioStatusChip
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainer
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest
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
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        FarmerHeader()

        CVioCard {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CVioIconBubble(
                    label = "AI",
                    containerColor = MaterialTheme.colorScheme.secondaryContainer,
                    contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
                )
                CVioSectionHeader(
                    title = "Start shrimp diagnosis",
                    subtitle = "Capture or upload a shrimp image for offline LiteRT analysis.",
                    modifier = Modifier.weight(1f),
                )
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(156.dp)
                    .clip(MaterialTheme.shapes.large)
                    .background(CVioSurfaceContainer),
            ) {
                Box(
                    modifier = Modifier
                        .align(Alignment.Center)
                        .size(112.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(CVioSurfaceContainerLowest),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "CVio",
                        style = MaterialTheme.typography.headlineMedium,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Bold,
                    )
                }
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(18.dp)
                        .size(42.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(MaterialTheme.colorScheme.primaryContainer),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "Scan",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Button(
                    onClick = onOpenCamera,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Take Photo")
                }
                OutlinedButton(
                    onClick = onSelectImage,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Gallery")
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            CVioMetricTile(
                label = "AI model",
                value = "Ready",
                modifier = Modifier.weight(1f),
                accent = MaterialTheme.colorScheme.tertiary,
            )
            CVioMetricTile(
                label = "Total scans",
                value = benchmarkMetrics.totalRuns.toString(),
                modifier = Modifier.weight(1f),
            )
        }

        CVioCard(containerColor = CVioSurfaceContainerLowest) {
            CVioSectionHeader(
                title = "Current model",
                subtitle = "On-device detection with packaged TFLite assets.",
            )
            Text(
                text = modelInfo.modelFile,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Input ${BenchmarkUtils.inputSizeText(modelInfo)} | ${modelInfo.labelsCount} classes",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            OutlinedButton(
                onClick = onChooseModel,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(text = "Choose model")
            }
        }

        RecentCheckups(logs = logs.take(2))
    }
}

@Composable
private fun FarmerHeader() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(bottomStart = 32.dp, bottomEnd = 32.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
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
}

@Composable
private fun RecentCheckups(logs: List<PredictionLogItem>) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        CVioSectionHeader(
            title = "Recent checkups",
            subtitle = "Latest saved inference results.",
        )
        if (logs.isEmpty()) {
            CVioCard(containerColor = CVioSurfaceContainerLow) {
                Text(
                    text = "No saved checkups yet",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = "Run inference and save a result to start tracking pond history.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            logs.forEach { item ->
                CVioCard {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        CVioIconBubble(label = item.predictedClass.take(1).uppercase())
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            CVioStatusChip(
                                text = if (item.isAboveThreshold) item.predictedClass else "Low confidence",
                                containerColor = if (item.isAboveThreshold) {
                                    MaterialTheme.colorScheme.secondaryContainer
                                } else {
                                    MaterialTheme.colorScheme.errorContainer
                                },
                                contentColor = if (item.isAboveThreshold) {
                                    MaterialTheme.colorScheme.onSecondaryContainer
                                } else {
                                    MaterialTheme.colorScheme.onErrorContainer
                                },
                            )
                            Text(
                                text = BenchmarkUtils.confidenceText(item.confidence),
                                style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(
                                text = DateTimeUtils.formatTimestamp(item.timestamp),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }
    }
}
