package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.ui.theme.AquaGreen
import rs.smobile.shrimpdisease.ui.theme.CVioPrimaryFixed
import rs.smobile.shrimpdisease.ui.theme.CVioSecondaryFixed
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.DarkNavy
import rs.smobile.shrimpdisease.ui.theme.DeepTeal
import rs.smobile.shrimpdisease.ui.theme.DiseaseRed
import rs.smobile.shrimpdisease.ui.theme.HealthyGreen
import rs.smobile.shrimpdisease.ui.theme.LightCyan
import rs.smobile.shrimpdisease.ui.theme.SoftCream
import rs.smobile.shrimpdisease.ui.theme.WarningOrange

@Composable
fun PrimaryActionButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.height(48.dp),
        shape = RoundedCornerShape(24.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.primary,
            contentColor = MaterialTheme.colorScheme.onPrimary,
        ),
        elevation = ButtonDefaults.buttonElevation(defaultElevation = 3.dp),
        contentPadding = PaddingValues(horizontal = 20.dp),
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
        )
    }
}

@Composable
fun SecondaryActionButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.height(48.dp),
        shape = RoundedCornerShape(24.dp),
        colors = ButtonDefaults.outlinedButtonColors(
            contentColor = MaterialTheme.colorScheme.primary,
            disabledContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
        ),
        border = ButtonDefaults.outlinedButtonBorder(enabled = enabled),
        contentPadding = PaddingValues(horizontal = 18.dp),
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
fun SettingsSectionCard(
    title: String,
    subtitle: String? = null,
    modifier: Modifier = Modifier,
    containerColor: Color = MaterialTheme.colorScheme.surface,
    content: @Composable ColumnScope.() -> Unit,
) {
    CVioCard(
        modifier = modifier,
        containerColor = containerColor,
        tonalElevation = 2.dp,
    ) {
        CVioSectionHeader(title = title, subtitle = subtitle)
        content()
    }
}

@Composable
fun DiagnosisHeroCard(
    title: String,
    subtitle: String,
    primaryAction: String,
    secondaryAction: String,
    onPrimaryAction: () -> Unit,
    onSecondaryAction: () -> Unit,
    modifier: Modifier = Modifier,
) {
    CVioCard(
        modifier = modifier,
        containerColor = MaterialTheme.colorScheme.surface,
        tonalElevation = 4.dp,
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CVioIconBubble(
                label = "AI",
                containerColor = CVioSecondaryFixed,
                contentColor = DeepTeal,
                size = 44.dp,
            )
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(148.dp)
                .clip(RoundedCornerShape(24.dp))
                .background(
                    Brush.linearGradient(
                        colors = listOf(SoftCream, LightCyan, CVioPrimaryFixed.copy(alpha = 0.65f)),
                    )
                ),
            contentAlignment = Alignment.Center,
        ) {
            ShrimpIllustration(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(120.dp)
                    .padding(horizontal = 22.dp),
            )
            Surface(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(14.dp),
                shape = RoundedCornerShape(999.dp),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.86f),
                contentColor = MaterialTheme.colorScheme.primary,
            ) {
                Text(
                    text = "On-device",
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                )
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            PrimaryActionButton(
                text = primaryAction,
                onClick = onPrimaryAction,
                modifier = Modifier.fillMaxWidth(),
            )
            SecondaryActionButton(
                text = secondaryAction,
                onClick = onSecondaryAction,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
fun ShrimpIllustration(
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val bodyBrush = Brush.linearGradient(
            colors = listOf(Color(0xFFFFEEE5), Color(0xFFE6A98E), Color(0xFFB5674C)),
            start = Offset(w * 0.2f, h * 0.1f),
            end = Offset(w * 0.85f, h * 0.7f),
        )
        drawOval(
            brush = bodyBrush,
            topLeft = Offset(w * 0.28f, h * 0.25f),
            size = Size(w * 0.42f, h * 0.42f),
        )
        drawOval(
            color = Color(0xFFFFF6F0),
            topLeft = Offset(w * 0.58f, h * 0.18f),
            size = Size(w * 0.16f, h * 0.18f),
        )
        repeat(5) { index ->
            val x = w * (0.36f + index * 0.07f)
            drawLine(
                color = Color(0xFF8C4A35).copy(alpha = 0.55f),
                start = Offset(x, h * 0.26f),
                end = Offset(x + w * 0.04f, h * 0.64f),
                strokeWidth = 3.dp.toPx(),
                cap = StrokeCap.Round,
            )
        }
        drawArc(
            color = Color(0xFF8C4A35),
            startAngle = 115f,
            sweepAngle = 190f,
            useCenter = false,
            topLeft = Offset(w * 0.16f, h * 0.14f),
            size = Size(w * 0.44f, h * 0.68f),
            style = Stroke(width = 4.dp.toPx(), cap = StrokeCap.Round),
        )
        drawLine(
            color = Color(0xFF8C4A35),
            start = Offset(w * 0.71f, h * 0.28f),
            end = Offset(w * 0.94f, h * 0.08f),
            strokeWidth = 2.dp.toPx(),
            cap = StrokeCap.Round,
        )
        drawLine(
            color = Color(0xFF8C4A35),
            start = Offset(w * 0.7f, h * 0.32f),
            end = Offset(w * 0.96f, h * 0.28f),
            strokeWidth = 2.dp.toPx(),
            cap = StrokeCap.Round,
        )
        repeat(4) { index ->
            val x = w * (0.36f + index * 0.08f)
            drawLine(
                color = Color(0xFF8C4A35).copy(alpha = 0.7f),
                start = Offset(x, h * 0.62f),
                end = Offset(x - w * 0.03f, h * 0.84f),
                strokeWidth = 2.dp.toPx(),
                cap = StrokeCap.Round,
            )
        }
        drawCircle(
            color = DarkNavy,
            radius = 3.dp.toPx(),
            center = Offset(w * 0.68f, h * 0.28f),
        )
    }
}

@Composable
fun CameraScanFrame(
    modifier: Modifier = Modifier,
    label: String = "Position shrimp within the frame for best results",
    content: (@Composable BoxScope.() -> Unit)? = null,
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(28.dp))
            .background(DarkNavy)
            .padding(14.dp)
            .drawBehind {
                val strokeWidth = 1.4.dp.toPx()
                val dash = PathEffect.dashPathEffect(floatArrayOf(12.dp.toPx(), 8.dp.toPx()))
                drawRoundRect(
                    color = CVioPrimaryFixed.copy(alpha = 0.65f),
                    topLeft = Offset(strokeWidth, strokeWidth),
                    size = Size(size.width - strokeWidth * 2f, size.height - strokeWidth * 2f),
                    cornerRadius = CornerRadius(24.dp.toPx(), 24.dp.toPx()),
                    style = Stroke(width = strokeWidth, pathEffect = dash),
                )
            },
        contentAlignment = Alignment.Center,
    ) {
        if (content != null) {
            content()
        } else {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(RoundedCornerShape(18.dp))
                        .border(2.dp, CVioPrimaryFixed.copy(alpha = 0.65f), RoundedCornerShape(18.dp)),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "AI",
                        style = MaterialTheme.typography.titleMedium,
                        color = CVioPrimaryFixed,
                        fontWeight = FontWeight.Bold,
                    )
                }
                Text(
                    text = label,
                    modifier = Modifier.padding(horizontal = 26.dp),
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}

@Composable
fun ResultStatusBadge(
    text: String,
    modifier: Modifier = Modifier,
    kind: ResultStatusKind = ResultStatusKind.Neutral,
) {
    val colors = when (kind) {
        ResultStatusKind.Healthy -> HealthyGreen.copy(alpha = 0.12f) to HealthyGreen
        ResultStatusKind.Disease -> DiseaseRed.copy(alpha = 0.12f) to DiseaseRed
        ResultStatusKind.Warning -> WarningOrange.copy(alpha = 0.18f) to WarningOrange
        ResultStatusKind.Neutral -> CVioSurfaceContainerLow to MaterialTheme.colorScheme.onSurfaceVariant
    }
    CVioStatusChip(
        text = text,
        modifier = modifier,
        containerColor = colors.first,
        contentColor = colors.second,
    )
}

enum class ResultStatusKind {
    Healthy,
    Disease,
    Warning,
    Neutral,
}

@Composable
fun ConfidenceScoreCard(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    accent: Color = MaterialTheme.colorScheme.primary,
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(22.dp))
            .background(LightCyan.copy(alpha = 0.75f))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.headlineMedium,
            color = accent,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
fun PermissionCard(
    title: String,
    message: String,
    actionLabel: String,
    onAction: () -> Unit,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier, containerColor = MaterialTheme.colorScheme.surface) {
        EmptyState(title = title, message = message, modifier = Modifier.padding(0.dp))
        PrimaryActionButton(
            text = actionLabel,
            onClick = onAction,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
fun DataPermissionToggle(
    title: String,
    message: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
fun AdminStatCard(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    accent: Color = MaterialTheme.colorScheme.primary,
) {
    CVioCard(
        modifier = modifier,
        containerColor = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.headlineMedium,
            color = accent,
            fontWeight = FontWeight.Bold,
        )
    }
}

@Composable
fun SoftChartPlaceholder(
    modifier: Modifier = Modifier,
    barCount: Int = 7,
    color: Color = AquaGreen,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(118.dp)
            .clip(RoundedCornerShape(24.dp))
            .background(LightCyan.copy(alpha = 0.65f))
            .padding(18.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.Bottom,
    ) {
        repeat(barCount) { index ->
            val height = listOf(0.34f, 0.62f, 0.44f, 0.84f, 0.58f, 0.72f, 0.48f)[index % 7]
            Spacer(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxSize(height)
                    .clip(RoundedCornerShape(topStart = 999.dp, topEnd = 999.dp))
                    .background(color.copy(alpha = 0.38f + index * 0.05f)),
            )
        }
    }
}

@Composable
fun CompactInfoRow(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(0.85f),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            modifier = Modifier
                .weight(1.15f)
                .padding(start = 12.dp),
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.End,
        )
    }
}

@Composable
fun TwoColumnActions(
    modifier: Modifier = Modifier,
    content: @Composable RowScope.() -> Unit,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
        content = content,
    )
}
