package rs.smobile.shrimpdisease.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.size
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

data class ShrimpNavigationItem(
    val route: String,
    val label: String,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScaffold(
    title: String,
    currentRoute: String,
    navigationItems: List<ShrimpNavigationItem>,
    showTopBar: Boolean = true,
    showBottomBar: Boolean,
    topBarNavigationIcon: (@Composable () -> Unit)? = null,
    onNavigate: (ShrimpNavigationItem) -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable (PaddingValues) -> Unit,
) {
    Scaffold(
        modifier = modifier,
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            AnimatedVisibility(
                visible = showTopBar,
                enter = fadeIn(tween(180)) + expandVertically(tween(220)),
                exit = fadeOut(tween(120)) + shrinkVertically(tween(180)),
            ) {
                CenterAlignedTopAppBar(
                    title = {
                        Text(
                            text = title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                        )
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.background,
                        titleContentColor = MaterialTheme.colorScheme.primary,
                    ),
                    navigationIcon = {
                        topBarNavigationIcon?.invoke()
                    },
                )
            }
        },
        bottomBar = {
            AnimatedVisibility(
                visible = showBottomBar,
                enter = fadeIn(tween(180)) + slideInVertically(tween(220)) { height -> height / 2 },
                exit = fadeOut(tween(120)) + slideOutVertically(tween(180)) { height -> height / 2 },
            ) {
                ShrimpBottomNavigation(
                    items = navigationItems,
                    currentRoute = currentRoute,
                    onNavigate = onNavigate,
                )
            }
        },
        content = content,
    )
}

@Composable
fun ShrimpBottomNavigation(
    items: List<ShrimpNavigationItem>,
    currentRoute: String,
    onNavigate: (ShrimpNavigationItem) -> Unit,
    modifier: Modifier = Modifier,
) {
    NavigationBar(
        modifier = modifier,
        containerColor = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
    ) {
        items.forEach { item ->
            val selected = currentRoute == item.route
            NavigationBarItem(
                selected = selected,
                onClick = { onNavigate(item) },
                icon = {
                    ShrimpLineIcon(
                        icon = navIconForRoute(item.route),
                        modifier = Modifier.size(24.dp),
                        color = if (selected) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                    )
                },
                label = {
                    Text(
                        text = compactNavigationLabel(item),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = MaterialTheme.colorScheme.primary,
                    selectedTextColor = MaterialTheme.colorScheme.primary,
                    indicatorColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.72f),
                    unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
                ),
            )
        }
    }
}

enum class ShrimpNavIcon {
    Back,
    Home,
    Diagnose,
    History,
    Profile,
    Settings,
    Phone,
    Email,
    Camera,
    Search,
    ArrowRight,
    MoreHorizontal,
    Dashboard,
    Users,
    Data,
    Models,
    Inference,
    Refresh,
}

@Composable
fun ShrimpIconButton(
    icon: ShrimpNavIcon,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    IconButton(
        onClick = onClick,
        modifier = modifier.semantics {
            this.contentDescription = contentDescription
        },
    ) {
        ShrimpLineIcon(
            icon = icon,
            modifier = Modifier.size(24.dp),
            color = LocalContentColor.current,
        )
    }
}

@Composable
fun ShrimpLineIcon(
    icon: ShrimpNavIcon,
    modifier: Modifier = Modifier,
    color: Color = LocalContentColor.current,
) {
    Canvas(modifier = modifier) {
        val stroke = 2.2.dp.toPx()
        val thinStroke = 1.7.dp.toPx()
        val w = size.width
        val h = size.height

        when (icon) {
            ShrimpNavIcon.Back -> {
                drawLine(color, Offset(w * 0.64f, h * 0.18f), Offset(w * 0.34f, h * 0.5f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.34f, h * 0.5f), Offset(w * 0.64f, h * 0.82f), stroke, StrokeCap.Round)
            }

            ShrimpNavIcon.Home -> {
                val roof = Path().apply {
                    moveTo(w * 0.18f, h * 0.48f)
                    lineTo(w * 0.5f, h * 0.2f)
                    lineTo(w * 0.82f, h * 0.48f)
                }
                drawPath(roof, color, style = Stroke(stroke, cap = StrokeCap.Round))
                drawRoundRect(
                    color = color,
                    topLeft = Offset(w * 0.28f, h * 0.46f),
                    size = Size(w * 0.44f, h * 0.38f),
                    cornerRadius = CornerRadius(3.dp.toPx()),
                    style = Stroke(stroke),
                )
            }

            ShrimpNavIcon.Diagnose, ShrimpNavIcon.Inference -> {
                drawLine(color, Offset(w * 0.2f, h * 0.25f), Offset(w * 0.2f, h * 0.42f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.2f, h * 0.25f), Offset(w * 0.38f, h * 0.25f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.8f, h * 0.25f), Offset(w * 0.62f, h * 0.25f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.8f, h * 0.25f), Offset(w * 0.8f, h * 0.42f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.2f, h * 0.75f), Offset(w * 0.2f, h * 0.58f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.2f, h * 0.75f), Offset(w * 0.38f, h * 0.75f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.8f, h * 0.75f), Offset(w * 0.62f, h * 0.75f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.8f, h * 0.75f), Offset(w * 0.8f, h * 0.58f), stroke, StrokeCap.Round)
                drawCircle(color, radius = w * 0.12f, center = Offset(w * 0.5f, h * 0.5f), style = Stroke(thinStroke))
            }

            ShrimpNavIcon.History -> {
                drawArc(
                    color = color,
                    startAngle = 35f,
                    sweepAngle = 300f,
                    useCenter = false,
                    topLeft = Offset(w * 0.18f, h * 0.18f),
                    size = Size(w * 0.64f, h * 0.64f),
                    style = Stroke(stroke, cap = StrokeCap.Round),
                )
                drawLine(color, Offset(w * 0.5f, h * 0.5f), Offset(w * 0.5f, h * 0.34f), thinStroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.5f, h * 0.5f), Offset(w * 0.64f, h * 0.58f), thinStroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.18f, h * 0.2f), Offset(w * 0.18f, h * 0.38f), stroke, StrokeCap.Round)
            }

            ShrimpNavIcon.Profile -> {
                drawCircle(color, radius = w * 0.13f, center = Offset(w * 0.5f, h * 0.34f), style = Stroke(stroke))
                drawArc(
                    color = color,
                    startAngle = 205f,
                    sweepAngle = 130f,
                    useCenter = false,
                    topLeft = Offset(w * 0.24f, h * 0.48f),
                    size = Size(w * 0.52f, h * 0.42f),
                    style = Stroke(stroke, cap = StrokeCap.Round),
                )
            }

            ShrimpNavIcon.Settings -> {
                drawCircle(color, radius = w * 0.18f, center = Offset(w * 0.5f, h * 0.5f), style = Stroke(stroke))
                drawCircle(color, radius = w * 0.06f, center = Offset(w * 0.5f, h * 0.5f), style = Stroke(thinStroke))
                repeat(8) { index ->
                    val angle = Math.toRadians((index * 45).toDouble())
                    val start = Offset(
                        x = w * 0.5f + kotlin.math.cos(angle).toFloat() * w * 0.25f,
                        y = h * 0.5f + kotlin.math.sin(angle).toFloat() * h * 0.25f,
                    )
                    val end = Offset(
                        x = w * 0.5f + kotlin.math.cos(angle).toFloat() * w * 0.34f,
                        y = h * 0.5f + kotlin.math.sin(angle).toFloat() * h * 0.34f,
                    )
                    drawLine(color, start, end, thinStroke, StrokeCap.Round)
                }
            }

            ShrimpNavIcon.Phone -> {
                val handset = Path().apply {
                    moveTo(w * 0.34f, h * 0.22f)
                    cubicTo(w * 0.25f, h * 0.28f, w * 0.25f, h * 0.48f, w * 0.4f, h * 0.64f)
                    cubicTo(w * 0.55f, h * 0.78f, w * 0.72f, h * 0.78f, w * 0.78f, h * 0.68f)
                }
                drawPath(handset, color, style = Stroke(stroke, cap = StrokeCap.Round))
                drawLine(color, Offset(w * 0.33f, h * 0.22f), Offset(w * 0.44f, h * 0.34f), thinStroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.66f, h * 0.58f), Offset(w * 0.78f, h * 0.69f), thinStroke, StrokeCap.Round)
            }

            ShrimpNavIcon.Email -> {
                drawRoundRect(
                    color = color,
                    topLeft = Offset(w * 0.18f, h * 0.28f),
                    size = Size(w * 0.64f, h * 0.46f),
                    cornerRadius = CornerRadius(5.dp.toPx()),
                    style = Stroke(stroke),
                )
                drawLine(color, Offset(w * 0.2f, h * 0.32f), Offset(w * 0.5f, h * 0.55f), thinStroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.8f, h * 0.32f), Offset(w * 0.5f, h * 0.55f), thinStroke, StrokeCap.Round)
            }

            ShrimpNavIcon.Camera -> {
                drawRoundRect(
                    color = color,
                    topLeft = Offset(w * 0.2f, h * 0.34f),
                    size = Size(w * 0.6f, h * 0.42f),
                    cornerRadius = CornerRadius(5.dp.toPx()),
                    style = Stroke(stroke),
                )
                drawRoundRect(
                    color = color,
                    topLeft = Offset(w * 0.34f, h * 0.24f),
                    size = Size(w * 0.22f, h * 0.13f),
                    cornerRadius = CornerRadius(4.dp.toPx()),
                    style = Stroke(thinStroke),
                )
                drawCircle(color, radius = w * 0.12f, center = Offset(w * 0.5f, h * 0.55f), style = Stroke(thinStroke))
            }

            ShrimpNavIcon.Search -> {
                drawCircle(
                    color = color,
                    radius = w * 0.22f,
                    center = Offset(w * 0.43f, h * 0.43f),
                    style = Stroke(stroke),
                )
                drawLine(
                    color = color,
                    start = Offset(w * 0.6f, h * 0.6f),
                    end = Offset(w * 0.8f, h * 0.8f),
                    strokeWidth = stroke,
                    cap = StrokeCap.Round,
                )
            }

            ShrimpNavIcon.ArrowRight -> {
                drawLine(
                    color = color,
                    start = Offset(w * 0.22f, h * 0.5f),
                    end = Offset(w * 0.76f, h * 0.5f),
                    strokeWidth = stroke,
                    cap = StrokeCap.Round,
                )
                drawLine(
                    color = color,
                    start = Offset(w * 0.56f, h * 0.3f),
                    end = Offset(w * 0.76f, h * 0.5f),
                    strokeWidth = stroke,
                    cap = StrokeCap.Round,
                )
                drawLine(
                    color = color,
                    start = Offset(w * 0.56f, h * 0.7f),
                    end = Offset(w * 0.76f, h * 0.5f),
                    strokeWidth = stroke,
                    cap = StrokeCap.Round,
                )
            }

            ShrimpNavIcon.MoreHorizontal -> {
                drawCircle(color, radius = w * 0.055f, center = Offset(w * 0.28f, h * 0.5f))
                drawCircle(color, radius = w * 0.055f, center = Offset(w * 0.5f, h * 0.5f))
                drawCircle(color, radius = w * 0.055f, center = Offset(w * 0.72f, h * 0.5f))
            }

            ShrimpNavIcon.Refresh -> {
                drawArc(
                    color = color,
                    startAngle = 30f,
                    sweepAngle = 275f,
                    useCenter = false,
                    topLeft = Offset(w * 0.18f, h * 0.18f),
                    size = Size(w * 0.64f, h * 0.64f),
                    style = Stroke(stroke, cap = StrokeCap.Round),
                )
                val arrow = Path().apply {
                    moveTo(w * 0.77f, h * 0.24f)
                    lineTo(w * 0.78f, h * 0.43f)
                    lineTo(w * 0.6f, h * 0.36f)
                }
                drawPath(arrow, color, style = Stroke(stroke, cap = StrokeCap.Round))
            }

            ShrimpNavIcon.Dashboard -> {
                drawRoundRect(color, Offset(w * 0.18f, h * 0.18f), Size(w * 0.24f, h * 0.26f), CornerRadius(4.dp.toPx()), style = Stroke(stroke))
                drawRoundRect(color, Offset(w * 0.58f, h * 0.18f), Size(w * 0.24f, h * 0.18f), CornerRadius(4.dp.toPx()), style = Stroke(stroke))
                drawRoundRect(color, Offset(w * 0.18f, h * 0.58f), Size(w * 0.24f, h * 0.22f), CornerRadius(4.dp.toPx()), style = Stroke(stroke))
                drawRoundRect(color, Offset(w * 0.58f, h * 0.5f), Size(w * 0.24f, h * 0.3f), CornerRadius(4.dp.toPx()), style = Stroke(stroke))
            }

            ShrimpNavIcon.Users -> {
                drawCircle(color, radius = w * 0.1f, center = Offset(w * 0.43f, h * 0.36f), style = Stroke(stroke))
                drawCircle(color, radius = w * 0.08f, center = Offset(w * 0.66f, h * 0.4f), style = Stroke(thinStroke))
                drawArc(color, 205f, 125f, false, Offset(w * 0.2f, h * 0.5f), Size(w * 0.46f, h * 0.34f), style = Stroke(stroke, cap = StrokeCap.Round))
                drawArc(color, 210f, 110f, false, Offset(w * 0.48f, h * 0.55f), Size(w * 0.34f, h * 0.26f), style = Stroke(thinStroke, cap = StrokeCap.Round))
            }

            ShrimpNavIcon.Data, ShrimpNavIcon.Models -> {
                drawOval(color, Offset(w * 0.24f, h * 0.16f), Size(w * 0.52f, h * 0.18f), style = Stroke(stroke))
                drawLine(color, Offset(w * 0.24f, h * 0.25f), Offset(w * 0.24f, h * 0.72f), stroke, StrokeCap.Round)
                drawLine(color, Offset(w * 0.76f, h * 0.25f), Offset(w * 0.76f, h * 0.72f), stroke, StrokeCap.Round)
                drawArc(color, 0f, 180f, false, Offset(w * 0.24f, h * 0.62f), Size(w * 0.52f, h * 0.2f), style = Stroke(stroke, cap = StrokeCap.Round))
            }
        }
    }
}

private fun compactNavigationLabel(item: ShrimpNavigationItem): String {
    return when (item.route) {
        "home" -> "Trang chủ"
        "inference" -> "Kiểm tra"
        "history" -> "Lịch sử"
        "profile" -> "Hồ sơ"
        "settings" -> "Cài đặt"
        "admin_dashboard" -> "Tổng quan"
        "admin_users" -> "Người dùng"
        "admin_inference" -> "Kiểm tra"
        "admin_data" -> "Dữ liệu"
        else -> item.label
    }
}

private fun navIconForRoute(route: String): ShrimpNavIcon {
    return when (route) {
        "home" -> ShrimpNavIcon.Home
        "inference" -> ShrimpNavIcon.Diagnose
        "history" -> ShrimpNavIcon.History
        "profile" -> ShrimpNavIcon.Profile
        "settings" -> ShrimpNavIcon.Settings
        "admin_dashboard" -> ShrimpNavIcon.Dashboard
        "admin_users" -> ShrimpNavIcon.Users
        "admin_inference" -> ShrimpNavIcon.Inference
        "admin_data" -> ShrimpNavIcon.Data
        else -> ShrimpNavIcon.Home
    }
}
