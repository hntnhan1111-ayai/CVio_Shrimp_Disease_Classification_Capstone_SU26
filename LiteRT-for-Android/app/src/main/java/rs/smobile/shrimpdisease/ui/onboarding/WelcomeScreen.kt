package rs.smobile.shrimpdisease.ui.onboarding

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.ui.components.AppLogoImage
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.theme.CVioPrimaryFixed
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainer
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerHigh
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest

@Composable
fun WelcomeScreen(
    onGetStarted: () -> Unit,
    onLoginAsFarmer: () -> Unit,
    onLoginAsAdmin: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(CVioSurfaceContainer),
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(top = 56.dp, start = 20.dp, end = 20.dp, bottom = 24.dp),
            contentAlignment = Alignment.Center,
        ) {
            ScanningShrimpVisual()
        }

        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter),
            shape = RoundedCornerShape(topStart = 40.dp, topEnd = 40.dp),
            color = CVioSurfaceContainerLowest,
            shadowElevation = 12.dp,
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(18.dp),
            ) {
                Box(
                    modifier = Modifier
                        .size(width = 48.dp, height = 6.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.65f)),
                )
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(
                        text = "Shrimp Disease Classification",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface,
                        textAlign = TextAlign.Center,
                    )
                    Text(
                        text = "AI-powered shrimp health diagnosis",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.primary,
                        textAlign = TextAlign.Center,
                    )
                    Text(
                        text = "Capture or upload a shrimp image to detect possible disease signs.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                    )
                }

                PrimaryActionButton(
                    text = "Get Started",
                    onClick = onGetStarted,
                    modifier = Modifier.fillMaxWidth(),
                )

                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Text(
                        text = "Login as Farmer",
                        modifier = Modifier.clickable(onClick = onLoginAsFarmer),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.secondary,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "Login as Admin",
                        modifier = Modifier.clickable(onClick = onLoginAsAdmin),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

@Composable
private fun ScanningShrimpVisual() {
    val transition = rememberInfiniteTransition(label = "welcome-scan")
    val floatOffset by transition.animateFloat(
        initialValue = 0f,
        targetValue = -10f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 3200, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "float",
    )
    val scanProgress by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart,
        ),
        label = "scan",
    )

    Box(
        modifier = Modifier
            .size(300.dp)
            .offset(y = floatOffset.dp),
        contentAlignment = Alignment.Center,
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            shape = RoundedCornerShape(999.dp),
            color = CVioSurfaceContainerLowest.copy(alpha = 0.92f),
            shadowElevation = 14.dp,
        ) {
            Box(contentAlignment = Alignment.Center) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.radialGradient(
                                colors = listOf(
                                    CVioSurfaceContainerHigh.copy(alpha = 0.68f),
                                    Color.Transparent,
                                ),
                            )
                        ),
                )
                AppLogoImage(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(58.dp),
                )
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val y = size.height * scanProgress
                    drawLine(
                        brush = Brush.horizontalGradient(
                            colors = listOf(
                                Color.Transparent,
                                CVioPrimaryFixed.copy(alpha = 0.95f),
                                Color.Transparent,
                            ),
                        ),
                        start = Offset(0f, y),
                        end = Offset(size.width, y),
                        strokeWidth = 4.dp.toPx(),
                    )
                }
            }
        }
    }
}
