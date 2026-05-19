package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.R
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest

@Composable
fun AppLogoImage(
    modifier: Modifier = Modifier,
    contentDescription: String = "CVio logo",
) {
    Image(
        painter = painterResource(id = R.drawable.cvio_logo),
        contentDescription = contentDescription,
        modifier = modifier,
        contentScale = ContentScale.Fit,
    )
}

@Composable
fun AppLogoMark(
    modifier: Modifier = Modifier,
    size: Dp = 48.dp,
    padding: Dp = 6.dp,
) {
    AppLogoImage(
        modifier = modifier
            .size(size)
            .clip(RoundedCornerShape(999.dp))
            .background(CVioSurfaceContainerLowest)
            .padding(padding),
    )
}

@Composable
fun AppBrandLogo(
    modifier: Modifier = Modifier,
    showText: Boolean = true,
) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        AppLogoMark(size = 32.dp, padding = 3.dp)
        if (showText) {
            Text(
                text = "CVio",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}
