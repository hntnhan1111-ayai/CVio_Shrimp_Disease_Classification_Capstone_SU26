package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.R
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest

@Composable
fun AppLogoImage(
    modifier: Modifier = Modifier,
    contentDescription: String = "CVio logo",
    contentScale: ContentScale = ContentScale.Fit,
) {
    Image(
        painter = painterResource(id = R.drawable.cvio_logo),
        contentDescription = contentDescription,
        modifier = modifier,
        contentScale = contentScale,
    )
}

@Composable
fun AppLogoMark(
    modifier: Modifier = Modifier,
    size: Dp = 48.dp,
    padding: Dp = 6.dp,
    showFullName: Boolean = true,
) {
    AppLogoImage(
        modifier = modifier
            .size(
                width = if (showFullName) size * 2.55f else size,
                height = size,
            )
            .clip(RoundedCornerShape(if (showFullName) 18.dp else 999.dp))
            .background(CVioSurfaceContainerLowest)
            .padding(padding),
        contentScale = ContentScale.Crop,
    )
}

@Composable
fun AppBrandLogo(
    modifier: Modifier = Modifier,
    showText: Boolean = true,
) {
    Row(
        modifier = modifier,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        AppLogoMark(
            size = if (showText) 40.dp else 32.dp,
            padding = if (showText) 2.dp else 3.dp,
            showFullName = showText,
        )
    }
}
