package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun InferenceActionPanel(
    hasImage: Boolean,
    hasResult: Boolean,
    isLoading: Boolean,
    onRunInference: () -> Unit,
    onSaveResult: () -> Unit,
    onGoHome: () -> Unit,
    onHistory: () -> Unit,
    modifier: Modifier = Modifier,
) {
    CVioCard(modifier = modifier) {
        Text(
            text = "Thao tác",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )
        if (isLoading) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                Text(
                    text = "Đang phân tích ảnh tôm...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        PrimaryActionButton(
            text = if (hasResult) "Kiểm tra lại" else "Bắt đầu kiểm tra",
            enabled = hasImage && !isLoading,
            onClick = onRunInference,
            modifier = Modifier.fillMaxWidth(),
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            SecondaryActionButton(
                text = "Lưu kết quả",
                enabled = hasResult,
                onClick = onSaveResult,
                modifier = Modifier.weight(1f),
            )
            SecondaryActionButton(
                text = "Lịch sử",
                onClick = onHistory,
                modifier = Modifier.weight(1f),
            )
        }

        SecondaryActionButton(
            text = "Trang chủ",
            onClick = onGoHome,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}
