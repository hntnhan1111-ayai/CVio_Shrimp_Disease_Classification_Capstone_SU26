package rs.smobile.shrimpdisease.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (isLoading) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                    Text(
                        text = "Running inference...",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Button(
                    enabled = hasImage && !isLoading,
                    onClick = onRunInference,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Run Inference")
                }
                OutlinedButton(
                    enabled = hasResult,
                    onClick = onSaveResult,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Save Result")
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                OutlinedButton(
                    onClick = onGoHome,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "Home")
                }
                OutlinedButton(
                    onClick = onHistory,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = "History")
                }
            }
        }
    }
}
