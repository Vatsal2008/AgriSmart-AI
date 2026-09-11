package com.agrismart.app.feature.scan

import android.content.Intent
import android.net.Uri
import android.speech.tts.TextToSpeech
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.agrismart.app.R
import com.agrismart.app.core.designsystem.components.PrimaryButton
import com.agrismart.app.core.designsystem.components.SecondaryButton
import com.agrismart.app.core.designsystem.theme.LightAttention
import com.agrismart.app.domain.model.PredictionResult
import com.agrismart.app.domain.model.ResultStatus
import java.util.Locale

@Composable
fun ResultScreen(
    result: PredictionResult,
    onCheckAnotherClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current

    val tts = remember {
        var ttsEngine: TextToSpeech? = null
        ttsEngine = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                ttsEngine?.language = Locale.ENGLISH
            }
        }
        ttsEngine
    }

    DisposableEffect(Unit) {
        onDispose {
            tts.stop()
            tts.shutdown()
        }
    }

    val statusColor = when (result.status) {
        ResultStatus.DISEASE -> LightAttention
        ResultStatus.HEALTHY -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
    }

    val statusText = when (result.status) {
        ResultStatus.DISEASE -> stringResource(R.string.status_disease)
        ResultStatus.HEALTHY -> stringResource(R.string.status_healthy)
        else -> stringResource(R.string.status_not_sure)
    }

    val statusIcon = when (result.status) {
        ResultStatus.DISEASE -> Icons.Default.Warning
        ResultStatus.HEALTHY -> Icons.Default.CheckCircle
        else -> Icons.Default.Info
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp)
    ) {
        Text(
            text = result.cropName,
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f)
        )

        Text(
            text = result.diseaseName ?: if (result.status == ResultStatus.HEALTHY) "Healthy leaf" else stringResource(R.string.status_not_sure),
            style = MaterialTheme.typography.displayLarge,
            color = MaterialTheme.colorScheme.onBackground,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Status Chip
        Row(
            modifier = Modifier
                .background(statusColor.copy(alpha = 0.12f), MaterialTheme.shapes.small)
                .padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(statusIcon, contentDescription = null, tint = statusColor, modifier = Modifier.size(20.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text(text = statusText, style = MaterialTheme.typography.labelLarge, color = statusColor, fontWeight = FontWeight.Bold)
        }

        Spacer(modifier = Modifier.height(16.dp))

        // How sure plain words
        Text(
            text = "Confidence: ${if (result.confidenceLevel == "high") stringResource(R.string.how_sure_high) else stringResource(R.string.how_sure_medium)} (${(result.confidence * 100).toInt()}%)",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)
        )

        Spacer(modifier = Modifier.height(20.dp))

        // Listen Button
        SecondaryButton(
            onClick = {
                val speechText = "${result.cropName}. ${result.diseaseName ?: "Healthy leaf"}. ${result.adviceSummary}"
                tts.speak(speechText, TextToSpeech.QUEUE_FLUSH, null, null)
            }
        ) {
            Icon(painter = painterResource(id = R.drawable.ic_volume_up), contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text(text = stringResource(R.string.listen), style = MaterialTheme.typography.labelLarge)
        }

        Spacer(modifier = Modifier.height(24.dp))

        // What to do
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.small,
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = stringResource(R.string.what_to_do),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = result.adviceSummary,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface
                )

                result.adviceSteps.forEach { step ->
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "• $step",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Check another plant
        PrimaryButton(onClick = onCheckAnotherClick) {
            Text(text = stringResource(R.string.check_another), style = MaterialTheme.typography.labelLarge)
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Disclaimer & Call Kisan Call Centre
        Text(
            text = stringResource(R.string.disclaimer),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)
        )

        Spacer(modifier = Modifier.height(8.dp))

        SecondaryButton(
            onClick = {
                val intent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:18001801551"))
                context.startActivity(intent)
            }
        ) {
            Icon(Icons.Default.Call, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text(text = stringResource(R.string.call_kcc), style = MaterialTheme.typography.labelLarge)
        }
    }
}
