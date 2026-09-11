package com.agrismart.app.feature.scan

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.view.ViewGroup
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.agrismart.app.R
import com.agrismart.app.core.designsystem.components.PrimaryButton
import com.agrismart.app.core.designsystem.theme.LightAttention
import com.agrismart.app.data.repo.PredictionRepository
import com.agrismart.app.domain.model.PredictionResult
import com.agrismart.app.domain.model.ResultStatus
import kotlinx.coroutines.delay
import java.util.concurrent.Executors

@Composable
fun LiveCameraScreen(
    predictionRepository: PredictionRepository,
    onResultConfirmed: (PredictionResult) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    var livePrediction by remember { mutableStateOf<PredictionResult?>(null) }
    var liveHint by remember { mutableStateOf("Hold steady over leaf") }
    var isScanning by remember { mutableStateOf(true) }

    // Simulated real-time frame analyzer cycling mock predictions
    LaunchedEffect(isScanning) {
        val crops = listOf("tomato", "apple", "potato", "corn")
        var index = 0
        while (isScanning) {
            val cropId = crops[index % crops.size]
            livePrediction = predictionRepository.getPrediction(cropId)
            liveHint = when (index % 3) {
                0 -> "Good daylight • Hold still"
                1 -> "Leaf in center frame"
                else -> "Hold steady for best accuracy"
            }
            index++
            delay(2000) // update real-time analysis every 2 sec
        }
    }

    if (!hasCameraPermission) {
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(painter = painterResource(id = R.drawable.ic_photo_camera), contentDescription = null, modifier = Modifier.size(64.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.height(16.dp))
            Text(text = stringResource(R.string.camera_primer), style = MaterialTheme.typography.titleLarge, textAlign = TextAlign.Center)
            Spacer(modifier = Modifier.height(24.dp))
            PrimaryButton(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                Text(text = stringResource(R.string.allow_camera), style = MaterialTheme.typography.labelLarge)
            }
        }
        return
    }

    Box(modifier = modifier.fillMaxSize().background(androidx.compose.ui.graphics.Color.Black)) {
        // CameraX Preview
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }

                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview)
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Center Square Leaf Frame Overlay
        Box(
            modifier = Modifier
                .size(280.dp)
                .align(Alignment.Center)
                .border(3.dp, MaterialTheme.colorScheme.primary, MaterialTheme.shapes.medium)
        )

        // Real-Time Top Scan Card Overlay
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(16.dp)
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f)),
                shape = MaterialTheme.shapes.medium
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(10.dp)
                                    .background(MaterialTheme.colorScheme.primary, CircleShape)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "REAL-TIME LIVE SCAN",
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }

                        Text(
                            text = liveHint,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    livePrediction?.let { pred ->
                        val statusColor = when (pred.status) {
                            ResultStatus.DISEASE -> LightAttention
                            ResultStatus.HEALTHY -> MaterialTheme.colorScheme.primary
                            else -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        }

                        val statusIcon = when (pred.status) {
                            ResultStatus.DISEASE -> Icons.Default.Warning
                            ResultStatus.HEALTHY -> Icons.Default.CheckCircle
                            else -> Icons.Default.Info
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(statusIcon, contentDescription = null, tint = statusColor, modifier = Modifier.size(24.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text(
                                    text = "${pred.cropName}: ${pred.diseaseName ?: "Healthy Leaf"}",
                                    style = MaterialTheme.typography.titleLarge,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "${if (pred.confidenceLevel == "high") "Very likely" else "Likely"} (${(pred.confidence * 100).toInt()}%)",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = statusColor
                                )
                            }
                        }
                    }
                }
            }
        }

        // Bottom Capture Controls
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            PrimaryButton(
                onClick = {
                    livePrediction?.let { onResultConfirmed(it) }
                }
            ) {
                Icon(painter = painterResource(id = R.drawable.ic_photo_camera), contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(text = "Capture & Confirm Result", style = MaterialTheme.typography.labelLarge)
            }
        }
    }
}
