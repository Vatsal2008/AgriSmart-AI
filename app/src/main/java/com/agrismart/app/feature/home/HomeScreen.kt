package com.agrismart.app.feature.home

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.agrismart.app.R
import com.agrismart.app.core.designsystem.components.PrimaryButton
import com.agrismart.app.core.designsystem.components.SecondaryButton

@Composable
fun HomeScreen(
    onCheckPlantClick: () -> Unit,
    onLiveCameraClick: () -> Unit,
    onRecentTestsClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(20.dp)
    ) {
        Text(
            text = stringResource(R.string.app_name),
            style = MaterialTheme.typography.displayLarge,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = stringResource(R.string.app_subtitle),
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f)
        )

        Spacer(modifier = Modifier.height(32.dp))

        PrimaryButton(onClick = onLiveCameraClick) {
            Icon(painter = painterResource(id = R.drawable.ic_photo_camera), contentDescription = null)
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = "Live Camera Scan", style = MaterialTheme.typography.labelLarge)
        }

        Spacer(modifier = Modifier.height(16.dp))

        SecondaryButton(onClick = onCheckPlantClick) {
            Text(text = stringResource(R.string.home_check), style = MaterialTheme.typography.labelLarge)
        }

        Spacer(modifier = Modifier.height(16.dp))

        SecondaryButton(onClick = onRecentTestsClick) {
            Text(text = stringResource(R.string.home_recent), style = MaterialTheme.typography.labelLarge)
        }
    }
}

