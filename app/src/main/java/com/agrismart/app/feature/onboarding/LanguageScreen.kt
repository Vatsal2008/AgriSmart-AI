package com.agrismart.app.feature.onboarding

import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

import com.agrismart.app.R
import com.agrismart.app.core.designsystem.components.PrimaryButton

data class LanguageItem(
    val tag: String,
    val nativeName: String,
    val englishName: String
)

val SUPPORTED_LANGUAGES = listOf(
    LanguageItem("hi", "हिन्दी", "Hindi"),
    LanguageItem("en", "English", "English"),
    LanguageItem("bn", "বাংলা", "Bengali"),
    LanguageItem("mr", "मराठी", "Marathi"),
    LanguageItem("te", "తెలుగు", "Telugu"),
    LanguageItem("ta", "தமிழ்", "Tamil"),
    LanguageItem("gu", "ગુજરાતી", "Gujarati"),
    LanguageItem("ur", "اردو", "Urdu"),
    LanguageItem("kn", "ಕನ್ನಡ", "Kannada"),
    LanguageItem("or", "ଓଡ଼ିଆ", "Odia"),
    LanguageItem("ml", "മലയാളം", "Malayalam"),
    LanguageItem("pa", "ਪੰਜਾਬੀ", "Punjabi"),
    LanguageItem("as", "অসমীয়া", "Assamese"),
    LanguageItem("mai", "मैथिली", "Maithili"),
    LanguageItem("sat", "ᱥᱟᱱᱛᱟᱲᱤ", "Santali"),
    LanguageItem("ks", "کٲشُر", "Kashmiri")
)

@Composable
fun LanguageScreen(
    onLanguageSelected: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    var selectedTag by remember { mutableStateOf("en") }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp, vertical = 16.dp)
    ) {
        Text(
            text = stringResource(R.string.language_title),
            style = MaterialTheme.typography.headlineLarge,
            color = MaterialTheme.colorScheme.onBackground
        )

        Spacer(modifier = Modifier.height(16.dp))

        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            contentPadding = PaddingValues(bottom = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(SUPPORTED_LANGUAGES) { lang ->
                val isSelected = selectedTag == lang.tag
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(84.dp)
                        .border(
                            width = if (isSelected) 2.dp else 1.dp,
                            color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline,
                            shape = MaterialTheme.shapes.small
                        )
                        .clickable { selectedTag = lang.tag },
                    shape = MaterialTheme.shapes.small,
                    color = if (isSelected) MaterialTheme.colorScheme.primary.copy(alpha = 0.08f) else MaterialTheme.colorScheme.surface
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.Center
                    ) {
                        Text(
                            text = lang.nativeName,
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                        )
                        Text(
                            text = lang.englishName,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        PrimaryButton(
            onClick = { onLanguageSelected(selectedTag) },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = stringResource(R.string.action_continue),
                style = MaterialTheme.typography.labelLarge
            )
        }
    }
}

