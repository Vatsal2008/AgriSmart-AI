package com.agrismart.app.data.api

import android.content.Context
import com.agrismart.app.BuildConfig
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ApiConfig @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val prefs = context.getSharedPreferences("agrismart_api_config", Context.MODE_PRIVATE)

    var serverUrl: String
        get() {
            val saved = prefs.getString("custom_server_url", "") ?: ""
            if (saved.isNotBlank()) return saved
            val raw = BuildConfig.API_BASE_URL.trim()
            return if (raw.isNotBlank()) raw else "http://192.168.1.100:8765"
        }
        set(value) {
            prefs.edit().putString("custom_server_url", value.trim()).apply()
        }

    val isConfigured: Boolean
        get() = serverUrl.isNotBlank()

    val baseUrl: String
        get() {
            val url = serverUrl
            return if (url.endsWith("/")) url else "$url/"
        }
}
