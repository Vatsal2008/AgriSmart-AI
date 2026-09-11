package com.agrismart.app.data.api

import android.content.Context
import com.agrismart.app.data.api.model.CropsResponse
import com.agrismart.app.data.api.model.HealthResponse
import com.agrismart.app.data.api.model.PredictResponse
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.delay
import kotlinx.serialization.json.Json
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class FakeAgriSmartApi @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun getHealth(): HealthResponse {
        delay(300)
        return HealthResponse(status = "ok", modelVersion = "model-v2")
    }

    suspend fun getCrops(): CropsResponse {
        delay(500)
        val content = context.assets.open("crops_fallback.json").bufferedReader().use { it.readText() }
        return json.decodeFromString(CropsResponse.serializer(), content)
    }

    suspend fun predict(cropId: String): PredictResponse {
        delay(1200) // 1-2 sec artificial delay
        val fileName = when (cropId) {
            "tomato" -> "predict_disease.json"
            "apple" -> "predict_healthy.json"
            "potato" -> "predict_mismatch.json"
            "other" -> "predict_unsupported.json"
            else -> "predict_uncertain.json"
        }
        val content = context.assets.open("mock/$fileName").bufferedReader().use { it.readText() }
        return json.decodeFromString(PredictResponse.serializer(), content)
    }
}
