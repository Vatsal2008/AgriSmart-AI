package com.agrismart.app.data.repo

import android.content.Context
import com.agrismart.app.data.api.model.CropsResponse
import com.agrismart.app.domain.model.Crop
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class CropRepository @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val json = Json { ignoreUnknownKeys = true }

    fun getOfflineCrops(): List<Crop> {
        return try {
            val content = context.assets.open("crops_fallback.json").bufferedReader().use { it.readText() }
            val response = json.decodeFromString(CropsResponse.serializer(), content)
            response.crops.map { dto ->
                Crop(
                    id = dto.id,
                    name = dto.name,
                    coverage = dto.coverage,
                    coverageNote = dto.coverageNote,
                    classes = dto.classes
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }
}
