package com.agrismart.app.data.api.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class HealthResponse(
    val status: String,
    @SerialName("model_version") val modelVersion: String
)

@Serializable
data class CropsResponse(
    val version: String,
    val crops: List<CropDto>
)

@Serializable
data class CropDto(
    val id: String,
    val name: String,
    val coverage: String,
    @SerialName("coverage_note") val coverageNote: String? = null,
    val classes: List<String> = emptyList()
)

@Serializable
data class PredictResponse(
    @SerialName("test_id") val testId: String,
    @SerialName("client_test_id") val clientTestId: String,
    @SerialName("model_version") val modelVersion: String,
    val status: String,
    val crop: NamedItemDto,
    @SerialName("crop_check") val cropCheck: CropCheckDto? = null,
    val disease: DiseaseDto? = null,
    val confidence: Float,
    @SerialName("confidence_level") val confidenceLevel: String,
    val alternatives: List<AlternativeDto> = emptyList(),
    @SerialName("coverage_note") val coverageNote: String? = null,
    val advice: AdviceDto? = null,
    val quality: QualityDto? = null
)

@Serializable
data class NamedItemDto(
    val id: String,
    val name: String
)

@Serializable
data class CropCheckDto(
    @SerialName("matches_selection") val matchesSelection: Boolean,
    @SerialName("detected_crop") val detectedCrop: NamedItemDto? = null
)

@Serializable
data class DiseaseDto(
    val id: String,
    val name: String,
    val label: String
)

@Serializable
data class AlternativeDto(
    val id: String,
    val name: String,
    val confidence: Float
)

@Serializable
data class AdviceDto(
    val summary: String,
    val steps: List<String> = emptyList(),
    val prevention: List<String> = emptyList(),
    val source: String? = null
)

@Serializable
data class QualityDto(
    val warnings: List<String> = emptyList()
)

@Serializable
data class ErrorResponse(
    val error: ApiErrorDetail
)

@Serializable
data class ApiErrorDetail(
    val code: String,
    val message: String
)
