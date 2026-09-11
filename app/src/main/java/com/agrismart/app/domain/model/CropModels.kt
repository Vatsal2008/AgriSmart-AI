package com.agrismart.app.domain.model

data class Crop(
    val id: String,
    val name: String,
    val coverage: String, // "full", "healthy_only", "diseases_only"
    val coverageNote: String? = null,
    val classes: List<String> = emptyList()
)

enum class ResultStatus {
    DISEASE, HEALTHY, UNCERTAIN, UNSUPPORTED, CROP_MISMATCH
}

data class PredictionResult(
    val testId: String,
    val clientTestId: String,
    val status: ResultStatus,
    val cropName: String,
    val diseaseName: String?,
    val confidence: Float,
    val confidenceLevel: String, // "high", "medium", "low"
    val coverageNote: String?,
    val adviceSummary: String,
    val adviceSteps: List<String>,
    val preventionSteps: List<String>,
    val imageUri: String? = null
)

data class TestRecord(
    val id: String,
    val cropId: String,
    val cropName: String,
    val resultTitle: String,
    val status: ResultStatus,
    val dateText: String,
    val imageUri: String? = null
)
