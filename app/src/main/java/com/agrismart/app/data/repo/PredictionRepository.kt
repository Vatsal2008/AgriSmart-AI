package com.agrismart.app.data.repo

import com.agrismart.app.data.api.FakeAgriSmartApi
import com.agrismart.app.domain.model.PredictionResult
import com.agrismart.app.domain.model.ResultStatus
import com.agrismart.app.domain.model.TestRecord
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class PredictionRepository @Inject constructor(
    private val fakeApi: FakeAgriSmartApi
) {
    private val historyList = mutableListOf<TestRecord>(
        TestRecord("1", "tomato", "Tomato", "Late blight", ResultStatus.DISEASE, "Today, 10:30 AM"),
        TestRecord("2", "apple", "Apple", "Healthy leaf", ResultStatus.HEALTHY, "Yesterday, 4:15 PM"),
        TestRecord("3", "corn", "Corn", "Not sure", ResultStatus.UNCERTAIN, " Sep 09, 2:00 PM")
    )

    suspend fun getPrediction(cropId: String): PredictionResult {
        val dto = fakeApi.predict(cropId)
        val status = when (dto.status) {
            "disease" -> ResultStatus.DISEASE
            "healthy" -> ResultStatus.HEALTHY
            "crop_mismatch" -> ResultStatus.CROP_MISMATCH
            "unsupported" -> ResultStatus.UNSUPPORTED
            else -> ResultStatus.UNCERTAIN
        }

        val result = PredictionResult(
            testId = dto.testId,
            clientTestId = dto.clientTestId,
            status = status,
            cropName = dto.crop.name,
            diseaseName = dto.disease?.name,
            confidence = dto.confidence,
            confidenceLevel = dto.confidenceLevel,
            coverageNote = dto.coverageNote,
            adviceSummary = dto.advice?.summary ?: "No advice available",
            adviceSteps = dto.advice?.steps ?: emptyList(),
            preventionSteps = dto.advice?.prevention ?: emptyList()
        )

        // Save to local history
        val newRecord = TestRecord(
            id = dto.testId,
            cropId = cropId,
            cropName = dto.crop.name,
            resultTitle = dto.disease?.name ?: if (status == ResultStatus.HEALTHY) "Healthy leaf" else "Not sure",
            status = status,
            dateText = "Just now"
        )
        historyList.add(0, newRecord)

        return result
    }

    fun getRecentTests(): List<TestRecord> = historyList.toList()
}
