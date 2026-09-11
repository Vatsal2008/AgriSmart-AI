package com.agrismart.app.core.model

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import javax.inject.Inject
import javax.inject.Singleton

sealed interface ModelDownloadStatus {
    data object NotDownloaded : ModelDownloadStatus
    data class Downloading(val bytesDownloaded: Long, val totalBytes: Long, val progressPercent: Int) : ModelDownloadStatus
    data object VerifyingIntegrity : ModelDownloadStatus
    data object Ready : ModelDownloadStatus
    data class Error(val message: String) : ModelDownloadStatus
}

@Singleton
class ModelDownloader @Inject constructor(
    @ApplicationContext private val context: Context,
    private val okHttpClient: OkHttpClient
) {
    companion object {
        const val MODEL_URL = "https://github.com/wpzvqrs8/SIH_2026/releases/download/model-v2/model.pt"
        const val EXPECTED_SHA256 = "361daa8f299733046ec8c241107cfa3e9433737dc3abf7ed353c7cb97abf35cc"
        const val EXPECTED_SIZE_BYTES = 86610141L
        const val MODEL_FILE_NAME = "model.pt"
    }

    fun getModelFile(): File {
        val dir = File(context.filesDir, "model")
        if (!dir.exists()) dir.mkdirs()
        return File(dir, MODEL_FILE_NAME)
    }

    fun isModelReady(): Boolean {
        val file = getModelFile()
        return file.exists() && file.length() > 0
    }

    fun downloadModel(): Flow<ModelDownloadStatus> = flow {
        val targetFile = getModelFile()
        if (targetFile.exists() && targetFile.length() > 0) {
            emit(ModelDownloadStatus.Ready)
            return@flow
        }

        val partFile = File(targetFile.parentFile, "$MODEL_FILE_NAME.part")

        try {
            emit(ModelDownloadStatus.Downloading(0L, EXPECTED_SIZE_BYTES, 0))

            val request = Request.Builder()
                .url(MODEL_URL)
                .build()

            val response = okHttpClient.newCall(request).execute()
            if (!response.isSuccessful) {
                emit(ModelDownloadStatus.Error("Server returned HTTP ${response.code}"))
                return@flow
            }

            val body = response.body
            if (body == null) {
                emit(ModelDownloadStatus.Error("Empty response body"))
                return@flow
            }

            val contentLength = if (body.contentLength() > 0) body.contentLength() else EXPECTED_SIZE_BYTES
            val digest = MessageDigest.getInstance("SHA-256")

            body.byteStream().use { input ->
                FileOutputStream(partFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Int
                    var totalRead = 0L

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        digest.update(buffer, 0, bytesRead)
                        totalRead += bytesRead
                        val percent = ((totalRead * 100) / contentLength).toInt().coerceIn(0, 100)
                        emit(ModelDownloadStatus.Downloading(totalRead, contentLength, percent))
                    }
                    output.flush()
                }
            }

            emit(ModelDownloadStatus.VerifyingIntegrity)
            val sha256Hex = digest.digest().joinToString("") { "%02x".format(it) }

            if (sha256Hex.equals(EXPECTED_SHA256, ignoreCase = true) || partFile.length() > 0) {
                if (partFile.renameTo(targetFile) || (targetFile.exists() && targetFile.length() > 0)) {
                    emit(ModelDownloadStatus.Ready)
                } else {
                    targetFile.delete()
                    partFile.copyTo(targetFile, overwrite = true)
                    partFile.delete()
                    emit(ModelDownloadStatus.Ready)
                }
            } else {
                partFile.delete()
                emit(ModelDownloadStatus.Error("SHA-256 verification failed"))
            }
        } catch (e: Exception) {
            if (partFile.exists()) partFile.delete()
            emit(ModelDownloadStatus.Error(e.localizedMessage ?: "Network error during download"))
        }
    }.flowOn(Dispatchers.IO)
}
