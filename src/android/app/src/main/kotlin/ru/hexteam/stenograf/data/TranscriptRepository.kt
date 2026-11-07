package ru.hexteam.stenograf.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import ru.hexteam.stenograf.core.model.TranscriptRecord
import java.io.File

/**
 * Локальное хранилище расшифровок на устройстве (каталог `files/records/`).
 * Позволяет сохранять историю всех заседаний, переименовывать говорящих,
 * редактировать реплики и хранить сгенерированный локальной LLM протокол.
 */
class TranscriptRepository(context: Context) {

  private val recordsDir = File(context.filesDir, "records").apply { mkdirs() }
  private val json = Json {
    prettyPrint = true
    ignoreUnknownKeys = true
    encodeDefaults = true
  }

  suspend fun loadAll(): List<TranscriptRecord> = withContext(Dispatchers.IO) {
    recordsDir.listFiles { f -> f.extension == "json" }
      ?.mapNotNull { file ->
        runCatching { json.decodeFromString<TranscriptRecord>(file.readText()) }.getOrNull()
      }
      ?.sortedByDescending { it.createdAt }
      ?: emptyList()
  }

  suspend fun save(record: TranscriptRecord) = withContext(Dispatchers.IO) {
    val file = File(recordsDir, "${record.id}.json")
    file.writeText(json.encodeToString(record))
  }

  suspend fun delete(record: TranscriptRecord) = withContext(Dispatchers.IO) {
    File(recordsDir, "${record.id}.json").delete()
    runCatching { File(record.audioPath).delete() }
  }
}
