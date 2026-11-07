package ru.hexteam.stenograf.core.export

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import ru.hexteam.stenograf.core.model.TranscriptRecord
import ru.hexteam.stenograf.core.model.Utterance

/**
 * Строгий формат JSON-вывода согласно стр. 14–15 задания Nuclear IT Hack:
 * ```json
 * {
 *   "transcript": [
 *     {
 *       "speaker_id": "speaker_1",
 *       "start_ms": 1250,
 *       "end_ms": 4380,
 *       "text": "Добрый день!"
 *     }
 *   ]
 * }
 * ```
 */
@Serializable
data class HackathonTranscriptJson(
  @SerialName("transcript")
  val transcript: List<HackathonTurnJson>,
)

@Serializable
data class HackathonTurnJson(
  @SerialName("speaker_id")
  val speakerId: String,
  @SerialName("start_ms")
  val startMs: Long,
  @SerialName("end_ms")
  val endMs: Long,
  @SerialName("text")
  val text: String,
)

enum class ExportFormat(
  val label: String,
  val extension: String,
  val mimeType: String,
  val description: String,
) {
  JSON_HACKATHON(
    label = "JSON (эталон хакатона)",
    extension = "json",
    mimeType = "application/json",
    description = "Структура transcript / speaker_id / start_ms / end_ms / text",
  ),
  MARKDOWN(
    label = "Протокол Markdown (.md)",
    extension = "md",
    mimeType = "text/markdown",
    description = "Расшифровка с именами участников + ИИ-протокол заседания",
  ),
  SRT(
    label = "Субтитры SRT (.srt)",
    extension = "srt",
    mimeType = "application/x-subrip",
    description = "Таймкоды и реплики для видеоплееров",
  ),
  TXT(
    label = "Текст (.txt)",
    extension = "txt",
    mimeType = "text/plain",
    description = "Читаемая стенограмма с таймкодами",
  ),
}

object TranscriptExporter {

  private val prettyJson = Json {
    prettyPrint = true
    encodeDefaults = true
  }

  /** Формирует JSON строго по спецификации хакатона (стр. 15 презентации задания). */
  fun toHackathonJson(utterances: List<Utterance>): String {
    val payload = HackathonTranscriptJson(
      transcript = utterances.map { u ->
        HackathonTurnJson(
          speakerId = u.speakerId,
          startMs = u.startMs,
          endMs = u.endMs,
          text = u.text,
        )
      }
    )
    return prettyJson.encodeToString(payload)
  }

  fun export(record: TranscriptRecord, format: ExportFormat): String = when (format) {
    ExportFormat.JSON_HACKATHON -> toHackathonJson(record.utterances)
    ExportFormat.MARKDOWN -> toMarkdown(record)
    ExportFormat.SRT -> toSrt(record)
    ExportFormat.TXT -> toPlainText(record)
  }

  fun toPlainText(record: TranscriptRecord): String = buildString {
    appendLine("Стенограмма: ${record.title}")
    appendLine("Длительность: ${formatTimeMs(record.durationMs)} | Участников: ${record.speakers.size}")
    appendLine("=".repeat(60))
    appendLine()
    for (u in record.utterances) {
      val name = record.displayName(u.speakerId)
      appendLine("[${formatTimeMs(u.startMs)} – ${formatTimeMs(u.endMs)}] $name:")
      appendLine(u.text)
      appendLine()
    }
    if (!record.protocol.isNullOrBlank()) {
      appendLine("=".repeat(60))
      appendLine("ПРОТОКОЛ ЗАСЕДАНИЯ (локальная LLM)")
      appendLine("=".repeat(60))
      val cleanProtocol = record.protocol
        .replace(Regex("(?m)^#{1,6}\\s*"), "")
        .replace("**", "")
        .replace("`", "")
      appendLine(cleanProtocol)
    }
  }

  fun toMarkdown(record: TranscriptRecord): String = buildString {
    appendLine("# ${record.title}")
    appendLine()
    appendLine("- **Длительность:** `${formatTimeMs(record.durationMs)}`")
    appendLine("- **Участников:** ${record.speakers.size} (${record.speakers.joinToString { "${record.displayName(it)} [`$it`]" }})")
    appendLine("- **Скорость обработки (RTF):** ${"%.2f".format(record.rtf)}")
    appendLine()
    if (!record.protocol.isNullOrBlank()) {
      appendLine("## Протокол заседания (локальная LLM)")
      appendLine()
      appendLine(record.protocol)
      appendLine()
    }
    appendLine("## Полная расшифровка диалога")
    appendLine()
    for (u in record.utterances) {
      val name = record.displayName(u.speakerId)
      appendLine("### `${formatTimeMs(u.startMs)} – ${formatTimeMs(u.endMs)}` · **$name** (`${u.speakerId}`)")
      appendLine(u.text)
      appendLine()
    }
  }

  fun toSrt(record: TranscriptRecord): String = buildString {
    record.utterances.forEachIndexed { idx, u ->
      appendLine(idx + 1)
      appendLine("${formatSrtTime(u.startMs)} --> ${formatSrtTime(u.endMs)}")
      appendLine("[${record.displayName(u.speakerId)}] ${u.text}")
      appendLine()
    }
  }

  fun formatTimeMs(ms: Long): String {
    val totalSec = (ms / 1000).coerceAtLeast(0)
    val min = totalSec / 60
    val sec = totalSec % 60
    val tenths = (ms % 1000) / 100
    return "%02d:%02d.%d".format(min, sec, tenths)
  }

  private val TIMECODE_REGEX = Regex("\\[?(\\d{1,3}):(\\d{2})(?:\\.(\\d))?\\]?")

  /**
   * Извлекает первый таймкод вида `MM:SS.d` или `[MM:SS.d]` из строки протокола и переводит его в миллисекунды.
   */
  fun extractFirstTimestampMs(text: String): Long? {
    val match = TIMECODE_REGEX.find(text) ?: return null
    val min = match.groupValues[1].toLongOrNull() ?: return null
    val sec = match.groupValues[2].toLongOrNull() ?: return null
    val tenths = match.groupValues.getOrNull(3)?.takeIf { it.isNotEmpty() }?.toLongOrNull() ?: 0L
    if (sec >= 60) return null
    return min * 60_000L + sec * 1_000L + tenths * 100L
  }

  private fun formatSrtTime(ms: Long): String {
    val clamped = ms.coerceAtLeast(0)
    val h = clamped / 3_600_000
    val m = (clamped % 3_600_000) / 60_000
    val s = (clamped % 60_000) / 1_000
    val millis = clamped % 1_000
    return "%02d:%02d:%02d,%03d".format(h, m, s, millis)
  }
}
