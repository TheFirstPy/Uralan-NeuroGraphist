package ru.hexteam.stenograf.core.model

import kotlinx.serialization.Serializable

/** Слово с временными метками (мс от начала записи). */
@Serializable
data class Word(
  val text: String,
  val startMs: Long,
  val endMs: Long,
  /** Перед словом стояло диалоговое тире (`—`) модели GigaAM v3, сигнализирующее о смене реплики. */
  val turnBreakBefore: Boolean = false,
)

/** Реплика одного говорящего. */
@Serializable
data class Utterance(
  val id: Int,
  val speakerId: String,
  val startMs: Long,
  val endMs: Long,
  val text: String,
  val words: List<Word> = emptyList(),
  /** Реплика была отредактирована пользователем вручную. */
  val edited: Boolean = false,
)

/** Длительность этапов обработки, мс. */
@Serializable
data class StageTimings(
  val decodeMs: Long = 0,
  val preprocessMs: Long = 0,
  val diarizationMs: Long = 0,
  val asrMs: Long = 0,
  val alignMs: Long = 0,
  val totalMs: Long = 0,
)

/** Режим скорости и качества обработки. */
@Serializable
enum class QualityPreset(
  val label: String,
  val subtitle: String,
) {
  FAST(
    label = "Быстро",
    subtitle = "Высокая скорость, облегчённая диаризация — оптимально для длинных записей (30–60 мин)",
  ),
  STANDARD(
    label = "Стандарт",
    subtitle = "Оптимальный баланс скорости и точности распознавания и диаризации",
  ),
  HIGH(
    label = "Долго (макс. качество)",
    subtitle = "Максимальная точность: шумоподавление GTCRN + детальная сегментация pyannote и узкие окна ASR",
  ),
}

/** Параметры, с которыми была получена расшифровка. */
@Serializable
data class ProcessingParams(
  val qualityPreset: QualityPreset = QualityPreset.STANDARD,
  val numSpeakers: Int = 0,
  val clusterThreshold: Float = 0.50f,
  val denoise: Boolean = false,
  val numThreads: Int = 4,
)

/** Полная запись о расшифровке, хранится локально в JSON. */
@Serializable
data class TranscriptRecord(
  val id: String,
  val title: String,
  val createdAt: Long,
  /** Путь к WAV 16 кГц моно (для воспроизведения и повторной обработки фрагментов). */
  val audioPath: String,
  val durationMs: Long,
  val utterances: List<Utterance>,
  /** speaker_N -> отображаемое имя, заданное пользователем. */
  val speakerNames: Map<String, String> = emptyMap(),
  /** Эталонные эмбеддинги голосов: speaker_N -> вектор. */
  val speakerEmbeddings: Map<String, List<Float>> = emptyMap(),
  val timings: StageTimings = StageTimings(),
  val params: ProcessingParams = ProcessingParams(),
  /** Протокол встречи, сформированный локальной LLM (Markdown). */
  val protocol: String? = null,
) {
  val speakers: List<String>
    get() = utterances.map { it.speakerId }.distinct().sortedBy { speakerIndex(it) }

  val rtf: Double
    get() = if (durationMs > 0) timings.totalMs.toDouble() / durationMs else 0.0

  fun displayName(speakerId: String): String =
    speakerNames[speakerId]?.takeIf { it.isNotBlank() } ?: defaultSpeakerName(speakerId)

  companion object {
    fun speakerIndex(id: String): Int = id.substringAfterLast('_').toIntOrNull() ?: 0
    fun defaultSpeakerName(id: String): String = "Говорящий ${speakerIndex(id)}"
  }
}
