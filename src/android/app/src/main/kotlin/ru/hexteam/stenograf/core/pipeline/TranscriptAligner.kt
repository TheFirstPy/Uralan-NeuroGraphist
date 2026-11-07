package ru.hexteam.stenograf.core.pipeline

import ru.hexteam.stenograf.core.model.Utterance
import ru.hexteam.stenograf.core.model.Word
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** Сегмент активности говорящего от модуля диаризации (время в мс). */
data class DiarizationSlice(
  val startMs: Long,
  val endMs: Long,
  val speakerIndex: Int,
)

/** Сырой токен субсловного распознавателя (GigaAM v3 RNN-T) с временной меткой. */
data class RawToken(
  val token: String,
  val timestampMs: Long,
)

/**
 * Алгоритм объединения субсловных токенов ASR и сегментов диаризации в структурированные реплики:
 * 1) Собирает BPE-токены (с префиксом пробела ` ` или `▁` и знаками препинания) в целые слова с точными `startMs` и `endMs`.
 * 2) Каждому слову сопоставляет говорящего по максимальному перекрытию с сегментами диаризации (или ближайшему по времени).
 * 3) Сглаживает одиночные ложные переключения спикера внутри быстрой фразы (медианный фильтр при паузе < 350 мс).
 * 4) Группирует слова в реплики по смене говорящего или длинной паузе (> 1500 мс), отфильтровывает случайные
 *    шумовые микрокластеры-одиночки и приводит ID спикеров к порядку появления (`speaker_1`, `speaker_2`, ...).
 */
object TranscriptAligner {

  private val PUNCT_ONLY = setOf(".", ",", "?", "!", ":", ";", "—", "-", "»", "«", "...")

  /** Собирает субсловные токены в список слов с временными границами в пределах окна [chunkStartMs, chunkEndMs]. */
  fun tokensToWords(
    tokens: List<RawToken>,
    chunkStartMs: Long,
    chunkEndMs: Long,
  ): List<Word> {
    if (tokens.isEmpty()) return emptyList()
    val words = mutableListOf<Word>()
    var currentText = StringBuilder()
    var wordStart = -1L
    var lastTokenTime = chunkStartMs

    fun flushWord(nextWordStartMs: Long) {
      if (currentText.isEmpty()) return
      val raw = currentText.toString().trim()
      if (raw.isNotEmpty() && raw !in PUNCT_ONLY) {
        val s = wordStart.coerceAtLeast(chunkStartMs)
        val estimatedEnd = max(s + 80L, min(nextWordStartMs, lastTokenTime + 240L))
        val e = estimatedEnd.coerceIn(s + 40L, max(s + 40L, chunkEndMs))
        words += Word(text = raw, startMs = s, endMs = e)
      }
      currentText = StringBuilder()
      wordStart = -1L
    }

    for (raw in tokens) {
      val piece = raw.token.replace('▁', ' ')
      val tMs = (chunkStartMs + raw.timestampMs).coerceIn(chunkStartMs, chunkEndMs)
      if (piece.isBlank()) {
        // Самостоятельный токен "▁" в словаре SentencePiece (GigaAM v3) обозначает пробел между словами
        flushWord(tMs)
        continue
      }
      val trimmed = piece.trim()
      val isPunct = trimmed in PUNCT_ONLY
      val startsNewWord = piece.startsWith(" ") && !isPunct && currentText.isNotEmpty()

      if (startsNewWord) {
        flushWord(tMs)
      }

      if (isPunct) {
        if (currentText.isNotEmpty()) {
          currentText.append(trimmed)
          lastTokenTime = max(lastTokenTime, tMs)
        } else if (words.isNotEmpty()) {
          val prev = words.removeAt(words.lastIndex)
          words += prev.copy(
            text = prev.text + trimmed,
            endMs = max(prev.endMs, min(chunkEndMs, tMs + 60L)),
          )
        }
      } else {
        if (wordStart < 0L) {
          wordStart = max(chunkStartMs, tMs - 40L)
        }
        if (currentText.isEmpty()) {
          currentText.append(trimmed)
        } else {
          currentText.append(piece)
        }
        lastTokenTime = tMs
      }
    }
    flushWord(chunkEndMs)
    return words
  }

  /** Назначает каждому слову индекс спикера по пересечению с сегментами диаризации. */
  fun assignSpeakersToWords(
    words: List<Word>,
    slices: List<DiarizationSlice>,
  ): List<Pair<Word, Int>> {
    if (words.isEmpty()) return emptyList()
    if (slices.isEmpty()) return words.map { it to 0 }

    val rawAssigned = words.map { word ->
      val spk = findBestSpeaker(word.startMs, word.endMs, slices)
      word to spk
    }.toMutableList()

    // Устойчивое определение участников: убираем одиночный «скачок» спикера на 1–2 коротких слова
    // внутри непрерывной фразы, если вокруг него один и тот же говорящий и нет паузы.
    if (rawAssigned.size >= 3) {
      for (i in 1 until rawAssigned.lastIndex) {
        val (prevW, prevSpk) = rawAssigned[i - 1]
        val (curW, curSpk) = rawAssigned[i]
        val (nextW, nextSpk) = rawAssigned[i + 1]
        val gapBefore = curW.startMs - prevW.endMs
        val gapAfter = nextW.startMs - curW.endMs
        val wordDur = curW.endMs - curW.startMs
        if (prevSpk == nextSpk && curSpk != prevSpk && gapBefore < 400L && gapAfter < 400L && wordDur < 600L) {
          rawAssigned[i] = curW to prevSpk
        }
      }
    }
    return rawAssigned
  }

  private fun findBestSpeaker(startMs: Long, endMs: Long, slices: List<DiarizationSlice>): Int {
    val overlapBySpeaker = HashMap<Int, Long>()
    for (s in slices) {
      val ov = min(endMs, s.endMs) - max(startMs, s.startMs)
      if (ov > 0L) {
        overlapBySpeaker[s.speakerIndex] = (overlapBySpeaker[s.speakerIndex] ?: 0L) + ov
      }
    }
    if (overlapBySpeaker.isNotEmpty()) {
      return overlapBySpeaker.maxByOrNull { it.value }!!.key
    }
    // Если прямого перекрытия нет — берём ближайший по времени сегмент
    val mid = (startMs + endMs) / 2
    return slices.minByOrNull { s ->
      when {
        mid < s.startMs -> s.startMs - mid
        mid > s.endMs -> mid - s.endMs
        else -> 0L
      }
    }?.speakerIndex ?: 0
  }

  /**
   * Формирует итоговые реплики из слов с назначенными спикерами.
   * При автоопределении числа участников (`filterMinorSpeakers = true`) поглощает ложные
   * шумовые кластеры-одиночки и канонизирует идентификаторы в порядке первого появления: `speaker_1`, `speaker_2`, ...
   */
  fun buildUtterances(
    assignedWords: List<Pair<Word, Int>>,
    maxPauseWithinTurnMs: Long = 1500L,
    filterMinorSpeakers: Boolean = true,
  ): List<Utterance> {
    if (assignedWords.isEmpty()) return emptyList()

    data class RawTurn(
      var rawSpeaker: Int,
      val words: MutableList<Word>,
    ) {
      val startMs: Long get() = words.first().startMs
      val endMs: Long get() = words.last().endMs
      val durationMs: Long get() = (endMs - startMs).coerceAtLeast(80L)
    }

    val rawTurns = mutableListOf<RawTurn>()
    var currentWords = mutableListOf<Word>()
    var currentRawSpeaker = assignedWords.first().second

    fun flushRawTurn() {
      if (currentWords.isEmpty()) return
      rawTurns += RawTurn(currentRawSpeaker, currentWords)
      currentWords = mutableListOf()
    }

    for ((word, spk) in assignedWords) {
      if (currentWords.isEmpty()) {
        currentRawSpeaker = spk
        currentWords += word
      } else {
        val pause = word.startMs - currentWords.last().endMs
        val prevEndsSentence = currentWords.last().text.let {
          it.endsWith(".") || it.endsWith("!") || it.endsWith("?")
        }
        val shouldSplit = spk != currentRawSpeaker ||
          pause > maxPauseWithinTurnMs ||
          (pause > 850L && prevEndsSentence)
        if (shouldSplit) {
          flushRawTurn()
          currentRawSpeaker = spk
        }
        currentWords += word
      }
    }
    flushRawTurn()

    // Подавление только откровенно шумовых микрокластеров-одиночек при автокластеризации:
    // срабатывает ТОЛЬКО если обнаружено > 3 спикеров и у кластера всего 1 микро-реплика
    // короче 1,1 с (< 1,5% от общей речи и <= 2 слов). Реальных 2-го и 3-го говорящих не сливает никогда.
    if (filterMinorSpeakers && rawTurns.size >= 8) {
      val totalSpeechMs = rawTurns.sumOf { it.durationMs }.coerceAtLeast(1L)
      val statsBySpk = rawTurns.groupBy { it.rawSpeaker }
      if (statsBySpk.size > 3) {
        val majorSpeakers = statsBySpk.filter { (_, turns) ->
          val dur = turns.sumOf { it.durationMs }
          val totalWords = turns.sumOf { it.words.size }
          turns.size >= 2 || dur >= 1100L || totalWords >= 3 || (dur.toDouble() / totalSpeechMs) >= 0.015
        }.keys

        if (majorSpeakers.size >= 2 && majorSpeakers.size < statsBySpk.size) {
          for (i in rawTurns.indices) {
            val t = rawTurns[i]
            if (t.rawSpeaker !in majorSpeakers) {
              val nearestMajor = rawTurns
                .filter { it.rawSpeaker in majorSpeakers }
                .minByOrNull { other ->
                  min(abs(t.startMs - other.endMs), abs(other.startMs - t.endMs))
                }?.rawSpeaker ?: majorSpeakers.first()
              t.rawSpeaker = nearestMajor
            }
          }
        }
      }
    }

    // Склеиваем соседние реплики одного и того же спикера, если пауза между ними небольшая
    val mergedTurns = mutableListOf<RawTurn>()
    for (t in rawTurns) {
      if (mergedTurns.isNotEmpty()) {
        val prev = mergedTurns.last()
        val pause = t.startMs - prev.endMs
        if (prev.rawSpeaker == t.rawSpeaker && pause <= maxPauseWithinTurnMs) {
          prev.words.addAll(t.words)
          continue
        }
      }
      mergedTurns += t
    }

    // Перенумерация спикеров по порядку первого появления в диалоге (speaker_1, speaker_2, ...)
    val speakerOrder = LinkedHashMap<Int, String>()
    fun canonicalSpeaker(rawIdx: Int): String =
      speakerOrder.getOrPut(rawIdx) { "speaker_${speakerOrder.size + 1}" }

    val utterances = mutableListOf<Utterance>()
    for (turn in mergedTurns) {
      val text = normalizeRussianPunctuation(turn.words.joinToString(" ") { it.text })
      if (text.isNotBlank()) {
        val spkId = canonicalSpeaker(turn.rawSpeaker)
        utterances += Utterance(
          id = utterances.size + 1,
          speakerId = spkId,
          startMs = turn.startMs,
          endMs = turn.endMs,
          text = text,
          words = turn.words.toList(),
        )
      }
    }
    return utterances
  }

  /** Приводит русскую пунктуацию и регистр начала реплики к аккуратному виду. */
  fun normalizeRussianPunctuation(raw: String): String {
    var s = raw.trim()
      .replace(Regex("\\s+([,.:;!?»])"), "$1")
      .replace(Regex("([«])\\s+"), "$1")
      .replace(Regex("\\s{2,}"), " ")
    if (s.isEmpty()) return s
    val firstChar = s[0]
    if (firstChar.isLetter() && firstChar.isLowerCase()) {
      s = firstChar.uppercaseChar() + s.substring(1)
    }
    val last = s.last()
    if (last.isLetterOrDigit() || last == '»') {
      s += "."
    }
    return s
  }

  /**
   * Анализирует перебивания (наложения речи `pyannote`) и эмоциональный/интонационный тон каждой реплики
   * по энергии сигнала (RMS) и темпу речи:
   * - `hasOverlap = true`, если во время реплики зафиксирована одновременная активность двух разных спикеров
   *   в `diarSlices` (перекрытие >= 180 мс) или мгновенный перехват инициативы у другого спикера (пауза <= 140 мс);
   * - `prosodyTag`: `"intense"` (перебивание на повышенном тоне), `"elevated"` (повышенный тон / эмоция),
   *   `"fast"` (высокий темп речи > 3,2 слов/с), `"calm"` (спокойный тон).
   */
  fun enrichProsodyAndOverlaps(
    utterances: List<Utterance>,
    slices: List<DiarizationSlice>,
    samples: FloatArray,
    sampleRate: Int,
  ): List<Utterance> {
    if (utterances.isEmpty()) return utterances

    // 1. Вычисляем RMS-энергию активных фреймов каждой реплики
    val rmsValues = FloatArray(utterances.size)
    for (i in utterances.indices) {
      val u = utterances[i]
      val fromIdx = ((u.startMs * sampleRate) / 1000L).toInt().coerceIn(0, samples.size)
      val toIdx = ((u.endMs * sampleRate) / 1000L).toInt().coerceIn(fromIdx, samples.size)
      if (toIdx - fromIdx < sampleRate / 10) {
        rmsValues[i] = 0.05f
        continue
      }
      var sumSq = 0.0
      var count = 0
      var pos = fromIdx
      val frame = 320 // 20 мс
      while (pos + frame <= toIdx) {
        var fSq = 0.0
        for (k in pos until pos + frame) {
          val v = samples[k].toDouble()
          fSq += v * v
        }
        val fRms = kotlin.math.sqrt(fSq / frame)
        if (fRms > 0.012) {
          sumSq += fSq
          count += frame
        }
        pos += frame
      }
      rmsValues[i] = if (count > 0) kotlin.math.sqrt(sumSq / count).toFloat() else 0.05f
    }

    val sortedRms = rmsValues.filter { it > 0.01f }.sorted()
    val baselineRms = if (sortedRms.isNotEmpty()) {
      val mid = sortedRms.size / 2
      val median = if (sortedRms.size % 2 == 0) {
        (sortedRms[mid - 1] + sortedRms[mid]) * 0.5f
      } else {
        sortedRms[mid]
      }
      median.coerceAtLeast(0.025f)
    } else {
      0.08f
    }

    // 2. Находим интервалы одновременной речи разных спикеров в сегментах диаризации
    data class OverlapWindow(val startMs: Long, val endMs: Long)
    val overlapWindows = mutableListOf<OverlapWindow>()
    for (i in slices.indices) {
      val a = slices[i]
      for (j in i + 1 until slices.size) {
        val b = slices[j]
        if (a.speakerIndex != b.speakerIndex) {
          val ovStart = max(a.startMs, b.startMs)
          val ovEnd = min(a.endMs, b.endMs)
          if (ovEnd - ovStart >= 180L) {
            overlapWindows += OverlapWindow(ovStart, ovEnd)
          }
        }
      }
    }

    return utterances.mapIndexed { idx, u ->
      val ratio = (rmsValues[idx] / baselineRms).coerceIn(0.45f, 2.5f)
      val prev = if (idx > 0) utterances[idx - 1] else null

      // Проверяем пересечение с окнами одновременной речи двух спикеров
      val hasDiarOverlap = overlapWindows.any { ow ->
        val ov = min(u.endMs, ow.endMs) - max(u.startMs, ow.startMs)
        ov >= 180L
      }
      // Проверяем быстрое перебивание предыдущего собеседника (смена спикера с паузой <= 140 мс)
      val isRapidInterruption = prev != null &&
        prev.speakerId != u.speakerId &&
        (u.startMs - prev.endMs) <= 140L

      val hasOverlap = hasDiarOverlap || isRapidInterruption

      val durSec = ((u.endMs - u.startMs).coerceAtLeast(400L)) / 1000.0f
      val wordCount = if (u.words.isNotEmpty()) u.words.size else u.text.split(Regex("\\s+")).count { it.isNotBlank() }
      val wordsPerSec = wordCount / durSec

      val prosodyTag = when {
        hasOverlap && ratio >= 1.15f -> "intense"
        ratio >= 1.25f || (u.text.endsWith("!") && ratio >= 1.10f) -> "elevated"
        wordsPerSec >= 3.2f && wordCount >= 5 -> "fast"
        else -> "calm"
      }

      u.copy(
        hasOverlap = hasOverlap,
        prosodyTag = prosodyTag,
        energyRatio = ratio,
      )
    }
  }
}
