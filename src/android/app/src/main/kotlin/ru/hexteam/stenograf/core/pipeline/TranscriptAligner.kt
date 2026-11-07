package ru.hexteam.stenograf.core.pipeline

import ru.hexteam.stenograf.core.model.Utterance
import ru.hexteam.stenograf.core.model.Word
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

/** Элементарный фразовый сегмент для точного извлечения голосового эмбеддинга и высоты тона (F0). */
data class PhraseSegment(
  val startMs: Long,
  val endMs: Long,
  val text: String,
  val hasTurnDash: Boolean,
)

/**
 * Алгоритм объединения субсловных токенов ASR и сегментов диаризации в структурированные реплики:
 * 1) Собирает BPE-токены (с префиксом пробела ` ` или `▁` и знаками препинания) в целые слова с точными `startMs` и `endMs`,
 *    отсеивая нелексические звуки колебания (`э-э-э`, `а-а-а`, `хмм`) и фиксируя диалоговые тире `—` как маркеры смены реплики.
 * 2) Группирует слова в чистые фразовые сегменты ([buildPhraseSegments]) по диалоговым тире, границам предложений и паузам.
 * 3) Каждому слову сопоставляет говорящего по максимальному перекрытию с сегментами диаризации (или ближайшему по времени).
 * 4) Сглаживает одиночные ложные переключения спикера внутри быстрой фразы (медианный фильтр при паузе < 300 мс).
 * 5) Группирует слова в реплики по смене говорящего или длинной паузе (> 1500 мс) и приводит ID спикеров к порядку появления (`speaker_1`, `speaker_2`, ...).
 */
object TranscriptAligner {

  private val PUNCT_ONLY = setOf(".", ",", "?", "!", ":", ";", "—", "-", "»", "«", "...")
  private val REAL_SHORT_WORDS = setOf(
    "а", "и", "о", "у", "в", "к", "с", "я", "ага", "угу", "да", "ну",
    "не", "ни", "но", "же", "ли", "бы", "то", "на", "по", "за", "из",
    "от", "до", "со", "во", "ко", "об", "ох", "ах", "ой", "эй", "эм"
  )
  private val VOCAL_NOISE_EXACT = setOf(
    "э", "ээ", "эээ", "э-э", "э-э-э", "а-а", "а-а-а", "аа", "ааа",
    "мм", "ммм", "хм", "хмм", "м-м"
  )
  private val VOCAL_NOISE_HYPHEN = Regex("^[эамх]+(-[эамх]+)+$", RegexOption.IGNORE_CASE)

  /** Проверяет, является ли токен протяжным нелексическим мычанием (`э-э-э`, `а-а-а`, `хмм`). */
  fun isVocalHesitation(wordText: String): Boolean {
    val clean = wordText.lowercase().replace(Regex("[^\\p{L}\\p{N}-]"), "")
    if (clean.isEmpty()) return true
    if (clean in REAL_SHORT_WORDS) return false
    if (clean in VOCAL_NOISE_EXACT) return true
    return VOCAL_NOISE_HYPHEN.matches(clean)
  }

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
    var pendingTurnDash = false

    fun flushWord(nextWordStartMs: Long) {
      if (currentText.isEmpty()) return
      val raw = currentText.toString().trim()
      if (raw.isNotEmpty() && raw !in PUNCT_ONLY && !isVocalHesitation(raw)) {
        val s = wordStart.coerceAtLeast(chunkStartMs)
        val estimatedEnd = max(s + 80L, min(nextWordStartMs, lastTokenTime + 240L))
        val e = estimatedEnd.coerceIn(s + 40L, max(s + 40L, chunkEndMs))
        words += Word(text = raw, startMs = s, endMs = e, turnBreakBefore = pendingTurnDash)
        pendingTurnDash = false
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
      if (trimmed == "—") {
        // Диалоговое тире от GigaAM v3 фиксирует смену реплики для следующего слова, не приклеиваясь к предыдущему
        flushWord(tMs)
        pendingTurnDash = true
        continue
      }
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

  /**
   * Разбивает поток распознанных слов на элементарные фразовые сегменты по диалоговым тире (`—`),
   * знакам конца предложения (`.`, `?`, `!`, `…`) и паузам (>= 280 мс), поглощая слишком короткие
   * одиночные слова-сироты (< 550 мс) без диалогового тире, чтобы каждый сегмент содержал чистый голос одного спикера.
   */
  fun buildPhraseSegments(words: List<Word>): List<PhraseSegment> {
    if (words.isEmpty()) return emptyList()
    val rawPhrases = mutableListOf<PhraseSegment>()
    var cur = mutableListOf<Word>()

    fun flushCur() {
      if (cur.isEmpty()) return
      rawPhrases += PhraseSegment(
        startMs = cur.first().startMs,
        endMs = cur.last().endMs,
        text = cur.joinToString(" ") { it.text },
        hasTurnDash = cur.first().turnBreakBefore,
      )
      cur = mutableListOf()
    }

    for (w in words) {
      if (cur.isNotEmpty()) {
        val prevTxt = cur.last().text
        val endsSent = prevTxt.endsWith(".") || prevTxt.endsWith("?") || prevTxt.endsWith("!") || prevTxt.endsWith("…")
        val gap = w.startMs - cur.last().endMs
        val spanDur = w.endMs - cur.first().startMs
        val curDur = cur.last().endMs - cur.first().startMs
        val shouldSplit = (w.turnBreakBefore && curDur >= 350L) ||
          (endsSent && gap >= 80L && curDur >= 650L) ||
          (gap >= 280L && curDur >= 550L) ||
          spanDur >= 4200L
        if (shouldSplit) {
          flushCur()
        }
      }
      cur += w
    }
    flushCur()

    val merged = mutableListOf<PhraseSegment>()
    for (p in rawPhrases) {
      val dur = p.endMs - p.startMs
      val last = merged.lastOrNull()
      if (last != null && dur < 550L && !p.hasTurnDash && (p.startMs - last.endMs) < 320L) {
        merged[merged.lastIndex] = last.copy(
          endMs = p.endMs,
          text = "${last.text} ${p.text}",
        )
      } else {
        merged += p
      }
    }
    return merged
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

    // Устойчивое определение участников: убираем одиночный «скачок» спикера на 1 короткое слово
    // внутри непрерывной фразы, если вокруг него один и тот же говорящий и нет паузы.
    if (rawAssigned.size >= 3) {
      for (i in 1 until rawAssigned.lastIndex) {
        val (prevW, prevSpk) = rawAssigned[i - 1]
        val (curW, curSpk) = rawAssigned[i]
        val (nextW, nextSpk) = rawAssigned[i + 1]
        val gapBefore = curW.startMs - prevW.endMs
        val gapAfter = nextW.startMs - curW.endMs
        val wordDur = curW.endMs - curW.startMs
        if (prevSpk == nextSpk && curSpk != prevSpk && gapBefore < 300L && gapAfter < 300L && wordDur < 450L) {
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
   * Канонизирует идентификаторы говорящих в порядке первого появления: `speaker_1`, `speaker_2`, ...
   */
  fun buildUtterances(
    assignedWords: List<Pair<Word, Int>>,
    maxPauseWithinTurnMs: Long = 1500L,
  ): List<Utterance> {
    if (assignedWords.isEmpty()) return emptyList()

    // Перенумерация спикеров по порядку первого появления в диалоге (speaker_1, speaker_2, ...)
    val speakerOrder = LinkedHashMap<Int, String>()
    fun canonicalSpeaker(rawIdx: Int): String =
      speakerOrder.getOrPut(rawIdx) { "speaker_${speakerOrder.size + 1}" }

    val utterances = mutableListOf<Utterance>()
    var currentWords = mutableListOf<Word>()
    var currentRawSpeaker = assignedWords.first().second

    fun flushTurn() {
      if (currentWords.isEmpty()) return
      val text = normalizeRussianPunctuation(currentWords.joinToString(" ") { it.text })
      if (text.isNotBlank()) {
        val spkId = canonicalSpeaker(currentRawSpeaker)
        utterances += Utterance(
          id = utterances.size + 1,
          speakerId = spkId,
          startMs = currentWords.first().startMs,
          endMs = currentWords.last().endMs,
          text = text,
          words = currentWords.toList(),
        )
      }
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
          flushTurn()
          currentRawSpeaker = spk
        }
        currentWords += word
      }
    }
    flushTurn()
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
}
