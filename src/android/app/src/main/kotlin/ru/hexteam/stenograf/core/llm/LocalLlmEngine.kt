package ru.hexteam.stenograf.core.llm

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import ru.hexteam.stenograf.core.export.TranscriptExporter
import ru.hexteam.stenograf.core.model.TranscriptRecord
import ru.hexteam.stenograf.core.model.Utterance
import ru.hexteam.stenograf.core.pipeline.TranscriptAligner
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Локальный движок большой языковой модели (Qwen3-0.6B INT4 на Google AI Edge LiteRT-LM)
 * и нейро-семантического экстрактора тезисов:
 *  - разбивает длинные реплики на отдельные предложения с точными пословными таймкодами (`Utterance.words`);
 *  - очищает разговорные слова-паразиты и сжимает мысли до лаконичных тезисов (до 120–130 символов);
 *  - классифицирует высказывания на главные тезисы, ключевые вопросы/проблемы и принятые решения/поручения с таймкодами;
 *  - синтезирует связное резюме совещания через локальную модель `Qwen3-0.6B` (`/no_think`) полностью офлайн.
 */
class LocalLlmEngine(private val context: Context) {

  private val mutex = Mutex()
  private var engine: Engine? = null

  companion object {
    private const val ASSET_MODEL_PATH = "llm/qwen3-0.6b.litertlm"

    private val STOP_WORDS = setOf(
      "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",
      "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",
      "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
      "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
      "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
      "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем",
      "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
      "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь",
      "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
      "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после",
      "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много",
      "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
      "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю",
      "между", "это", "просто", "вообще", "короче", "кстати", "собственно", "значит", "типа",
      "скажем", "принципе", "говорить", "сказать", "очень", "кажется", "наверное", "ладно",
    )

    private val FILLER_PREFIX_REGEX = Regex(
      "^(?:(?:ну|а+|э+|м+|ох|слушай(?:те)?|смотри(?:те)?|короче(?: говоря)?|в общем(?:-то)?|" +
        "собственно(?: говоря)?|так сказать|скажем так|знаешь|знаете|как бы|в принципе|на самом деле|" +
        "то есть|давай(?:те)? так|по сути)[,\\s—-]+)+",
      RegexOption.IGNORE_CASE,
    )

    private val ACTION_MARKERS = listOf(
      "нужно", "необходимо", "надо", "решили", "договорились", "сделаем", "сделать",
      "подготов", "отправ", "провер", "настро", "внедр", "исправ", "запуст", "обнов",
      "перенес", "согласов", "утверд", "поруч", "задач", "срок", "дедлайн", "следующ",
      "давайте", "предлагаю", "возьму", "займусь", "закроем", "проконтрол", "обязательно",
    )

    private val PROBLEM_MARKERS = listOf(
      "проблем", "ошибк", "сбой", "риск", "сложност", "вопрос", "почему", "зачем",
      "непонятн", "не работ", "тормоз", "падает", "не хват", "огранич", "узкое место",
      "выяснить", "уточнить", "проверить", "беспокоит", "жалоб",
    )

    private val THESIS_MARKERS = listOf(
      "главн", "важн", "ключев", "основн", "итог", "результат", "вывод", "цель",
      "проект", "систем", "модел", "качеств", "скорост", "клиент", "данн", "архитектур",
      "показател", "процент", "бюджет", "план", "этап", "решен", "эффективн", "оптимиз",
    )
  }

  /** Отдельное предложение, извлечённое из реплики с точным таймкодом начала. */
  data class ExtractedSentence(
    val speakerId: String,
    val speakerName: String,
    val startMs: Long,
    val endMs: Long,
    val rawText: String,
    val conciseText: String,
    val thesisScore: Double,
    val isActionItem: Boolean,
    val isQuestionOrProblem: Boolean,
  )

  /** Распаковывает модель из assets при первом обращении и инициализирует LiteRT-LM Engine. */
  private fun ensureEngine(): Engine {
    engine?.let { if (it.isInitialized()) return it }
    val modelFile = File(context.filesDir, "llm/qwen3-0.6b.litertlm")
    if (!modelFile.exists() || modelFile.length() < 10_000_000L) {
      modelFile.parentFile?.mkdirs()
      context.assets.open(ASSET_MODEL_PATH).use { input ->
        FileOutputStream(modelFile).use { output ->
          input.copyTo(output, bufferSize = 1 shl 18)
        }
      }
    }
    val cacheDir = File(context.cacheDir, "litert_cache").apply { mkdirs() }
    val cfg = EngineConfig(
      modelPath = modelFile.absolutePath,
      backend = Backend.CPU(4),
      visionBackend = null,
      audioBackend = null,
      maxNumTokens = 4096,
      cacheDir = cacheDir.absolutePath,
    )
    val eng = Engine(cfg)
    eng.initialize()
    engine = eng
    return eng
  }

  /**
   * Генерирует структурированный протокол совещания по расшифровке:
   * 1) Извлекает из всех реплик отдельные предложения с точными таймкодами (`[MM:SS.d]`),
   *    сжимает длинные высказывания до ёмких тезисов и ранжирует их по семантической значимости (TF-IDF + маркеры решений/вопросов).
   * 2) Передаёт отобранные тезисы в локальную нейросеть `Qwen3-0.6B` (`/no_think`) для синтеза связного резюме
   *    и объединяет его с точными таймкодами ключевых тезисов, вопросов и поручений.
   */
  suspend fun generateMeetingProtocol(
    record: TranscriptRecord,
    onPartial: (String) -> Unit = {},
  ): String = withContext(Dispatchers.IO) {
    mutex.withLock {
      val sentences = extractAndScoreSentences(record)
      val baseAnalyticalProtocol = buildSmartProtocol(record, sentences, llmExecutiveSummary = null)
      onPartial(baseAnalyticalProtocol)

      // Формируем компактный контекст из лучших тезисов с таймкодами для Qwen3-0.6B,
      // чтобы нейросеть синтезировала осмысленное резюме без переполнения окна контекста.
      val topForLlm = sentences
        .sortedByDescending { it.thesisScore + (if (it.isActionItem) 1.5 else 0.0) }
        .take(16)
        .sortedBy { it.startMs }

      val condensedInput = topForLlm.joinToString("\n") { s ->
        "[${TranscriptExporter.formatTimeMs(s.startMs)}] ${s.speakerName}: ${s.conciseText}"
      }

      val systemPrompt =
        "Ты — «Уралан: Нейрографист», интеллектуальный секретарь совещаний. Пиши строго на русском языке, лаконично, деловым стилем без воды. /no_think"
      val userPrompt = buildString {
        appendLine("Ниже приведены ключевые высказывания участников совещания с таймкодами.")
        appendLine("Напиши краткое аналитическое резюме (3–4 предложения): о чём был разговор, какая главная тема обсуждалась и к каким выводам пришли участники. Не копируй реплики дословно, а сформулируй суть своими словами. /no_think")
        appendLine()
        appendLine("Высказывания:")
        appendLine(condensedInput)
      }

      try {
        val eng = ensureEngine()
        val convCfg = ConversationConfig(
          systemInstruction = Contents.of(systemPrompt),
          initialMessages = emptyList(),
          tools = emptyList(),
          samplerConfig = SamplerConfig(20, 0.85, 0.25, 42),
          automaticToolCalling = false,
          channels = emptyList(),
          extraContext = mapOf("enable_thinking" to false),
          loraConfig = null,
          prefillPrefaceOnInit = false,
          maxOutputToken = 512,
          thinkingConfig = ThinkingConfig(false, 0),
        )
        eng.createConversation(convCfg).use { conv ->
          val sb = StringBuilder()
          conv.sendMessageAsync(
            userPrompt,
            mapOf("enable_thinking" to false),
            null,
            null,
            null,
            512,
            ThinkingConfig(false, 0),
            null,
          ).collect { msg: Message ->
            val chunk = msg.contents.contents
              .filterIsInstance<com.google.ai.edge.litertlm.Content.Text>()
              .joinToString("") { it.text }
            if (chunk.isNotEmpty()) {
              sb.append(chunk)
              val cleaned = cleanLlmSummaryText(stripThinkTags(sb.toString()))
              if (cleaned.length >= 25) {
                onPartial(buildSmartProtocol(record, sentences, llmExecutiveSummary = cleaned))
              }
            }
          }
          val finalSummary = cleanLlmSummaryText(stripThinkTags(sb.toString()))
          if (finalSummary.length >= 25) {
            val fullProtocol = buildSmartProtocol(record, sentences, llmExecutiveSummary = finalSummary)
            onPartial(fullProtocol)
            return@withLock fullProtocol
          }
        }
      } catch (_: Throwable) {
        // Если на устройстве/эмуляторе LiteRT-LM недоступен — используем семантический протокол с точными таймкодами
      }

      baseAnalyticalProtocol
    }
  }

  /**
   * Разбивает реплики на отдельные предложения, вычисляет для каждого предложения точный таймкод начала
   * по пословным меткам `Utterance.words`, сжимает длинные фразы до ёмких тезисов и оценивает их важность.
   */
  internal fun extractAndScoreSentences(record: TranscriptRecord): List<ExtractedSentence> {
    if (record.utterances.isEmpty()) return emptyList()

    data class RawCandidate(
      val speakerId: String,
      val speakerName: String,
      val startMs: Long,
      val endMs: Long,
      val rawText: String,
      val conciseText: String,
      val stems: List<String>,
    )

    val candidates = mutableListOf<RawCandidate>()

    for (u in record.utterances) {
      val spkName = record.displayName(u.speakerId)
      val sentenceSpans = splitUtteranceIntoTimedSentences(u)
      for ((sMs, eMs, sentText) in sentenceSpans) {
        val concise = compressToConciseThesis(sentText, maxChars = 128)
        val stems = tokenizeStems(concise)
        // Отбрасываем однословные междометия («Да.», «Угу.», «Алло.», «Понял.»)
        if (concise.length < 14 || stems.size < 2) continue
        candidates += RawCandidate(
          speakerId = u.speakerId,
          speakerName = spkName,
          startMs = sMs,
          endMs = eMs,
          rawText = sentText,
          conciseText = concise,
          stems = stems,
        )
      }
    }

    if (candidates.isEmpty()) {
      // Если все реплики совсем короткие — берём их как есть
      return record.utterances.map { u ->
        ExtractedSentence(
          speakerId = u.speakerId,
          speakerName = record.displayName(u.speakerId),
          startMs = u.startMs,
          endMs = u.endMs,
          rawText = u.text,
          conciseText = compressToConciseThesis(u.text, 120),
          thesisScore = 1.0,
          isActionItem = false,
          isQuestionOrProblem = u.text.contains("?"),
        )
      }
    }

    // Считаем TF-IDF вес ключевых корней по всему диалогу
    val docFreq = HashMap<String, Int>()
    val termFreq = HashMap<String, Int>()
    for (c in candidates) {
      val unique = c.stems.toSet()
      for (st in unique) docFreq[st] = (docFreq[st] ?: 0) + 1
      for (st in c.stems) termFreq[st] = (termFreq[st] ?: 0) + 1
    }

    val totalDocs = candidates.size.toDouble().coerceAtLeast(1.0)
    val result = ArrayList<ExtractedSentence>(candidates.size)

    for ((idx, c) in candidates.withIndex()) {
      val lower = c.rawText.lowercase(Locale.forLanguageTag("ru"))
      var tfidfSum = 0.0
      for (st in c.stems.distinct()) {
        val tf = (termFreq[st] ?: 1).toDouble()
        val df = (docFreq[st] ?: 1).toDouble()
        val idf = ln(1.0 + totalDocs / df)
        tfidfSum += (1.0 + ln(tf)) * idf
      }
      // Нормируем на корень из числа слов, чтобы не завышать оценку гигантских предложений
      val normTfIdf = tfidfSum / kotlin.math.sqrt(c.stems.size.toDouble().coerceAtLeast(2.0))

      val hasAction = ACTION_MARKERS.any { lower.contains(it) }
      val hasProblem = c.rawText.contains("?") || PROBLEM_MARKERS.any { lower.contains(it) }
      val hasThesisKeyword = THESIS_MARKERS.any { lower.contains(it) }
      val hasNumbers = c.rawText.any { it.isDigit() }

      // Оптимальная длина тезиса — от 30 до 120 символов (не обрывок и не простыня)
      val lenBonus = when (c.conciseText.length) {
        in 35..115 -> 1.35
        in 22..130 -> 1.10
        else -> 0.85
      }

      // Небольшой бонус начальным (постановка темы) и финальным (подведение итогов) высказываниям
      val posFrac = idx.toDouble() / candidates.size.coerceAtLeast(1)
      val positionBonus = if (posFrac <= 0.15 || posFrac >= 0.82) 1.15 else 1.0

      val score = normTfIdf * lenBonus * positionBonus +
        (if (hasThesisKeyword) 1.4 else 0.0) +
        (if (hasNumbers) 1.1 else 0.0) +
        (if (hasAction) 0.9 else 0.0)

      result += ExtractedSentence(
        speakerId = c.speakerId,
        speakerName = c.speakerName,
        startMs = c.startMs,
        endMs = c.endMs,
        rawText = c.rawText,
        conciseText = c.conciseText,
        thesisScore = score,
        isActionItem = hasAction,
        isQuestionOrProblem = hasProblem,
      )
    }

    return result
  }

  /**
   * Разбивает реплику на отдельные предложения и определяет точный `startMs` и `endMs` каждого предложения
   * по пословным меткам `utterance.words`.
   */
  private fun splitUtteranceIntoTimedSentences(utterance: Utterance): List<Triple<Long, Long, String>> {
    val words = utterance.words
    if (words.isNotEmpty()) {
      val sentences = mutableListOf<Triple<Long, Long, String>>()
      var curWords = mutableListOf<ru.hexteam.stenograf.core.model.Word>()
      for (w in words) {
        curWords += w
        val endsSent = w.text.endsWith(".") || w.text.endsWith("!") || w.text.endsWith("?")
        // Также делим очень длинные безостановочные фразы (> 22 слов) по точке с запятой, тире или союзам «потому что / поэтому / при этом»
        val isLongClauseBreak = curWords.size >= 18 && (w.text.endsWith(",") || w.text.endsWith(";") || w.text == "—")
        if (endsSent || isLongClauseBreak) {
          val text = TranscriptAligner.normalizeRussianPunctuation(curWords.joinToString(" ") { it.text })
          sentences += Triple(curWords.first().startMs, curWords.last().endMs, text)
          curWords = mutableListOf()
        }
      }
      if (curWords.isNotEmpty()) {
        val text = TranscriptAligner.normalizeRussianPunctuation(curWords.joinToString(" ") { it.text })
        sentences += Triple(curWords.first().startMs, curWords.last().endMs, text)
      }
      return sentences
    }

    // Резервный вариант, если пословные метки отсутствуют: делим по знакам конца предложения и интерполируем время
    val parts = utterance.text
      .split(Regex("(?<=[.!?])\\s+"))
      .map { it.trim() }
      .filter { it.isNotEmpty() }
    if (parts.isEmpty()) return listOf(Triple(utterance.startMs, utterance.endMs, utterance.text))
    val totalChars = parts.sumOf { it.length }.coerceAtLeast(1)
    val totalDur = (utterance.endMs - utterance.startMs).coerceAtLeast(100L)
    var accChars = 0
    return parts.map { part ->
      val sMs = utterance.startMs + (totalDur * accChars) / totalChars
      accChars += part.length
      val eMs = utterance.startMs + (totalDur * accChars) / totalChars
      Triple(sMs, eMs, TranscriptAligner.normalizeRussianPunctuation(part))
    }
  }

  /**
   * Превращает разговорное предложение в лаконичный тезис (не более [maxChars] символов):
   * удаляет вводные слова-паразиты в начале и при необходимости оставляет главную смысловую часть до запятой/тире.
   */
  internal fun compressToConciseThesis(rawSentence: String, maxChars: Int = 128): String {
    var cleaned = rawSentence.trim()
      .replace(FILLER_PREFIX_REGEX, "")
      .replace(Regex("\\s{2,}"), " ")
      .trim()

    if (cleaned.isEmpty()) cleaned = rawSentence.trim()

    // Убираем повторные вводные конструкции внутри фразы
    cleaned = cleaned
      .replace(Regex("(?i)\\b(короче говоря|грубо говоря|скажем так|так сказать|как говорится),?\\s*"), "")
      .replace(Regex("\\s+([,.:;!?])"), "$1")
      .trim()

    if (cleaned.length > maxChars) {
      // Ищем естественную границу смысловой клаузы (точка с запятой, тире, запятая) в диапазоне [55, maxChars]
      val window = cleaned.substring(0, maxChars)
      val clauseBreak = listOf("; ", " — ", ", а ", ", но ", ", чтобы ", ", поэтому ", ", так как ", ", где ", ", когда ", ", ")
        .map { sep -> window.lastIndexOf(sep) }
        .filter { idx -> idx >= 50 }
        .maxOrNull()

      cleaned = if (clauseBreak != null) {
        window.substring(0, clauseBreak).trimEnd(',', ';', '—', '-', ' ') + "."
      } else {
        val lastSpace = window.lastIndexOf(' ').takeIf { it >= 50 } ?: (maxChars - 1)
        window.substring(0, lastSpace).trimEnd(',', ';', '—', '-', '.') + "…"
      }
    }

    return TranscriptAligner.normalizeRussianPunctuation(cleaned)
  }

  private fun tokenizeStems(text: String): List<String> {
    return Regex("[а-яёa-z0-9]{3,}", RegexOption.IGNORE_CASE)
      .findAll(text.lowercase(Locale.forLanguageTag("ru")))
      .map { it.value }
      .filter { it !in STOP_WORDS }
      .map { word ->
        // Простая русская стеммизация (отсечение окончаний до основы >= 4 букв) для сопоставления однокоренных слов
        if (word.length >= 6) word.substring(0, word.length - 2)
        else if (word.length == 5) word.substring(0, 4)
        else word
      }
      .toList()
  }

  /**
   * Собирает чисто отформатированный протокол совещания с краткими тезисами и точными таймкодами.
   */
  private fun buildSmartProtocol(
    record: TranscriptRecord,
    sentences: List<ExtractedSentence>,
    llmExecutiveSummary: String?,
  ): String = buildString {
    val topKeywords = extractTopTopicWords(sentences, limit = 5)
    val topicLine = if (topKeywords.isNotEmpty()) {
      topKeywords.joinToString(", ")
    } else {
      "обсуждение рабочих вопросов"
    }

    appendLine("### Краткое резюме совещания")
    if (!llmExecutiveSummary.isNullOrBlank()) {
      appendLine(llmExecutiveSummary.trim())
    } else {
      val leadThesis = sentences.maxByOrNull { it.thesisScore }?.conciseText
      appendLine(
        "Запись длительностью **${TranscriptExporter.formatTimeMs(record.durationMs)}**, " +
          "участников: **${record.speakers.size}** (${record.speakers.joinToString { record.displayName(it) }}), " +
          "всего реплик: **${record.utterances.size}**."
      )
      appendLine("Ключевые темы разговора: **$topicLine**.")
      if (!leadThesis.isNullOrBlank()) {
        appendLine("Центральный тезис: *$leadThesis*")
      }
    }

    appendLine()
    appendLine("### Главные тезисы (с таймкодами)")
    // Выбираем до 6 наиболее значимых непересекающихся по смыслу тезисов и сортируем их хронологически
    val selectedTheses = selectDiverseSentences(
      pool = sentences.sortedByDescending { it.thesisScore },
      maxCount = 6,
    ).sortedBy { it.startMs }

    if (selectedTheses.isEmpty()) {
      appendLine("- Недостаточно данных для выделения тезисов.")
    } else {
      for (th in selectedTheses) {
        appendLine(
          "- `[${TranscriptExporter.formatTimeMs(th.startMs)}]` **${th.speakerName}**: ${th.conciseText}"
        )
      }
    }

    // Раздел решений и поручений
    val actions = selectDiverseSentences(
      pool = sentences.filter { it.isActionItem }.sortedByDescending { it.thesisScore },
      maxCount = 4,
    ).sortedBy { it.startMs }

    if (actions.isNotEmpty()) {
      appendLine()
      appendLine("### Принятые решения и поручения")
      for (act in actions) {
        appendLine(
          "- `[${TranscriptExporter.formatTimeMs(act.startMs)}]` **${act.speakerName}**: ${act.conciseText}"
        )
      }
    }

    // Раздел ключевых вопросов и проблем
    val questions = selectDiverseSentences(
      pool = sentences
        .filter { it.isQuestionOrProblem && it !in actions }
        .sortedByDescending { it.thesisScore },
      maxCount = 4,
    ).sortedBy { it.startMs }

    if (questions.isNotEmpty()) {
      appendLine()
      appendLine("### Ключевые вопросы и обсуждаемые проблемы")
      for (q in questions) {
        appendLine(
          "- `[${TranscriptExporter.formatTimeMs(q.startMs)}]` **${q.speakerName}**: ${q.conciseText}"
        )
      }
    }

    // Раздел динамики дискуссии: перебивания, спорные моменты и повышенный эмоциональный тон
    val hotMoments = record.utterances
      .filter { it.hasOverlap || it.prosodyTag == "intense" || it.prosodyTag == "elevated" }
      .sortedByDescending { (if (it.hasOverlap) 1.5f else 0f) + it.energyRatio }
      .take(4)
      .sortedBy { it.startMs }

    if (hotMoments.isNotEmpty()) {
      appendLine()
      appendLine("### Динамика дискуссии: перебивания и эмоциональный тон")
      for (hm in hotMoments) {
        val label = when {
          hm.prosodyTag == "intense" -> "Спор / Перебивание на повышенном тоне"
          hm.hasOverlap -> "Перебивание / Одновременная речь"
          else -> "Повышенный эмоциональный тон"
        }
        val brief = compressToConciseThesis(hm.text, maxChars = 105)
        appendLine(
          "- `[${TranscriptExporter.formatTimeMs(hm.startMs)}]` **${record.displayName(hm.speakerId)}** (*$label*): $brief"
        )
      }
    }

    appendLine()
    appendLine("### Активность и вклад участников")
    for (spk in record.speakers) {
      val turns = record.utterances.filter { it.speakerId == spk }
      val durSec = turns.sumOf { (it.endMs - it.startMs).coerceAtLeast(0) } / 1000.0
      val sharePct = if (record.durationMs > 0) {
        ((turns.sumOf { (it.endMs - it.startMs).coerceAtLeast(0) } * 100.0) / record.durationMs).toInt().coerceIn(1, 100)
      } else 0
      val bestSpkThesis = sentences
        .filter { it.speakerId == spk }
        .maxByOrNull { it.thesisScore }
        ?.let { " — `[${TranscriptExporter.formatTimeMs(it.startMs)}]` ${it.conciseText}" }
        ?: ""
      appendLine(
        "- **${record.displayName(spk)}**: реплик — ${turns.size}, эфирное время — ${"%.1f".format(durSec)} с ($sharePct%)$bestSpkThesis"
      )
    }
  }

  /**
   * Отбирает разнообразные по содержанию предложения (без дублирования одних и тех же фраз).
   */
  private fun selectDiverseSentences(
    pool: List<ExtractedSentence>,
    maxCount: Int,
  ): List<ExtractedSentence> {
    val picked = mutableListOf<ExtractedSentence>()
    for (cand in pool) {
      if (picked.size >= maxCount) break
      val candStems = tokenizeStems(cand.conciseText).toSet()
      val isDuplicate = picked.any { existing ->
        val exStems = tokenizeStems(existing.conciseText).toSet()
        if (candStems.isEmpty() || exStems.isEmpty()) {
          existing.conciseText.equals(cand.conciseText, ignoreCase = true)
        } else {
          val inter = candStems.intersect(exStems).size.toDouble()
          val union = max(1, min(candStems.size, exStems.size)).toDouble()
          (inter / union) >= 0.68 || kotlin.math.abs(existing.startMs - cand.startMs) < 1500L
        }
      }
      if (!isDuplicate) {
        picked += cand
      }
    }
    return picked
  }

  private fun extractTopTopicWords(sentences: List<ExtractedSentence>, limit: Int): List<String> {
    val counts = LinkedHashMap<String, Int>()
    val surfaceForm = HashMap<String, String>()
    for (s in sentences) {
      val words = Regex("[а-яёa-z]{4,}", RegexOption.IGNORE_CASE)
        .findAll(s.conciseText.lowercase(Locale.forLanguageTag("ru")))
        .map { it.value }
        .filter { it !in STOP_WORDS }
      for (w in words) {
        val stem = if (w.length >= 6) w.substring(0, w.length - 2) else w
        counts[stem] = (counts[stem] ?: 0) + 1
        val prev = surfaceForm[stem]
        if (prev == null || w.length < prev.length) {
          surfaceForm[stem] = w
        }
      }
    }
    return counts.entries
      .sortedByDescending { it.value }
      .take(limit)
      .mapNotNull { surfaceForm[it.key] }
  }

  private fun cleanLlmSummaryText(raw: String): String {
    return raw
      .replace(Regex("^#{1,6}\\s*Краткое резюме.*$", RegexOption.MULTILINE), "")
      .replace(Regex("/no_think"), "")
      .trim()
  }

  /**
   * Интеллектуальная постобработка реплики через локальную LLM (восстановление пунктуации,
   * устранение оговорок при сохранении исходных слов).
   */
  suspend fun refineUtteranceText(utterance: Utterance, contextBefore: String?): String =
    withContext(Dispatchers.IO) {
      mutex.withLock {
        try {
          val eng = ensureEngine()
          val convCfg = ConversationConfig(
            systemInstruction = Contents.of(
              "Ты — редактор русской стенограммы. Исправь только пунктуацию, регистр и очевидные опечатки распознавания речи, не меняя смысл и не добавляя новых слов. Верни только исправленную реплику без пояснений. /no_think"
            ),
            initialMessages = emptyList(),
            tools = emptyList(),
            samplerConfig = SamplerConfig(10, 0.85, 0.15, 42),
            automaticToolCalling = false,
            channels = emptyList(),
            extraContext = mapOf("enable_thinking" to false),
            loraConfig = null,
            prefillPrefaceOnInit = false,
            maxOutputToken = 192,
            thinkingConfig = ThinkingConfig(false, 0),
          )
          eng.createConversation(convCfg).use { conv ->
            val prompt = buildString {
              if (!contextBefore.isNullOrBlank()) {
                appendLine("Предыдущая реплика: $contextBefore")
              }
              append("Реплика для коррекции: ${utterance.text} /no_think")
            }
            val sb = StringBuilder()
            conv.sendMessageAsync(
              prompt,
              mapOf("enable_thinking" to false),
              null,
              null,
              null,
              192,
              ThinkingConfig(false, 0),
              null,
            ).collect { msg ->
              val chunk = msg.contents.contents
                .filterIsInstance<com.google.ai.edge.litertlm.Content.Text>()
                .joinToString("") { it.text }
              sb.append(chunk)
            }
            val cleaned = stripThinkTags(sb.toString())
              .removePrefix("Реплика для коррекции:")
              .replace("/no_think", "")
              .trim()
              .trim('"')
            if (cleaned.isNotBlank() && cleaned.length <= utterance.text.length * 2 + 20) {
              return@withLock TranscriptAligner.normalizeRussianPunctuation(cleaned)
            }
          }
        } catch (_: Throwable) {
        }
        TranscriptAligner.normalizeRussianPunctuation(utterance.text)
      }
    }

  private fun stripThinkTags(raw: String): String =
    raw.replace(Regex("<think>[\\s\\S]*?</think>"), "")
      .replace(Regex("<think>[\\s\\S]*$"), "")
      .trim()

  fun close() {
    try {
      engine?.close()
    } catch (_: Throwable) {
    }
    engine = null
  }
}
