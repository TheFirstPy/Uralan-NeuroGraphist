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
import kotlin.math.sqrt

/**
 * Локальный движок большой языковой модели (Qwen3-0.6B INT4 на Google AI Edge LiteRT-LM)
 * и глубокого семантического анализатора стенограмм:
 *  - выделяет из реплик законченные по смыслу предложения с точными пословными таймкодами (`Utterance.words`),
 *    не разрывая фразы посередине на запятых;
 *  - очищает устную речь от вводных слов-паразитов и повторов, сохраняя полную смысловую структуру высказывания;
 *  - связывает прозвучавшие вопросы с ответами собеседников в следующих репликах;
 *  - строит хронологическую повестку встречи по смысловым этапам, позиции каждого спикера, главные тезисы и поручения;
 *  - синтезирует аналитическое резюме совещания через локальную модель `Qwen3-0.6B` полностью офлайн.
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
      "алло", "здравствуйте", "привет", "добрый", "день", "спасибо", "пожалуйста", "понятно",
    )

    private val FILLER_PREFIX_REGEX = Regex(
      "^(?:(?:ну|а+|э+|м+|ох|угу|ага|слушай(?:те)?|смотри(?:те)?|короче(?: говоря)?|в общем(?:-то)?|" +
        "собственно(?: говоря)?|так сказать|скажем так|знаешь|знаете|как бы|в принципе|на самом деле|" +
        "то есть|давай(?:те)? так|по сути)[,\\s—-]+)+",
      RegexOption.IGNORE_CASE,
    )

    private val ACTION_MARKERS = listOf(
      "нужно", "необходимо", "надо", "решили", "договорились", "сделаем", "сделать",
      "подготов", "отправ", "провер", "настро", "внедр", "исправ", "запуст", "обнов",
      "перенес", "согласов", "утверд", "поруч", "задач", "срок", "дедлайн", "следующ",
      "давайте", "предлагаю", "возьму", "займусь", "закроем", "проконтрол", "обязательно",
      "запиш", "оформ", "пришл", "свяж", "провед",
    )

    private val PROBLEM_MARKERS = listOf(
      "проблем", "ошибк", "сбой", "риск", "сложност", "вопрос", "почему", "зачем",
      "непонятн", "не работ", "тормоз", "падает", "не хват", "огранич", "узкое место",
      "выяснить", "уточнить", "проверить", "беспокоит", "жалоб", "как именно", "каким образом",
    )

    private val THESIS_MARKERS = listOf(
      "главн", "важн", "ключев", "основн", "итог", "результат", "вывод", "цель",
      "проект", "систем", "модел", "качеств", "скорост", "клиент", "данн", "архитектур",
      "показател", "процент", "бюджет", "план", "этап", "решен", "эффективн", "оптимиз",
      "приложени", "сервис", "алгоритм", "разработ", "тестир", "конвейер", "нейросет",
    )
  }

  /** Отдельное предложение, извлечённое из реплики с точным таймкодом начала. */
  data class ExtractedSentence(
    val utteranceIndex: Int,
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
    if (!modelFile.exists() || modelFile.length() < 100_000_000L) {
      modelFile.parentFile?.mkdirs()
      val tmpFile = File(modelFile.parentFile, "qwen3-0.6b.litertlm.tmp")
      context.assets.open(ASSET_MODEL_PATH).use { input ->
        FileOutputStream(tmpFile).use { output ->
          input.copyTo(output, bufferSize = 1 shl 18)
        }
      }
      if (!tmpFile.renameTo(modelFile)) {
        tmpFile.copyTo(modelFile, overwrite = true)
        tmpFile.delete()
      }
    }
    val cacheDir = File(context.cacheDir, "litert_cache").apply { mkdirs() }
    val cfg = EngineConfig(
      modelPath = modelFile.absolutePath,
      backend = Backend.CPU(4),
      visionBackend = null,
      audioBackend = null,
      maxNumTokens = 2048,
      cacheDir = cacheDir.absolutePath,
    )
    val eng = Engine(cfg)
    eng.initialize()
    engine = eng
    return eng
  }

  /**
   * Генерирует подробный, качественно структурированный протокол совещания по расшифровке:
   * 1) Извлекает из всех реплик полные по смыслу предложения с точными таймкодами (`[MM:SS.d]`),
   *    очищает устные слова-паразиты и ранжирует высказывания по семантической значимости (TF-IDF + структурные маркеры).
   * 2) Запускает локальную нейросеть `Qwen3-0.6B` (`LiteRT-LM`) для синтеза связного аналитического резюме и выводов
   *    с потоковым отображением генерации в UI, после чего объединяет результат с 6-раздельным аналитическим протоколом.
   */
  suspend fun generateMeetingProtocol(
    record: TranscriptRecord,
    onPartial: (String) -> Unit = {},
  ): String = withContext(Dispatchers.IO) {
    mutex.withLock {
      val sentences = extractAndScoreSentences(record)

      // Отбираем до 14 ключевых содержательных высказываний в хронологическом порядке для контекста Qwen3-0.6B
      val topForLlm = sentences
        .sortedByDescending { it.thesisScore + (if (it.isActionItem) 1.2 else 0.0) }
        .take(14)
        .sortedBy { it.startMs }

      val condensedInput = topForLlm.joinToString("\n") { s ->
        "[${TranscriptExporter.formatTimeMs(s.startMs)}] ${s.speakerName}: ${s.conciseText}"
      }

      val systemPrompt =
        "Ты — «Уралан: Нейрографист», профессиональный аналитик и секретарь деловых встреч. Пиши строго на русском языке, грамотным деловым стилем. /no_think"
      val userPrompt = buildString {
        appendLine("Проанализируй выдержки из стенограммы встречи и напиши содержательное аналитическое резюме в 4–5 предложениях:")
        appendLine("1) Главная тема и цель разговора;")
        appendLine("2) Какие ключевые аргументы, предложения или проблемы озвучили участники;")
        appendLine("3) К какому общему итогу или решению пришли в конце обсуждения.")
        appendLine("Формулируй мысли связно, своими словами, без дословного копирования реплик и без вводных фраз вроде «В данном разговоре». /no_think")
        appendLine()
        appendLine("Стенограмма ключевых моментов:")
        appendLine(condensedInput)
      }

      var llmSummaryText: String? = null
      try {
        val eng = ensureEngine()
        llmSummaryText = runLlmSummaryGeneration(eng, systemPrompt, userPrompt) { streamingText ->
          onPartial(buildSmartProtocol(record, sentences, llmExecutiveSummary = streamingText))
        }
      } catch (_: Throwable) {
        // Если первая конфигурация ConversationConfig не поддерживается на конкретном CPU/прошивке — пробуем базовую
        try {
          val eng = ensureEngine()
          llmSummaryText = runLlmSummaryGenerationFallback(eng, systemPrompt, userPrompt) { streamingText ->
            onPartial(buildSmartProtocol(record, sentences, llmExecutiveSummary = streamingText))
          }
        } catch (_: Throwable) {
        }
      }

      val finalProtocol = buildSmartProtocol(record, sentences, llmExecutiveSummary = llmSummaryText)
      onPartial(finalProtocol)
      finalProtocol
    }
  }

  private suspend fun runLlmSummaryGeneration(
    eng: Engine,
    systemPrompt: String,
    userPrompt: String,
    onStreamingUpdate: (String) -> Unit,
  ): String? {
    val convCfg = ConversationConfig(
      systemInstruction = Contents.of(systemPrompt),
      initialMessages = emptyList(),
      tools = emptyList(),
      samplerConfig = SamplerConfig(25, 0.85, 0.25, 42),
      automaticToolCalling = false,
      channels = emptyList(),
      extraContext = mapOf("enable_thinking" to false),
      loraConfig = null,
      prefillPrefaceOnInit = false,
      maxOutputToken = 512,
      thinkingConfig = ThinkingConfig(false),
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
        ThinkingConfig(false),
        null,
      ).collect { msg: Message ->
        val chunk = msg.contents.contents
          .filterIsInstance<com.google.ai.edge.litertlm.Content.Text>()
          .joinToString("") { it.text }
        if (chunk.isNotEmpty()) {
          sb.append(chunk)
          val cleaned = cleanLlmSummaryText(stripThinkTags(sb.toString()))
          if (cleaned.length >= 20) {
            onStreamingUpdate(cleaned)
          }
        }
      }
      val finalSummary = cleanLlmSummaryText(stripThinkTags(sb.toString()))
      return finalSummary.takeIf { it.length >= 25 }
    }
  }

  private suspend fun runLlmSummaryGenerationFallback(
    eng: Engine,
    systemPrompt: String,
    userPrompt: String,
    onStreamingUpdate: (String) -> Unit,
  ): String? {
    val convCfg = ConversationConfig(
      Contents.of(systemPrompt),
      emptyList(),
      emptyList(),
      SamplerConfig(25, 0.85, 0.25, 42),
    )
    eng.createConversation(convCfg).use { conv ->
      val sb = StringBuilder()
      conv.sendMessageAsync(
        userPrompt,
        emptyMap(),
        null,
        null,
        null,
        512,
        null,
        null,
      ).collect { msg: Message ->
        val chunk = msg.contents.contents
          .filterIsInstance<com.google.ai.edge.litertlm.Content.Text>()
          .joinToString("") { it.text }
        if (chunk.isNotEmpty()) {
          sb.append(chunk)
          val cleaned = cleanLlmSummaryText(stripThinkTags(sb.toString()))
          if (cleaned.length >= 20) {
            onStreamingUpdate(cleaned)
          }
        }
      }
      val finalSummary = cleanLlmSummaryText(stripThinkTags(sb.toString()))
      return finalSummary.takeIf { it.length >= 25 }
    }
  }

  /**
   * Разбивает реплики на отдельные законченные предложения, вычисляет для каждого предложения точный таймкод начала
   * по пословным меткам `Utterance.words`, очищает устные паразиты и оценивает информационную значимость.
   */
  internal fun extractAndScoreSentences(record: TranscriptRecord): List<ExtractedSentence> {
    if (record.utterances.isEmpty()) return emptyList()

    data class RawCandidate(
      val utteranceIndex: Int,
      val speakerId: String,
      val speakerName: String,
      val startMs: Long,
      val endMs: Long,
      val rawText: String,
      val conciseText: String,
      val stems: List<String>,
    )

    val candidates = mutableListOf<RawCandidate>()

    for ((uIdx, u) in record.utterances.withIndex()) {
      val spkName = record.displayName(u.speakerId)
      val sentenceSpans = splitUtteranceIntoTimedSentences(u)
      for ((sMs, eMs, sentText) in sentenceSpans) {
        val concise = compressToConciseThesis(sentText, maxChars = 210)
        val stems = tokenizeStems(concise)
        // Отбрасываем пустые приветствия и одиночные междометия («Да.», «Угу.», «Алло.», «Понял.»)
        if (concise.length < 18 || stems.size < 2) continue
        candidates += RawCandidate(
          utteranceIndex = uIdx,
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
      return record.utterances.mapIndexed { uIdx, u ->
        ExtractedSentence(
          utteranceIndex = uIdx,
          speakerId = u.speakerId,
          speakerName = record.displayName(u.speakerId),
          startMs = u.startMs,
          endMs = u.endMs,
          rawText = u.text,
          conciseText = compressToConciseThesis(u.text, 200),
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
      // Нормируем на корень из числа значимых слов, чтобы ценить информативную плотность мысли
      val normTfIdf = tfidfSum / sqrt(c.stems.size.toDouble().coerceAtLeast(2.0))

      val hasAction = ACTION_MARKERS.any { lower.contains(it) }
      val hasProblem = c.rawText.contains("?") || PROBLEM_MARKERS.any { lower.contains(it) }
      val hasThesisKeyword = THESIS_MARKERS.any { lower.contains(it) }
      val hasNumbers = c.rawText.any { it.isDigit() }

      // Предпочитаем полные, содержательные предложения (45..195 символов)
      val lenBonus = when (c.conciseText.length) {
        in 45..195 -> 1.40
        in 28..210 -> 1.15
        else -> 0.80
      }

      val posFrac = idx.toDouble() / candidates.size.coerceAtLeast(1)
      val positionBonus = if (posFrac <= 0.16 || posFrac >= 0.80) 1.15 else 1.0

      val score = normTfIdf * lenBonus * positionBonus +
        (if (hasThesisKeyword) 1.5 else 0.0) +
        (if (hasNumbers) 1.2 else 0.0) +
        (if (hasAction) 1.0 else 0.0)

      result += ExtractedSentence(
        utteranceIndex = c.utteranceIndex,
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
   * Разбивает реплику на отдельные законченные предложения и определяет точный `startMs` и `endMs`
   * каждого предложения по пословным меткам `utterance.words`.
   * Никогда не разрывает предложение на обычных запятых, чтобы сохранять целостность мысли.
   */
  private fun splitUtteranceIntoTimedSentences(utterance: Utterance): List<Triple<Long, Long, String>> {
    val words = utterance.words
    if (words.isNotEmpty()) {
      val sentences = mutableListOf<Triple<Long, Long, String>>()
      var curWords = mutableListOf<ru.hexteam.stenograf.core.model.Word>()
      for (w in words) {
        curWords += w
        val endsSent = w.text.endsWith(".") || w.text.endsWith("!") || w.text.endsWith("?") || w.text.endsWith("…")
        // Делим сверхдлинные фразы без точек только на точке с запятой (;), если уже накоплено >= 24 слов
        val isSemicolonBreak = curWords.size >= 24 && w.text.endsWith(";")
        if (endsSent || isSemicolonBreak) {
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

    val parts = utterance.text
      .split(Regex("(?<=[.!?…])\\s+"))
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
   * Очищает разговорное предложение от устных слов-паразитов и тавтологий, сохраняя полную
   * грамматическую и смысловую конструкцию (до [maxChars] символов, без обрезания на случайных запятых).
   */
  internal fun compressToConciseThesis(rawSentence: String, maxChars: Int = 210): String {
    var cleaned = rawSentence.trim()
      .replace(FILLER_PREFIX_REGEX, "")
      .replace(Regex("\\s{2,}"), " ")
      .trim()

    if (cleaned.isEmpty()) cleaned = rawSentence.trim()

    // Убираем разговорные слова-вставки и заикания (повторы одного и того же слова подряд)
    cleaned = cleaned
      .replace(Regex("(?i)\\b(короче говоря|грубо говоря|скажем так|так сказать|как говорится|собственно говоря),?\\s*"), "")
      .replace(Regex("(?i)\\b([а-яё]{2,})\\s+\\1\\b"), "$1")
      .replace(Regex("\\s+([,.:;!?])"), "$1")
      .trim()

    if (cleaned.length > maxChars) {
      val window = cleaned.substring(0, maxChars)
      val majorClauseBreak = listOf("; ", " — ", ", при этом ", ", кроме того ", ", а также ")
        .map { sep -> window.lastIndexOf(sep) }
        .filter { idx -> idx >= 110 }
        .maxOrNull()

      cleaned = if (majorClauseBreak != null) {
        window.substring(0, majorClauseBreak).trimEnd(',', ';', '—', '-', ' ') + "."
      } else {
        val lastSpace = window.lastIndexOf(' ').takeIf { it >= 90 } ?: (maxChars - 1)
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
        if (word.length >= 6) word.substring(0, word.length - 2)
        else if (word.length == 5) word.substring(0, 4)
        else word
      }
      .toList()
  }

  /**
   * Формирует развёрнутый 6-секционный протокол совещания с кликабельными таймкодами `[MM:SS.d]`.
   */
  private fun buildSmartProtocol(
    record: TranscriptRecord,
    sentences: List<ExtractedSentence>,
    llmExecutiveSummary: String?,
  ): String = buildString {
    val topKeywords = extractTopTopicWords(sentences, limit = 6)
    val topicLine = if (topKeywords.isNotEmpty()) {
      topKeywords.joinToString(", ")
    } else {
      "рабочее обсуждение"
    }

    // Определяем ведущего спикера по суммарному времени речи
    val dominantSpeakerId = record.speakers.maxByOrNull { spk ->
      record.utterances.filter { it.speakerId == spk }.sumOf { (it.endMs - it.startMs).coerceAtLeast(0L) }
    }
    val dominantSpeakerName = dominantSpeakerId?.let { record.displayName(it) } ?: "Говорящий 1"

    appendLine("### 1. Паспорт и аналитическое резюме встречи")
    appendLine(
      "- **Длительность записи:** ${TranscriptExporter.formatTimeMs(record.durationMs)} · " +
        "**Участников:** ${record.speakers.size} (${record.speakers.joinToString { record.displayName(it) }}) · " +
        "**Реплик:** ${record.utterances.size}"
    )
    appendLine("- **Предметная область и ключевые понятия:** $topicLine.")
    appendLine("- **Основной докладчик (ведущий диалога):** $dominantSpeakerName.")
    appendLine()

    if (!llmExecutiveSummary.isNullOrBlank()) {
      appendLine(llmExecutiveSummary.trim())
    } else {
      val topTheses = selectDiverseSentences(
        pool = sentences.sortedByDescending { it.thesisScore },
        maxCount = 3,
      ).sortedBy { it.startMs }
      if (topTheses.isNotEmpty()) {
        val Opening = topTheses.first()
        val Conclusion = topTheses.last()
        appendLine(
          "В ходе встречи участники (${record.speakers.joinToString { record.displayName(it) }}) " +
            "рассмотрели вопросы по направлениям: **$topicLine**. " +
            "Обсуждение началось с позиции **${Opening.speakerName}**: *«${Opening.conciseText}»*" +
            (if (Conclusion != Opening) " В итоговой части **${Conclusion.speakerName}** сформулировал вывод: *«${Conclusion.conciseText}»*" else "")
        )
      }
    }

    // 2. Повестка и хронология обсуждения по смысловым этапам
    if (sentences.size >= 3 && record.durationMs >= 15_000L) {
      appendLine()
      appendLine("### 2. Повестка и хронология обсуждения (по этапам)")
      val blockCount = when {
        record.durationMs >= 180_000L && sentences.size >= 9 -> 4
        sentences.size >= 6 -> 3
        else -> 2
      }
      val stageTitles = when (blockCount) {
        4 -> listOf(
          "Этап 1. Вводная часть и постановка контекста",
          "Этап 2. Разбор основных вопросов и аргументация",
          "Этап 3. Детализация предложений и уточнения",
          "Этап 4. Подведение итогов и фиксация договорённостей",
        )
        3 -> listOf(
          "Этап 1. Постановка темы и исходный контекст",
          "Этап 2. Основное обсуждение и обмен аргументами",
          "Этап 3. Согласование выводов и завершение",
        )
        else -> listOf(
          "Этап 1. Постановка вопроса и обсуждение",
          "Этап 2. Итоговые позиции и выводы",
        )
      }

      val sortedByTime = sentences.sortedBy { it.startMs }
      for (b in 0 until blockCount) {
        val fromIdx = (b * sortedByTime.size) / blockCount
        val toIdx = (((b + 1) * sortedByTime.size) / blockCount).coerceAtLeast(fromIdx + 1).coerceAtMost(sortedByTime.size)
        if (fromIdx >= toIdx) continue
        val slice = sortedByTime.subList(fromIdx, toIdx)
        val bestInBlock = slice.maxByOrNull { it.thesisScore } ?: slice.first()
        val blockEndMs = slice.maxOf { it.endMs }
        val blockKeywords = extractTopTopicWords(slice, limit = 3)
        val kwSuffix = if (blockKeywords.isNotEmpty()) " *(темы: ${blockKeywords.joinToString(", ")})*" else ""
        appendLine(
          "- `[${TranscriptExporter.formatTimeMs(bestInBlock.startMs)}]` **${stageTitles[b]}** " +
            "(до ${TranscriptExporter.formatTimeMs(blockEndMs)})$kwSuffix: **${bestInBlock.speakerName}** — ${bestInBlock.conciseText}"
        )
      }
    }

    // 3. Ключевые тезисы и аргументы (с таймкодами)
    appendLine()
    appendLine("### 3. Ключевые тезисы и аргументы (с таймкодами)")
    val selectedTheses = selectDiverseSentences(
      pool = sentences.sortedByDescending { it.thesisScore },
      maxCount = 7,
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

    // 4. Позиции и ключевой вклад каждого участника
    appendLine()
    appendLine("### 4. Позиции и вклад каждого участника")
    for (spk in record.speakers) {
      val turns = record.utterances.filter { it.speakerId == spk }
      val durSec = turns.sumOf { (it.endMs - it.startMs).coerceAtLeast(0L) } / 1000.0
      val wordCount = turns.sumOf { it.text.split(Regex("\\s+")).count { w -> w.isNotBlank() } }
      val sharePct = if (record.durationMs > 0) {
        ((turns.sumOf { (it.endMs - it.startMs).coerceAtLeast(0L) } * 100.0) / record.durationMs).toInt().coerceIn(1, 100)
      } else 0

      val spkTopTheses = selectDiverseSentences(
        pool = sentences.filter { it.speakerId == spk }.sortedByDescending { it.thesisScore },
        maxCount = 2,
      ).sortedBy { it.startMs }

      appendLine(
        "- **${record.displayName(spk)}** — реплик: ${turns.size}, слов: $wordCount, " +
          "эфирное время: ${"%.1f".format(durSec)} с ($sharePct%)"
      )
      for (st in spkTopTheses) {
        appendLine(
          "  - `[${TranscriptExporter.formatTimeMs(st.startMs)}]` Позиция: ${st.conciseText}"
        )
      }
    }

    // 5. Поднятые вопросы и ответы собеседников (с таймкодами)
    val questions = selectDiverseSentences(
      pool = sentences
        .filter { it.rawText.contains("?") || it.isQuestionOrProblem }
        .sortedByDescending { (if (it.rawText.contains("?")) 2.0 else 0.0) + it.thesisScore },
      maxCount = 4,
    ).sortedBy { it.startMs }

    if (questions.isNotEmpty()) {
      appendLine()
      appendLine("### 5. Поднятые вопросы и ответы в ходе встречи")
      for (q in questions) {
        // Ищем ответ другого спикера в следующей реплике диалога
        val nextReply = record.utterances
          .drop(q.utteranceIndex + 1)
          .take(2)
          .firstOrNull { it.speakerId != q.speakerId && it.text.length >= 12 }
        val replySuffix = if (nextReply != null) {
          val replyConcise = compressToConciseThesis(nextReply.text, maxChars = 150)
          " — *Ответ (${record.displayName(nextReply.speakerId)}, `[${TranscriptExporter.formatTimeMs(nextReply.startMs)}]`):* $replyConcise"
        } else {
          ""
        }
        appendLine(
          "- `[${TranscriptExporter.formatTimeMs(q.startMs)}]` **${q.speakerName}**: ${q.conciseText}$replySuffix"
        )
      }
    }

    // 6. Принятые решения, договорённости и поручения
    val actions = selectDiverseSentences(
      pool = sentences.filter { it.isActionItem }.sortedByDescending { it.thesisScore },
      maxCount = 5,
    ).sortedBy { it.startMs }

    appendLine()
    appendLine("### 6. Итоги, решения и следующие шаги")
    if (actions.isNotEmpty()) {
      for (act in actions) {
        appendLine(
          "- `[${TranscriptExporter.formatTimeMs(act.startMs)}]` **${act.speakerName}**: ${act.conciseText}"
        )
      }
    } else {
      // Если явных глаголов-поручений не было, приводим финальное резюмирующее высказывание диалога
      val closing = sentences.takeLast(3).maxByOrNull { it.thesisScore }
      if (closing != null) {
        appendLine(
          "- `[${TranscriptExporter.formatTimeMs(closing.startMs)}]` **Итоговое резюме (${closing.speakerName})**: ${closing.conciseText}"
        )
      } else {
        appendLine("- Явных поручений с дедлайнами в ходе записи не зафиксировано.")
      }
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
          (inter / union) >= 0.65 || kotlin.math.abs(existing.startMs - cand.startMs) < 1500L
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
      .replace(Regex("^#{1,6}\\s*.*$", RegexOption.MULTILINE), "")
      .replace("/no_think", "")
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
            thinkingConfig = ThinkingConfig(false),
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
              ThinkingConfig(false),
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
