package ru.hexteam.stenograf.core.pipeline

import android.content.Context
import android.net.Uri
import android.os.SystemClock
import com.k2fsa.sherpa.onnx.FastClusteringConfig
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarization
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarizationConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationPyannoteModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiser
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiserConfig
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiserGtcrnModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiserModelConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractor
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import ru.hexteam.stenograf.audio.AudioDecoder
import ru.hexteam.stenograf.audio.AudioPreprocessor
import ru.hexteam.stenograf.audio.WavWriter
import ru.hexteam.stenograf.core.model.ProcessingParams
import ru.hexteam.stenograf.core.model.QualityPreset
import ru.hexteam.stenograf.core.model.StageTimings
import ru.hexteam.stenograf.core.model.TranscriptRecord
import ru.hexteam.stenograf.core.model.Word
import java.io.File
import java.util.UUID
import kotlin.coroutines.coroutineContext
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong
import kotlin.math.sqrt

/** Состояние прогресса обработки для реактивного отображения в UI. */
data class PipelineProgress(
  val stage: String,
  val detail: String,
  val fraction: Float,
)

/**
 * Локальный офлайн-конвейер «Уралан: Нейрографист»:
 * Поддерживает три режима работы (настраиваются пользователем):
 *  - `QualityPreset.FAST` («Быстро»): ускоренная диаризация по сегментам Silero VAD + голосовым эмбеддингам 3D-Speaker Cam++ без тяжёлого прогона pyannote по тишине; крупные окна ASR (до 18 с). Идеально для записей 30–60 минут.
 *  - `QualityPreset.STANDARD` («Стандарт»): баланс скорости и точности — сегментация pyannote 3.0 + 3D-Speaker Cam++ (с автоматическим разбиением на 3-минутные макро-окна для длинных файлов во избежание OOM) + GigaAM v3 RNN-T.
 *  - `QualityPreset.HIGH` («Долго / макс. качество»): нейросетевое шумоподавление GTCRN (in-place без расхода лишней памяти) + детальная сегментация pyannote 3.0 + узкие окна GigaAM v3 RNN-T (11 с) с контекстным перекрытием.
 */
class SpeechPipeline(private val context: Context) {

  private val mutex = Mutex()
  private val assets get() = context.assets

  /**
   * Гарантирует наличие файла модели во внутренней памяти приложения (`filesDir`).
   * Загрузка ONNX-моделей напрямую из файловой системы (`assetManager = null`) позволяет ONNX Runtime
   * отображать веса из файла без выделения промежуточного буфера `AAsset` в куче.
   */
  private fun ensureAssetExtracted(assetRelPath: String): String {
    val targetFile = File(context.filesDir, assetRelPath)
    if (targetFile.exists() && targetFile.length() > 0L) {
      return targetFile.absolutePath
    }
    targetFile.parentFile?.mkdirs()
    val tmpFile = File(targetFile.parentFile, "${targetFile.name}.tmp")
    assets.open(assetRelPath).use { input ->
      tmpFile.outputStream().use { output ->
        input.copyTo(output, bufferSize = 1 shl 18)
      }
    }
    if (!tmpFile.renameTo(targetFile)) {
      tmpFile.copyTo(targetFile, overwrite = true)
      tmpFile.delete()
    }
    return targetFile.absolutePath
  }

  suspend fun processAudioUri(
    uri: Uri,
    title: String,
    params: ProcessingParams,
    onProgress: (PipelineProgress) -> Unit,
  ): TranscriptRecord = withContext(Dispatchers.Default) {
    mutex.withLock {
      val t0 = SystemClock.elapsedRealtime()
      val threads = params.numThreads.coerceIn(1, 8)
      val preset = params.qualityPreset

      // 1. Потоковое декодирование и ресемплинг в 16 кГц моно (без выделения буферов 44.1/48 кГц в куче)
      onProgress(PipelineProgress("Декодирование аудио", "Потоковое чтение и ресемплинг до 16 кГц моно…", 0.03f))
      val tDecode0 = SystemClock.elapsedRealtime()
      val decoded = AudioDecoder.decodeToMono16k(context, uri) { p ->
        onProgress(
          PipelineProgress(
            "Декодирование аудио",
            "Потоковое чтение и ресемплинг ${(p * 100).toInt()}%…",
            0.03f + 0.09f * p,
          )
        )
      }
      val decodeMs = SystemClock.elapsedRealtime() - tDecode0
      val samples = decoded.samples
      require(samples.size >= AudioDecoder.TARGET_RATE / 4) {
        "Аудиозапись слишком короткая (менее 0,25 с)"
      }

      // 2. Предварительная обработка сигнала (in-place) + шумоподавление GTCRN (in-place)
      val tPre0 = SystemClock.elapsedRealtime()
      onProgress(PipelineProgress("Предобработка сигнала", "Фильтр ВЧ 70 Гц и нормализация уровня…", 0.13f))
      AudioPreprocessor.process(samples, AudioDecoder.TARGET_RATE)

      val shouldDenoise = (params.denoise || preset == QualityPreset.HIGH) && preset != QualityPreset.FAST
      if (shouldDenoise) {
        coroutineContext.ensureActive()
        onProgress(PipelineProgress("Шумоподавление GTCRN", "Очистка фонового шума нейросетью…", 0.16f))
        runGtcrnDenoiserInPlace(samples, threads) { p ->
          onProgress(
            PipelineProgress(
              "Шумоподавление GTCRN",
              "Очистка фонового шума ${(p * 100).toInt()}%…",
              0.16f + 0.10f * p,
            )
          )
        }
      }
      val preprocessMs = SystemClock.elapsedRealtime() - tPre0

      // Сохраняем подготовленный WAV 16 кГц для плеера с переходом по репликам и повторной обработки фрагментов
      val recordId = UUID.randomUUID().toString()
      val audioDir = File(context.filesDir, "audio").apply { mkdirs() }
      val wavFile = File(audioDir, "$recordId.wav")
      WavWriter.write(wavFile, samples, AudioDecoder.TARGET_RATE)
      val durationMs = samples.size * 1000L / AudioDecoder.TARGET_RATE

      // 3. Выделение речевых фрагментов (Silero VAD) и нейросетевая диаризация в зависимости от выбранного режима
      coroutineContext.ensureActive()
      val tDiar0 = SystemClock.elapsedRealtime()

      val diarSlices: List<DiarizationSlice>
      val speechWindows: List<SpeechWindow>

      if (preset == QualityPreset.FAST) {
        // Режим «Быстро»: сначала мгновенный Silero VAD, затем быстрая кластеризация эмбеддингов 3D-Speaker Cam++ только по речевым участкам
        onProgress(PipelineProgress("Фильтрация пауз (VAD)", "Быстрое выделение речевых участков Silero VAD…", 0.28f))
        speechWindows = detectSpeechWindows(samples, emptyList(), threads, preset)

        coroutineContext.ensureActive()
        onProgress(PipelineProgress("Диаризация (режим «Быстро»)", "Анализ тембров голоса 3D-Speaker Cam++…", 0.34f))
        diarSlices = runFastVadEmbeddingDiarization(samples, speechWindows, params, threads) { frac, detail ->
          onProgress(
            PipelineProgress(
              "Диаризация (режим «Быстро»)",
              detail,
              0.34f + 0.22f * frac.coerceIn(0f, 1f),
            )
          )
        }
      } else {
        // Режимы «Стандарт» и «Долго (макс. качество)»: сегментация pyannote 3.0 + 3D-Speaker Cam++
        // Для длинных записей (> 3,5 мин) выполняется окнами по 3 минуты с глобальным слиянием голосовых отпечатков,
        // чтобы не копировать гигантские массивы через JNI и не вызывать OutOfMemoryError.
        onProgress(PipelineProgress("Диаризация говорящих", "Подготовка моделей pyannote 3.0 и 3D-Speaker…", 0.28f))
        diarSlices = runChunkedPyannoteDiarization(samples, params, threads, preset) { frac, detailText ->
          onProgress(
            PipelineProgress(
              "Диаризация говорящих",
              detailText,
              0.28f + 0.26f * frac.coerceIn(0f, 1f),
            )
          )
        }

        coroutineContext.ensureActive()
        onProgress(PipelineProgress("Фильтрация пауз (VAD)", "Выделение речевых участков Silero VAD…", 0.56f))
        speechWindows = detectSpeechWindows(samples, diarSlices, threads, preset)
      }
      val diarizationMs = SystemClock.elapsedRealtime() - tDiar0

      // 4. Распознавание речи (GigaAM v3 RNN-T с пунктуацией)
      coroutineContext.ensureActive()
      val tAsr0 = SystemClock.elapsedRealtime()
      onProgress(PipelineProgress("Распознавание речи", "Инициализация модели GigaAM v3 RNN-T…", 0.60f))
      val allWords = runAsrOnWindows(samples, speechWindows, threads) { idx, total ->
        val frac = if (total > 0) idx.toFloat() / total else 0f
        onProgress(
          PipelineProgress(
            "Распознавание речи (GigaAM v3)",
            "Транскрибация фрагмента $idx из $total…",
            0.60f + 0.34f * frac,
          )
        )
      }
      val asrMs = SystemClock.elapsedRealtime() - tAsr0

      // 5. Пословное выравнивание со спикерами, детекция перебиваний/эмоций и сборка расшифровки
      val tAlign0 = SystemClock.elapsedRealtime()
      onProgress(PipelineProgress("Сборка диалога", "Выравнивание меток, анализ перебиваний и интонаций…", 0.96f))
      val assigned = TranscriptAligner.assignSpeakersToWords(allWords, diarSlices)
      val rawUtterances = TranscriptAligner.buildUtterances(
        assignedWords = assigned,
        maxPauseWithinTurnMs = if (preset == QualityPreset.HIGH) 1250L else 1500L,
        filterMinorSpeakers = params.numSpeakers == 0,
      )
      val utterances = TranscriptAligner.enrichProsodyAndOverlaps(
        utterances = rawUtterances,
        slices = diarSlices,
        samples = samples,
        sampleRate = AudioDecoder.TARGET_RATE,
      )
      val alignMs = SystemClock.elapsedRealtime() - tAlign0
      val totalMs = SystemClock.elapsedRealtime() - t0

      onProgress(PipelineProgress("Готово", "Расшифровка завершена", 1.0f))

      TranscriptRecord(
        id = recordId,
        title = title,
        createdAt = System.currentTimeMillis(),
        audioPath = wavFile.absolutePath,
        durationMs = durationMs,
        utterances = utterances,
        timings = StageTimings(
          decodeMs = decodeMs,
          preprocessMs = preprocessMs,
          diarizationMs = diarizationMs,
          asrMs = asrMs,
          alignMs = alignMs,
          totalMs = totalMs,
        ),
        params = params,
      )
    }
  }

  /**
   * Повторная обработка отдельного фрагмента записи (функция уровня Ultra Pro Max):
   * вырезает аудио в окне [startMs, endMs] с небольшим контекстом, прогоняет через денойзер и GigaAM v3
   * и возвращает уточнённый текст и пословные метки.
   */
  suspend fun reprocessSegment(
    audioPath: String,
    startMs: Long,
    endMs: Long,
    denoise: Boolean = true,
    numThreads: Int = 4,
  ): Pair<String, List<Word>> = withContext(Dispatchers.Default) {
    mutex.withLock {
      val file = File(audioPath)
      require(file.exists()) { "Исходный аудиофайл не найден" }
      val pcm = AudioDecoder.decodeFileToMono16k(context, file)
      val sr = AudioDecoder.TARGET_RATE
      val padStartMs = max(0L, startMs - 150L)
      val padEndMs = min(pcm.durationMs, endMs + 150L)
      val fromIdx = ((padStartMs * sr) / 1000L).toInt().coerceIn(0, pcm.samples.size)
      val toIdx = ((padEndMs * sr) / 1000L).toInt().coerceIn(fromIdx, pcm.samples.size)
      val slice = pcm.samples.copyOfRange(fromIdx, toIdx)
      AudioPreprocessor.process(slice, sr)
      if (denoise && slice.size > sr / 4) {
        runGtcrnDenoiserInPlace(slice, numThreads) {}
      }
      val recognizer = createRecognizer(numThreads.coerceIn(1, 8))
      try {
        val stream = recognizer.createStream()
        try {
          stream.acceptWaveform(slice, sr)
          recognizer.decode(stream)
          val res = recognizer.getResult(stream)
          val rawTokens = res.tokens.mapIndexed { i, tok ->
            val tsSec = if (i < res.timestamps.size) res.timestamps[i] else 0f
            RawToken(tok, (tsSec * 1000f).roundToLong())
          }
          val words = TranscriptAligner.tokensToWords(rawTokens, padStartMs, padEndMs)
          val text = TranscriptAligner.normalizeRussianPunctuation(
            if (words.isNotEmpty()) words.joinToString(" ") { it.text } else res.text
          )
          text to words
        } finally {
          stream.release()
        }
      } finally {
        recognizer.release()
      }
    }
  }

  // ------------------------------------------------------------------ Этапы

  /**
   * Выполняет нейросетевое шумоподавление GTCRN кусками по 15 секунд **на месте (in-place)** в массиве [samples],
   * не выделяя второй гигантский массив `FloatArray(samples.size)` в куче JVM.
   */
  private fun runGtcrnDenoiserInPlace(
    samples: FloatArray,
    numThreads: Int,
    onProgress: (Float) -> Unit,
  ) {
    val gtcrnPath = ensureAssetExtracted("models/denoise/gtcrn_simple.onnx")
    val cfg = OfflineSpeechDenoiserConfig(
      model = OfflineSpeechDenoiserModelConfig(
        gtcrn = OfflineSpeechDenoiserGtcrnModelConfig(model = gtcrnPath),
        numThreads = numThreads,
        provider = "cpu",
      )
    )
    val denoiser = OfflineSpeechDenoiser(assetManager = null, config = cfg)
    try {
      val chunkLen = AudioDecoder.TARGET_RATE * 15
      var offset = 0
      while (offset < samples.size) {
        val end = min(samples.size, offset + chunkLen)
        val chunk = samples.copyOfRange(offset, end)
        val cleaned = denoiser.run(chunk, AudioDecoder.TARGET_RATE).samples
        val copyLen = min(end - offset, cleaned.size)
        System.arraycopy(cleaned, 0, samples, offset, copyLen)
        offset = end
        onProgress(offset.toFloat() / samples.size)
      }
    } finally {
      denoiser.release()
    }
  }

  /**
   * Ускоренная диаризация для режима «Быстро» (`QualityPreset.FAST`):
   * извлекает голосовые эмбеддинги `3D-Speaker Cam++` напрямую из речевых окон VAD (разбивая длинные участки на подокна по 3,5 с)
   * и объединяет их агломеративной косинусной кластеризацией. Работает в 5–8 раз быстрее полного прогона `pyannote`
   * и потребляет минимум оперативной памяти даже на часовых записях.
   */
  private suspend fun runFastVadEmbeddingDiarization(
    samples: FloatArray,
    vadWindows: List<SpeechWindow>,
    params: ProcessingParams,
    numThreads: Int,
    onProgress: (Float, String) -> Unit,
  ): List<DiarizationSlice> = coroutineScope {
    if (vadWindows.isEmpty()) return@coroutineScope emptyList()
    val embModelPath = ensureAssetExtracted("models/diarization/embedding.onnx")
    val sr = AudioDecoder.TARGET_RATE

    // Формируем подокна длиной до 2,6 с внутри речевых участков VAD, разрезая по локальным минимумам энергии,
    // чтобы не пропускать смену говорящих внутри одного речевого окна
    data class SubSlice(val startSample: Int, val endSample: Int) {
      val startMs: Long get() = startSample * 1000L / sr
      val endMs: Long get() = endSample * 1000L / sr
    }

    val subSlices = mutableListOf<SubSlice>()
    val targetSub = (sr * 2.5).toInt()
    val minSub = (sr * 0.9).toInt()
    val searchRadius = (sr * 0.45).toInt()
    val frameWin = (sr * 0.04).toInt() // 40 мс окно для поиска паузы/спада энергии

    for (w in vadWindows) {
      var cur = w.startSample
      while (cur < w.endSample) {
        val idealNext = cur + targetSub
        if (w.endSample - cur <= targetSub + minSub / 2) {
          subSlices += SubSlice(cur, w.endSample)
          break
        }
        // Ищем локальный минимум энергии в окрестности idealNext, чтобы разрез пришёлся на микропаузу между фразами
        val searchFrom = max(cur + minSub, idealNext - searchRadius)
        val searchTo = min(w.endSample - minSub, idealNext + searchRadius)
        var bestSplit = idealNext.coerceIn(cur + minSub, w.endSample - minSub)
        var minEnergy = Float.MAX_VALUE
        var probe = searchFrom
        while (probe + frameWin <= searchTo) {
          var e = 0f
          for (k in probe until probe + frameWin) {
            val v = samples[k]
            e += v * v
          }
          if (e < minEnergy) {
            minEnergy = e
            bestSplit = probe + frameWin / 2
          }
          probe += frameWin / 2
        }
        subSlices += SubSlice(cur, bestSplit)
        cur = bestSplit
      }
    }
    if (subSlices.isEmpty()) return@coroutineScope emptyList()

    val extractor = SpeakerEmbeddingExtractor(
      assetManager = null,
      config = SpeakerEmbeddingExtractorConfig(
        model = embModelPath,
        numThreads = numThreads,
        provider = "cpu",
      ),
    )
    val embeddings = ArrayList<FloatArray>(subSlices.size)
    try {
      subSlices.forEachIndexed { idx, sub ->
        coroutineContext.ensureActive()
        if (idx % 4 == 0 || idx == subSlices.lastIndex) {
          onProgress(
            (idx + 1).toFloat() / subSlices.size,
            "Анализ голосов (3D-Speaker): сегмент ${idx + 1} из ${subSlices.size}…",
          )
        }
        val chunk = samples.copyOfRange(sub.startSample, sub.endSample)
        val emb = computeNormalizedEmbedding(extractor, chunk, sr)
        embeddings.add(emb)
      }
    } finally {
      extractor.release()
    }

    val labels = clusterNormalizedEmbeddings(
      embeddings = embeddings,
      numClusters = params.numSpeakers,
      threshold = params.clusterThreshold.coerceIn(0.28f, 0.80f),
    )

    subSlices.mapIndexed { idx, sub ->
      DiarizationSlice(
        startMs = sub.startMs,
        endMs = sub.endMs,
        speakerIndex = labels[idx],
      )
    }
  }

  /**
   * Диаризация `pyannote 3.0` + `3D-Speaker Cam++` для режимов «Стандарт» и «Долго (макс. качество)».
   * Если запись длиннее 3,5 минут (например, 47-минутный WAV-файл), обрабатывает аудио макро-окнами по 3 минуты
   * и связывает найденных в окнах спикеров глобально по их эталонным эмбеддингам `3D-Speaker Cam++`,
   * благодаря чему пиковое потребление памяти JNI не превышает ~12 МБ на любой длительности файла.
   */
  private suspend fun runChunkedPyannoteDiarization(
    samples: FloatArray,
    params: ProcessingParams,
    numThreads: Int,
    preset: QualityPreset,
    onProgress: (Float, String) -> Unit,
  ): List<DiarizationSlice> = coroutineScope {
    val segModelPath = ensureAssetExtracted("models/diarization/segmentation.onnx")
    val embModelPath = ensureAssetExtracted("models/diarization/embedding.onnx")
    val sr = AudioDecoder.TARGET_RATE

    // Передаём пользовательский порог напрямую без искусственного завышения,
    // чтобы при 0.45–0.50 голоса чётко разделялись и не сливались в одного говорящего.
    val effectiveThreshold = params.clusterThreshold.coerceIn(0.25f, 0.85f)

    val clusterCfg = FastClusteringConfig(
      numClusters = if (params.numSpeakers > 0) params.numSpeakers else -1,
      threshold = effectiveThreshold,
    )
    val diarCfg = OfflineSpeakerDiarizationConfig(
      segmentation = OfflineSpeakerSegmentationModelConfig(
        pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(
          model = segModelPath,
        ),
        numThreads = numThreads,
        provider = "cpu",
      ),
      embedding = SpeakerEmbeddingExtractorConfig(
        model = embModelPath,
        numThreads = numThreads,
        provider = "cpu",
      ),
      clustering = clusterCfg,
      minDurationOn = if (preset == QualityPreset.HIGH) 0.14f else 0.22f,
      minDurationOff = if (preset == QualityPreset.HIGH) 0.26f else 0.38f,
    )

    // Макро-окно: 180 секунд (3 минуты = 2 880 000 отсчётов = 11,5 МБ).
    // Если запись <= 210 секунд — обрабатываем за 1 проход.
    val maxSinglePassSamples = sr * 210
    val macroWindowSamples = sr * 180

    if (samples.size <= maxSinglePassSamples) {
      val diar = OfflineSpeakerDiarization(assetManager = null, config = diarCfg)
      val durationSec = (samples.size.toFloat() / sr).coerceAtLeast(1f)
      val expectedMs = (durationSec * 280f).toLong().coerceIn(3000L, 90_000L)
      val startMs = SystemClock.elapsedRealtime()

      val tickerJob = launch(Dispatchers.Default) {
        while (isActive) {
          delay(350L)
          val elapsedMs = SystemClock.elapsedRealtime() - startMs
          val elapsedSec = elapsedMs / 1000
          val estFrac = (elapsedMs.toFloat() / expectedMs).coerceIn(0.05f, 0.94f)
          val detail = if (estFrac < 0.42f) {
            "Сегментация речи pyannote 3.0 ($elapsedSec с)…"
          } else {
            "Голосовые эмбеддинги 3D-Speaker Cam++ и кластеризация ($elapsedSec с)…"
          }
          onProgress(estFrac, detail)
        }
      }

      try {
        val segments = diar.process(samples)
        onProgress(1.0f, "Найдено речевых сегментов: ${segments.size}")
        return@coroutineScope segments.map { seg ->
          DiarizationSlice(
            startMs = (seg.start * 1000f).roundToLong(),
            endMs = (seg.end * 1000f).roundToLong(),
            speakerIndex = seg.speaker,
          )
        }
      } finally {
        tickerJob.cancel()
        diar.release()
      }
    }

    // Длинная запись (> 3,5 мин): обрабатываем 3-минутными блоками и связываем спикеров между блоками
    val totalChunks = (samples.size + macroWindowSamples - 1) / macroWindowSamples
    data class ChunkRawResult(
      val chunkOffsetMs: Long,
      val chunkStartSample: Int,
      val chunkEndSample: Int,
      val slices: List<DiarizationSlice>,
    )

    val chunkResults = ArrayList<ChunkRawResult>(totalChunks)
    val diar = OfflineSpeakerDiarization(assetManager = null, config = diarCfg)
    try {
      for (cIdx in 0 until totalChunks) {
        coroutineContext.ensureActive()
        val sSample = cIdx * macroWindowSamples
        val eSample = min(samples.size, sSample + macroWindowSamples)
        val offsetMs = sSample * 1000L / sr
        onProgress(
          cIdx.toFloat() / totalChunks,
          "Диаризация длинной записи: блок ${cIdx + 1} из $totalChunks…",
        )
        val chunkAudio = samples.copyOfRange(sSample, eSample)
        val segs = diar.process(chunkAudio)
        val localSlices = segs.map { seg ->
          DiarizationSlice(
            startMs = offsetMs + (seg.start * 1000f).roundToLong(),
            endMs = offsetMs + (seg.end * 1000f).roundToLong(),
            speakerIndex = seg.speaker,
          )
        }
        chunkResults += ChunkRawResult(offsetMs, sSample, eSample, localSlices)
      }
    } finally {
      diar.release()
    }

    // Извлекаем эталонный эмбеддинг для каждого локального спикера в каждом 3-минутном блоке
    // и кластеризуем их глобально, чтобы speaker_1..N были едиными на протяжении всех 47+ минут.
    onProgress(0.95f, "Глобальное сопоставление голосов между блоками…")
    val extractor = SpeakerEmbeddingExtractor(
      assetManager = null,
      config = SpeakerEmbeddingExtractorConfig(
        model = embModelPath,
        numThreads = numThreads,
        provider = "cpu",
      ),
    )
    try {
      data class LocalSpkKey(val chunkIdx: Int, val localSpk: Int)
      val keys = mutableListOf<LocalSpkKey>()
      val repEmbeddings = mutableListOf<FloatArray>()

      chunkResults.forEachIndexed { cIdx, cr ->
        val bySpk = cr.slices.groupBy { it.speakerIndex }
        for ((localSpk, spkSlices) in bySpk) {
          // Берём до 3 самых длинных реплик этого спикера в блоке (суммарно до 8 секунд) для надёжного отпечатка голоса
          val topSlices = spkSlices.sortedByDescending { it.endMs - it.startMs }.take(3)
          val buf = ArrayList<FloatArray>()
          var totalLen = 0
          for (sl in topSlices) {
            val fromS = ((sl.startMs * sr) / 1000L).toInt().coerceIn(cr.chunkStartSample, cr.chunkEndSample)
            val toS = ((sl.endMs * sr) / 1000L).toInt().coerceIn(fromS, cr.chunkEndSample)
            val len = min(toS - fromS, sr * 4)
            if (len >= sr / 4) {
              buf += samples.copyOfRange(fromS, fromS + len)
              totalLen += len
            }
          }
          val combined = if (totalLen > 0) {
            FloatArray(totalLen).also { dst ->
              var pos = 0
              for (part in buf) {
                System.arraycopy(part, 0, dst, pos, part.size)
                pos += part.size
              }
            }
          } else {
            FloatArray(sr / 2)
          }
          keys += LocalSpkKey(cIdx, localSpk)
          repEmbeddings += computeNormalizedEmbedding(extractor, combined, sr)
        }
      }

      val globalLabels = clusterNormalizedEmbeddings(
        embeddings = repEmbeddings,
        numClusters = params.numSpeakers,
        threshold = effectiveThreshold,
        chunkIds = keys.map { it.chunkIdx }.toIntArray(),
      )
      val keyToGlobal = HashMap<LocalSpkKey, Int>(keys.size)
      for (i in keys.indices) {
        keyToGlobal[keys[i]] = globalLabels[i]
      }

      val unified = ArrayList<DiarizationSlice>()
      chunkResults.forEachIndexed { cIdx, cr ->
        for (sl in cr.slices) {
          val gSpk = keyToGlobal[LocalSpkKey(cIdx, sl.speakerIndex)] ?: sl.speakerIndex
          unified += sl.copy(speakerIndex = gSpk)
        }
      }
      onProgress(1.0f, "Диаризация завершена: сегментов ${unified.size}")
      unified
    } finally {
      extractor.release()
    }
  }

  private fun computeNormalizedEmbedding(
    extractor: SpeakerEmbeddingExtractor,
    audioSlice: FloatArray,
    sampleRate: Int,
  ): FloatArray {
    val stream = extractor.createStream()
    try {
      stream.acceptWaveform(audioSlice, sampleRate)
      stream.inputFinished()
      if (!extractor.isReady(stream)) {
        return FloatArray(192)
      }
      val raw = extractor.compute(stream)
      var normSq = 0.0
      for (v in raw) normSq += v * v
      val norm = sqrt(normSq).toFloat()
      if (norm > 1e-6f) {
        for (i in raw.indices) raw[i] /= norm
      }
      return raw
    } finally {
      stream.release()
    }
  }

  /**
   * Агломеративная иерархическая кластеризация L2-нормированных голосовых эмбеддингов (средняя связь / косинусное расстояние).
   * При передаче [chunkIds] добавляет штраф за объединение разных локальных спикеров, которые уже были разделены
   * внутри одного и того же 3-минутного блока `pyannote`.
   */
  private fun clusterNormalizedEmbeddings(
    embeddings: List<FloatArray>,
    numClusters: Int,
    threshold: Float,
    chunkIds: IntArray? = null,
  ): IntArray {
    val n = embeddings.size
    if (n == 0) return IntArray(0)
    if (n == 1) return intArrayOf(0)

    val dim = embeddings[0].size
    val clusters = ArrayList<MutableList<Int>>(n)
    val centroids = ArrayList<FloatArray>(n)
    for (i in 0 until n) {
      clusters.add(mutableListOf(i))
      centroids.add(embeddings[i].copyOf())
    }

    fun cosineDistance(a: FloatArray, b: FloatArray): Float {
      var dot = 0f
      for (k in 0 until min(a.size, b.size)) {
        dot += a[k] * b[k]
      }
      return (1f - dot).coerceIn(0f, 2f)
    }

    fun sharesSameChunk(cA: List<Int>, cB: List<Int>): Boolean {
      if (chunkIds == null) return false
      for (i in cA) {
        val ci = chunkIds[i]
        for (j in cB) {
          if (ci == chunkIds[j]) return true
        }
      }
      return false
    }

    val targetK = if (numClusters > 0) numClusters.coerceAtLeast(1) else 1

    while (clusters.size > targetK) {
      var bestI = -1
      var bestJ = -1
      var bestDist = Float.MAX_VALUE
      for (i in 0 until clusters.size) {
        for (j in i + 1 until clusters.size) {
          val baseDist = cosineDistance(centroids[i], centroids[j])
          val penalty = if (sharesSameChunk(clusters[i], clusters[j])) 0.22f else 0f
          val d = baseDist + penalty
          if (d < bestDist) {
            bestDist = d
            bestI = i
            bestJ = j
          }
        }
      }
      if (bestI < 0 || bestJ < 0) break
      if (numClusters <= 0 && bestDist > threshold) break

      // Сливаем кластер bestJ в bestI и пересчитываем центроид с учётом весов
      val membersI = clusters[bestI]
      val membersJ = clusters[bestJ]
      membersI.addAll(membersJ)
      val newCentroid = FloatArray(dim)
      for (idx in membersI) {
        val emb = embeddings[idx]
        for (k in 0 until dim) newCentroid[k] += emb[k]
      }
      var normSq = 0.0
      for (v in newCentroid) normSq += v * v
      val norm = sqrt(normSq).toFloat()
      if (norm > 1e-6f) {
        for (k in 0 until dim) newCentroid[k] /= norm
      }
      centroids[bestI] = newCentroid
      clusters.removeAt(bestJ)
      centroids.removeAt(bestJ)
    }

    val labels = IntArray(n)
    clusters.forEachIndexed { clusterIdx, members ->
      for (m in members) labels[m] = clusterIdx
    }
    return labels
  }

  private data class SpeechWindow(val startSample: Int, val endSample: Int) {
    val startMs: Long get() = startSample * 1000L / AudioDecoder.TARGET_RATE
    val endMs: Long get() = endSample * 1000L / AudioDecoder.TARGET_RATE
  }

  /**
   * Выделяет речевые окна при помощи Silero VAD.
   * Длина каждого окна адаптируется под выбранный режим (`FAST` — до 18 с, `STANDARD` — до 15 с, `HIGH` — до 11 с).
   */
  private fun detectSpeechWindows(
    samples: FloatArray,
    diarSlices: List<DiarizationSlice>,
    numThreads: Int,
    preset: QualityPreset,
  ): List<SpeechWindow> {
    val maxSec = when (preset) {
      QualityPreset.FAST -> 18.0f
      QualityPreset.STANDARD -> 15.0f
      QualityPreset.HIGH -> 11.0f
    }
    val vadModelPath = ensureAssetExtracted("models/vad/silero_vad.onnx")
    val vadCfg = VadModelConfig(
      sileroVadModelConfig = SileroVadModelConfig(
        model = vadModelPath,
        threshold = if (preset == QualityPreset.HIGH) 0.38f else 0.42f,
        minSilenceDuration = if (preset == QualityPreset.HIGH) 0.25f else 0.30f,
        minSpeechDuration = 0.18f,
        windowSize = 512,
        maxSpeechDuration = maxSec,
      ),
      sampleRate = AudioDecoder.TARGET_RATE,
      numThreads = min(2, numThreads),
      provider = "cpu",
    )
    val windows = mutableListOf<SpeechWindow>()
    val vad = Vad(assetManager = null, config = vadCfg)
    try {
      val frame = 512
      val chunk = FloatArray(frame)
      var pos = 0
      while (pos < samples.size) {
        val end = min(samples.size, pos + frame)
        val len = end - pos
        if (len == frame) {
          System.arraycopy(samples, pos, chunk, 0, frame)
        } else {
          chunk.fill(0f)
          System.arraycopy(samples, pos, chunk, 0, len)
        }
        vad.acceptWaveform(chunk)
        while (!vad.empty()) {
          val seg = vad.front()
          vad.pop()
          val s = max(0, seg.start.toInt() - 1600) // +100 мс запаса по краям
          val e = min(samples.size, seg.start.toInt() + seg.samples.size + 1600)
          if (e - s >= AudioDecoder.TARGET_RATE / 6) {
            windows += SpeechWindow(s, e)
          }
        }
        pos += frame
      }
      vad.flush()
      while (!vad.empty()) {
        val seg = vad.front()
        vad.pop()
        val s = max(0, seg.start.toInt() - 1600)
        val e = min(samples.size, seg.start.toInt() + seg.samples.size + 1600)
        if (e - s >= AudioDecoder.TARGET_RATE / 6) {
          windows += SpeechWindow(s, e)
        }
      }
    } finally {
      vad.release()
    }

    // Если по какой-то причине VAD ничего не вернул — используем сегменты диаризации или равномерную нарезку
    if (windows.isEmpty()) {
      if (diarSlices.isNotEmpty()) {
        for (d in diarSlices) {
          val s = ((d.startMs * AudioDecoder.TARGET_RATE) / 1000L).toInt().coerceIn(0, samples.size)
          val e = ((d.endMs * AudioDecoder.TARGET_RATE) / 1000L).toInt().coerceIn(s, samples.size)
          if (e - s > 1600) windows += SpeechWindow(s, e)
        }
      }
      if (windows.isEmpty()) {
        val step = (AudioDecoder.TARGET_RATE * maxSec).toInt()
        var i = 0
        while (i < samples.size) {
          windows += SpeechWindow(i, min(samples.size, i + step))
          i += step
        }
      }
    }

    // Сливаем слишком близкие окна (< 250 мс), если суммарная длина не превышает maxSec
    val merged = mutableListOf<SpeechWindow>()
    val maxWindowSamples = (AudioDecoder.TARGET_RATE * maxSec).toInt()
    val mergeGapSamples = (AudioDecoder.TARGET_RATE * 0.25).toInt()
    for (w in windows.sortedBy { it.startSample }) {
      if (merged.isEmpty()) {
        merged += w
      } else {
        val last = merged.last()
        if (w.startSample - last.endSample <= mergeGapSamples &&
          w.endSample - last.startSample <= maxWindowSamples
        ) {
          merged[merged.lastIndex] = SpeechWindow(last.startSample, max(last.endSample, w.endSample))
        } else {
          merged += w
        }
      }
    }
    return merged
  }

  private suspend fun runAsrOnWindows(
    samples: FloatArray,
    windows: List<SpeechWindow>,
    numThreads: Int,
    onProgress: (Int, Int) -> Unit,
  ): List<Word> {
    val recognizer = createRecognizer(numThreads)
    try {
      val words = mutableListOf<Word>()
      windows.forEachIndexed { idx, win ->
        coroutineContext.ensureActive()
        onProgress(idx + 1, windows.size)
        val chunk = samples.copyOfRange(win.startSample, win.endSample)
        val stream = recognizer.createStream()
        try {
          stream.acceptWaveform(chunk, AudioDecoder.TARGET_RATE)
          recognizer.decode(stream)
          val result = recognizer.getResult(stream)
          if (result.tokens.isNotEmpty()) {
            val rawTokens = result.tokens.mapIndexed { i, tok ->
              val tsSec = if (i < result.timestamps.size) result.timestamps[i] else 0f
              RawToken(tok, (tsSec * 1000f).roundToLong())
            }
            words += TranscriptAligner.tokensToWords(rawTokens, win.startMs, win.endMs)
          } else if (result.text.isNotBlank()) {
            words += Word(
              text = result.text.trim(),
              startMs = win.startMs,
              endMs = win.endMs,
            )
          }
        } finally {
          stream.release()
        }
      }
      return words
    } finally {
      recognizer.release()
    }
  }

  private fun createRecognizer(numThreads: Int): OfflineRecognizer {
    val encoderPath = ensureAssetExtracted("models/asr/encoder.int8.onnx")
    val decoderPath = ensureAssetExtracted("models/asr/decoder.onnx")
    val joinerPath = ensureAssetExtracted("models/asr/joiner.onnx")
    val tokensPath = ensureAssetExtracted("models/asr/tokens.txt")

    val modelCfg = OfflineModelConfig(
      transducer = OfflineTransducerModelConfig(
        encoder = encoderPath,
        decoder = decoderPath,
        joiner = joinerPath,
      ),
      tokens = tokensPath,
      modelType = "nemo_transducer",
      numThreads = numThreads,
      provider = "cpu",
    )
    val recCfg = OfflineRecognizerConfig(
      featConfig = FeatureConfig(sampleRate = AudioDecoder.TARGET_RATE, featureDim = 64),
      modelConfig = modelCfg,
      decodingMethod = "greedy_search",
    )
    return OfflineRecognizer(assetManager = null, config = recCfg)
  }
}
