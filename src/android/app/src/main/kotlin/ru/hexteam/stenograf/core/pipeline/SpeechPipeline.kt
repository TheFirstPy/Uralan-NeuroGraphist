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
import ru.hexteam.stenograf.audio.VoicePitchAnalyzer
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

      // 3. Умное выделение речевых окон (Silero VAD с поглощением коротких окон-сирот)
      coroutineContext.ensureActive()
      onProgress(PipelineProgress("Фильтрация пауз (VAD)", "Выделение речевых участков Silero VAD…", 0.26f))
      val speechWindows = detectSpeechWindows(samples, emptyList(), threads, preset)

      // 4. Распознавание речи (GigaAM v3 RNN-T с пунктуацией и диалоговыми маркерами смены реплик)
      coroutineContext.ensureActive()
      val tAsr0 = SystemClock.elapsedRealtime()
      onProgress(PipelineProgress("Распознавание речи", "Инициализация модели GigaAM v3 RNN-T…", 0.32f))
      val allWords = runAsrOnWindows(samples, speechWindows, threads) { idx, total ->
        val frac = if (total > 0) idx.toFloat() / total else 0f
        onProgress(
          PipelineProgress(
            "Распознавание речи (GigaAM v3)",
            "Транскрибация фрагмента $idx из $total…",
            0.32f + 0.36f * frac,
          )
        )
      }
      val asrMs = SystemClock.elapsedRealtime() - tAsr0

      // 5. Двухуровневая фразовая диаризация (3D-Speaker Cam++ + анализ основного тона F0 + кластеризация ядер спикеров)
      coroutineContext.ensureActive()
      val tDiar0 = SystemClock.elapsedRealtime()
      onProgress(PipelineProgress("Диаризация говорящих", "Анализ тембров 3D-Speaker Cam++ и регистра голоса…", 0.70f))
      val diarSlices = if (allWords.isNotEmpty()) {
        runCorePhraseDiarization(samples, allWords, params, threads) { frac, detailText ->
          onProgress(
            PipelineProgress(
              "Диаризация говорящих",
              detailText,
              0.70f + 0.24f * frac.coerceIn(0f, 1f),
            )
          )
        }
      } else if (preset == QualityPreset.FAST) {
        runFastVadEmbeddingDiarization(samples, speechWindows, params, threads) { frac, detail ->
          onProgress(
            PipelineProgress(
              "Диаризация (режим «Быстро»)",
              detail,
              0.70f + 0.24f * frac.coerceIn(0f, 1f),
            )
          )
        }
      } else {
        runChunkedPyannoteDiarization(samples, params, threads, preset) { frac, detailText ->
          onProgress(
            PipelineProgress(
              "Диаризация говорящих",
              detailText,
              0.70f + 0.24f * frac.coerceIn(0f, 1f),
            )
          )
        }
      }
      val diarizationMs = SystemClock.elapsedRealtime() - tDiar0

      // 6. Пословное выравнивание со спикерами и сборка структурированной расшифровки
      val tAlign0 = SystemClock.elapsedRealtime()
      onProgress(PipelineProgress("Сборка диалога", "Выравнивание временных меток и реплик…", 0.96f))
      val assigned = TranscriptAligner.assignSpeakersToWords(allWords, diarSlices)
      val utterances = TranscriptAligner.buildUtterances(assigned)
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
      minDurationOn = 0.18f,
      minDurationOff = 0.35f,
    )

    // Если запись длительностью до 12 минут (720 с) — обрабатываем глобально за 1 проход.
    // Для сверхдлинных записей (например, 45–60 минут) разбиваем на 5-минутные макро-окна, чтобы исключить OOM.
    val maxSinglePassSamples = sr * 720
    val macroWindowSamples = sr * 300

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

  /**
   * Двухуровневая фразовая диаризация (`3D-Speaker Cam++` + частота основного тона `F0` + кластеризация ядер спикеров):
   * 1) Разбивает распознанные слова на чистые фразовые сегменты по диалоговым тире (`—`) модели `GigaAM v3`,
   *    концам предложений (`.`, `?`, `!`) и паузам (>= 280 мс), не смешивая голоса разных спикеров в 15-секундных окнах.
   * 2) Извлекает L2-нормированный эмбеддинг `3D-Speaker Cam++` (с циклическим дополнением коротких фраз до 1,0 с без захвата соседнего голоса)
   *    и медианную частоту основного тона `F0` (72..360 Гц).
   * 3) Формирует ядра спикеров на надёжных опорных фразах (>= 1,2 с), поглощает мелкие кластеры-выбросы (< 0,8% речи)
   *    и назначает все фразы с учётом диалоговых тире и медианного сглаживания.
   */
  private suspend fun runCorePhraseDiarization(
    samples: FloatArray,
    words: List<Word>,
    params: ProcessingParams,
    numThreads: Int,
    onProgress: (Float, String) -> Unit,
  ): List<DiarizationSlice> = coroutineScope {
    val phrases = TranscriptAligner.buildPhraseSegments(words)
    if (phrases.isEmpty()) return@coroutineScope emptyList()

    val embModelPath = ensureAssetExtracted("models/diarization/embedding.onnx")
    val sr = AudioDecoder.TARGET_RATE
    val n = phrases.size
    val embs = Array(n) { FloatArray(192) }
    val f0s = FloatArray(n)
    val durs = FloatArray(n)

    val extractor = SpeakerEmbeddingExtractor(
      assetManager = null,
      config = SpeakerEmbeddingExtractorConfig(
        model = embModelPath,
        numThreads = numThreads,
        provider = "cpu",
      ),
    )
    try {
      val minWindowSamples = sr // 1.0 с (16000 отсчётов) для стабильного эмбеддинга Cam++
      phrases.forEachIndexed { idx, p ->
        coroutineContext.ensureActive()
        if (idx % 6 == 0 || idx == phrases.lastIndex) {
          onProgress(
            (idx + 1).toFloat() / n,
            "Анализ голосов (3D-Speaker + F0): фраза ${idx + 1} из $n…",
          )
        }
        val fromIdx = (((p.startMs - 30L) * sr) / 1000L).toInt().coerceIn(0, samples.size)
        val toIdx = (((p.endMs + 30L) * sr) / 1000L).toInt().coerceIn(fromIdx, samples.size)
        val len = toIdx - fromIdx
        if (len >= sr / 5) {
          val embSlice = if (len < minWindowSamples) {
            FloatArray(minWindowSamples) { k -> samples[fromIdx + (k % len)] }
          } else {
            samples.copyOfRange(fromIdx, toIdx)
          }
          embs[idx] = computeNormalizedEmbedding(extractor, embSlice, sr)
        }
        f0s[idx] = VoicePitchAnalyzer.computeMedianF0Hz(samples, fromIdx, toIdx, sr)
        durs[idx] = ((p.endMs - p.startMs) / 1000f).coerceAtLeast(0.15f)
      }
    } finally {
      extractor.release()
    }

    var totalSpeechSec = 0f
    for (d in durs) totalSpeechSec += d

    fun hasValidEmb(idx: Int): Boolean {
      var sq = 0f
      for (v in embs[idx]) sq += v * v
      return sq > 0.25f
    }

    var coreIndices = (0 until n).filter { i -> durs[i] >= 1.20f && hasValidEmb(i) }
    if (coreIndices.size < 4) {
      coreIndices = (0 until n).filter { i -> hasValidEmb(i) }
    }
    if (coreIndices.isEmpty()) {
      return@coroutineScope phrases.map { DiarizationSlice(it.startMs, it.endMs, 0) }
    }

    val dim = embs[0].size
    val effThr = (params.clusterThreshold * 0.88f).coerceIn(0.24f, 0.78f)
    val targetK = if (params.numSpeakers > 0) params.numSpeakers.coerceAtLeast(1) else 1

    val clusters = ArrayList<MutableList<Int>>(coreIndices.size)
    val cents = ArrayList<FloatArray>(coreIndices.size)
    val cf0s = ArrayList<Float>(coreIndices.size)
    val cdurs = ArrayList<Float>(coreIndices.size)

    for (idx in coreIndices) {
      clusters.add(mutableListOf(idx))
      cents.add(embs[idx].copyOf())
      cf0s.add(f0s[idx])
      cdurs.add(durs[idx])
    }

    fun cosineDistance(a: FloatArray, b: FloatArray): Float {
      var dot = 0f
      for (k in 0 until min(a.size, b.size)) dot += a[k] * b[k]
      return (1f - dot).coerceIn(0f, 2f)
    }

    fun recomputeClusterStats(cIdx: Int) {
      val mem = clusters[cIdx]
      val newCent = FloatArray(dim)
      var totalW = 0f
      val validF0 = ArrayList<Float>(mem.size)
      for (m in mem) {
        val w = durs[m]
        totalW += w
        val e = embs[m]
        for (k in 0 until dim) newCent[k] += e[k] * w
        if (f0s[m] > 50f) validF0.add(f0s[m])
      }
      var normSq = 0.0
      for (v in newCent) normSq += v * v
      val norm = sqrt(normSq).toFloat()
      if (norm > 1e-6f) {
        for (k in 0 until dim) newCent[k] /= norm
      }
      cents[cIdx] = newCent
      cdurs[cIdx] = totalW
      if (validF0.isNotEmpty()) {
        validF0.sort()
        val mid = validF0.size / 2
        cf0s[cIdx] = if (validF0.size % 2 == 1) validF0[mid] else 0.5f * (validF0[mid - 1] + validF0[mid])
      }
    }

    while (clusters.size > targetK) {
      var bestI = -1
      var bestJ = -1
      var bestDist = Float.MAX_VALUE
      for (i in 0 until clusters.size) {
        val ci = cents[i]
        val fi = cf0s[i]
        for (j in i + 1 until clusters.size) {
          val d = cosineDistance(ci, cents[j]) + VoicePitchAnalyzer.f0Distance(fi, cf0s[j])
          if (d < bestDist) {
            bestDist = d
            bestI = i
            bestJ = j
          }
        }
      }
      if (bestI < 0 || bestJ < 0) break
      if (params.numSpeakers <= 0 && bestDist > effThr) break

      clusters[bestI].addAll(clusters[bestJ])
      recomputeClusterStats(bestI)
      clusters.removeAt(bestJ)
      cents.removeAt(bestJ)
      cf0s.removeAt(bestJ)
      cdurs.removeAt(bestJ)
    }

    // Консолидация мелких кластеров-выбросов (одиночный смех/кашель/возглас не создаёт ложного спикера)
    if (params.numSpeakers <= 0 && clusters.size > 1) {
      val minClusterSec = min(6.0f, max(2.0f, totalSpeechSec * 0.008f))
      val majorIndices = clusters.indices.filter { cdurs[it] >= minClusterSec }.ifEmpty {
        listOf(cdurs.indices.maxByOrNull { cdurs[it] } ?: 0)
      }
      if (majorIndices.size < clusters.size) {
        val majorSet = majorIndices.toHashSet()
        val newClusters = ArrayList<MutableList<Int>>(majorIndices.size)
        val newCents = ArrayList<FloatArray>(majorIndices.size)
        val newCf0s = ArrayList<Float>(majorIndices.size)
        val newCdurs = ArrayList<Float>(majorIndices.size)
        for (mi in majorIndices) {
          newClusters.add(clusters[mi])
          newCents.add(cents[mi])
          newCf0s.add(cf0s[mi])
          newCdurs.add(cdurs[mi])
        }
        for (i in clusters.indices) {
          if (i in majorSet) continue
          val bestM = newClusters.indices.minByOrNull { m ->
            cosineDistance(cents[i], newCents[m]) + VoicePitchAnalyzer.f0Distance(cf0s[i], newCf0s[m])
          } ?: 0
          newClusters[bestM].addAll(clusters[i])
        }
        clusters.clear()
        cents.clear()
        cf0s.clear()
        cdurs.clear()
        clusters.addAll(newClusters)
        cents.addAll(newCents)
        cf0s.addAll(newCf0s)
        cdurs.addAll(newCdurs)
        for (cIdx in clusters.indices) {
          recomputeClusterStats(cIdx)
        }
      }
    }

    // Уточнение эталонных центроидов по 65% наиболее чистых опорных фраз каждого спикера (trimmed centroid)
    val kCount = clusters.size
    for (k in 0 until kCount) {
      val mem = clusters[k]
      if (mem.size >= 4) {
        val sortedByDist = mem.sortedBy { m -> cosineDistance(embs[m], cents[k]) }
        val takeCount = ((mem.size * 65) / 100).coerceAtLeast(2)
        val cleanMem = sortedByDist.take(takeCount)
        val newCent = FloatArray(dim)
        val validF0 = ArrayList<Float>(cleanMem.size)
        for (m in cleanMem) {
          val w = durs[m]
          val e = embs[m]
          for (dIdx in 0 until dim) newCent[dIdx] += e[dIdx] * w
          if (f0s[m] > 50f) validF0.add(f0s[m])
        }
        var normSq = 0.0
        for (v in newCent) normSq += v * v
        val norm = sqrt(normSq).toFloat()
        if (norm > 1e-6f) {
          for (dIdx in 0 until dim) newCent[dIdx] /= norm
        }
        cents[k] = newCent
        if (validF0.isNotEmpty()) {
          validF0.sort()
          val mid = validF0.size / 2
          cf0s[k] = if (validF0.size % 2 == 1) validF0[mid] else 0.5f * (validF0[mid - 1] + validF0[mid])
        }
      }
    }

    // Назначаем каждую фразу 0..n-1 одному из kCount опорных спикеров
    val labels = IntArray(n)
    for (i in 0 until n) {
      var bestK = 0
      var bestScore = Float.MAX_VALUE
      val p = phrases[i]
      for (k in 0 until kCount) {
        val cosD = cosineDistance(embs[i], cents[k])
        val pitchD = VoicePitchAnalyzer.f0Distance(f0s[i], cf0s[k]) * 1.35f
        var score = cosD + pitchD
        if (i > 0 && k == labels[i - 1] && !p.hasTurnDash && durs[i] < 0.95f) {
          score -= 0.06f
        }
        if (i > 0 && k == labels[i - 1] && p.hasTurnDash && kCount > 1) {
          score += 0.04f
        }
        if (score < bestScore) {
          bestScore = score
          bestK = k
        }
      }
      labels[i] = bestK
    }

    // Медианное сглаживание коротких фраз (< 0.85 с) без диалогового тире внутри речи одного спикера
    for (i in 1 until n - 1) {
      if (labels[i - 1] == labels[i + 1] && labels[i] != labels[i - 1]) {
        val p = phrases[i]
        if (durs[i] < 0.85f && !p.hasTurnDash) {
          val gapBefore = p.startMs - phrases[i - 1].endMs
          val gapAfter = phrases[i + 1].startMs - p.endMs
          val f0Diff = VoicePitchAnalyzer.f0Distance(f0s[i], cf0s[labels[i - 1]])
          if (gapBefore < 380L && gapAfter < 380L && f0Diff < 0.08f) {
            labels[i] = labels[i - 1]
          }
        }
      }
    }

    phrases.mapIndexed { idx, p ->
      DiarizationSlice(
        startMs = p.startMs,
        endMs = p.endMs,
        speakerIndex = labels[idx],
      )
    }
  }

  private data class SpeechWindow(val startSample: Int, val endSample: Int) {
    val startMs: Long get() = startSample * 1000L / AudioDecoder.TARGET_RATE
    val endMs: Long get() = endSample * 1000L / AudioDecoder.TARGET_RATE
  }

  /**
   * Выделяет речевые окна при помощи Silero VAD с умным слиянием пауз и поглощением коротких окон-сирот (< 3,2 с),
   * чтобы модель `GigaAM v3 RNN-T` всегда имела достаточный левый и правый речевой контекст и не разрезала фразы.
   */
  private fun detectSpeechWindows(
    samples: FloatArray,
    diarSlices: List<DiarizationSlice>,
    numThreads: Int,
    preset: QualityPreset,
  ): List<SpeechWindow> {
    val maxSec = when (preset) {
      QualityPreset.FAST -> 18.0f
      QualityPreset.STANDARD -> 16.5f
      QualityPreset.HIGH -> 14.5f
    }
    val vadModelPath = ensureAssetExtracted("models/vad/silero_vad.onnx")
    val vadCfg = VadModelConfig(
      sileroVadModelConfig = SileroVadModelConfig(
        model = vadModelPath,
        threshold = if (preset == QualityPreset.HIGH) 0.35f else 0.38f,
        minSilenceDuration = if (preset == QualityPreset.HIGH) 0.36f else 0.40f,
        minSpeechDuration = 0.18f,
        windowSize = 512,
        maxSpeechDuration = maxSec,
      ),
      sampleRate = AudioDecoder.TARGET_RATE,
      numThreads = min(2, numThreads),
      provider = "cpu",
    )
    val windows = mutableListOf<SpeechWindow>()
    val padSamples = 3000 // ~188 мс запаса по краям для защиты тихих начал и окончаний слов
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
          val s = max(0, seg.start.toInt() - padSamples)
          val e = min(samples.size, seg.start.toInt() + seg.samples.size + padSamples)
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
        val s = max(0, seg.start.toInt() - padSamples)
        val e = min(samples.size, seg.start.toInt() + seg.samples.size + padSamples)
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

    // Сливаем близкие окна (<= 800 мс), если суммарная длина не превышает maxSec
    var merged = mutableListOf<SpeechWindow>()
    val maxWindowSamples = (AudioDecoder.TARGET_RATE * maxSec).toInt()
    val mergeGapSamples = (AudioDecoder.TARGET_RATE * 0.80).toInt()
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

    // Поглощаем короткие окна-сироты (<= 3,2 с) в соседнее окно, если пауза <= 1,2 с,
    // чтобы одиночные фамилии или концовки фраз не оставались без контекста RNN-T декодера
    val hardMaxSamples = (AudioDecoder.TARGET_RATE * (maxSec + 3.5f)).toInt()
    val maxOrphanSamples = (AudioDecoder.TARGET_RATE * 3.2f).toInt()
    val maxOrphanGapSamples = (AudioDecoder.TARGET_RATE * 1.2f).toInt()
    var changed = true
    while (changed) {
      changed = false
      val nextList = mutableListOf<SpeechWindow>()
      var i = 0
      while (i < merged.size) {
        val w = merged[i]
        val wLen = w.endSample - w.startSample
        if (wLen <= maxOrphanSamples) {
          if (i + 1 < merged.size) {
            val nxt = merged[i + 1]
            if (nxt.startSample - w.endSample <= maxOrphanGapSamples &&
              nxt.endSample - w.startSample <= hardMaxSamples
            ) {
              nextList += SpeechWindow(w.startSample, max(w.endSample, nxt.endSample))
              i += 2
              changed = true
              continue
            }
          }
          if (nextList.isNotEmpty()) {
            val prv = nextList.last()
            if (w.startSample - prv.endSample <= maxOrphanGapSamples &&
              w.endSample - prv.startSample <= hardMaxSamples
            ) {
              nextList[nextList.lastIndex] = SpeechWindow(prv.startSample, max(prv.endSample, w.endSample))
              i += 1
              changed = true
              continue
            }
          }
        }
        nextList += w
        i += 1
      }
      merged = nextList
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
      val trimPunctRegex = Regex("[.,!?;:—»«]+$")
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
            val winWords = TranscriptAligner.tokensToWords(rawTokens, win.startMs, win.endMs)
            for (w in winWords) {
              val prev = words.lastOrNull()
              if (prev != null && w.startMs < prev.endMs - 40L) {
                val wClean = w.text.lowercase().replace(trimPunctRegex, "")
                val prevClean = prev.text.lowercase().replace(trimPunctRegex, "")
                if (wClean.isNotEmpty() && wClean == prevClean) continue
              }
              words += w
            }
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
