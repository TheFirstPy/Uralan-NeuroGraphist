package ru.hexteam.stenograf.ui

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import ru.hexteam.stenograf.audio.AudioPlayerController
import ru.hexteam.stenograf.audio.AudioRecorder
import ru.hexteam.stenograf.core.export.ExportFormat
import ru.hexteam.stenograf.core.export.TranscriptExporter
import ru.hexteam.stenograf.core.llm.LocalLlmEngine
import ru.hexteam.stenograf.core.model.ProcessingParams
import ru.hexteam.stenograf.core.model.TranscriptRecord
import ru.hexteam.stenograf.core.pipeline.PipelineProgress
import ru.hexteam.stenograf.core.pipeline.SpeechPipeline
import ru.hexteam.stenograf.data.TranscriptRepository
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Встроенная демонстрационная запись («Диалоги котиков в Кото́вице»). */
data class BuiltinSample(
  val id: String,
  val title: String,
  val subtitle: String,
  val assetPath: String,
  val recommendedSpeakers: Int,
  val recommendDenoise: Boolean,
)

data class StenografUiState(
  val records: List<TranscriptRecord> = emptyList(),
  val activeRecord: TranscriptRecord? = null,
  val isProcessing: Boolean = false,
  val progress: PipelineProgress? = null,
  val params: ProcessingParams = ProcessingParams(
    numSpeakers = 0,
    clusterThreshold = 0.50f,
    denoise = false,
    numThreads = 4,
  ),
  val isGeneratingProtocol: Boolean = false,
  val reprocessUtteranceId: Int? = null,
  val statusMessage: String? = null,
  val errorMessage: String? = null,
  val searchQuery: String = "",
  val selectedSpeakerFilter: String? = null,
)

class StenografViewModel(app: Application) : AndroidViewModel(app) {

  private val repository = TranscriptRepository(app)
  private val pipeline = SpeechPipeline(app)
  private val llmEngine = LocalLlmEngine(app)
  val recorder = AudioRecorder(viewModelScope)
  val player = AudioPlayerController(viewModelScope)

  private var processingJob: Job? = null

  private val _uiState = MutableStateFlow(StenografUiState())
  val uiState: StateFlow<StenografUiState> = _uiState

  val builtinSamples = listOf(
    BuiltinSample(
      id = "real_1_finance_call",
      title = "Звонок 1: Консультант и клиент (телефон)",
      subtitle = "Реальный телефонный разговор (AxonData) · 2 говорящих · 42 с",
      assetPath = "samples/real_1_finance_call.wav",
      recommendedSpeakers = 2,
      recommendDenoise = false,
    ),
    BuiltinSample(
      id = "real_2_podlodka_speech_ai",
      title = "Диалог 2: Подкаст Podlodka — Распознавание речи",
      subtitle = "Реальная беседа ведущих и гостей (bond005/podlodka_speech) · 2 спикера",
      assetPath = "samples/real_2_podlodka_speech_ai.wav",
      recommendedSpeakers = 2,
      recommendDenoise = false,
    ),
    BuiltinSample(
      id = "real_3_podlodka_roundtable",
      title = "Диалог 3: Круглый стол Podlodka (3 спикера)",
      subtitle = "Реальные голоса 3 участников (архитектура, логирование, ML) · 3 спикера",
      assetPath = "samples/real_3_podlodka_roundtable.wav",
      recommendedSpeakers = 3,
      recommendDenoise = false,
    ),
    BuiltinSample(
      id = "real_4_dela_podcast",
      title = "Подкаст «Дела №67»: Позов, Ахмедова, Шастун",
      subtitle = "Реальный видеоподкаст (2 мин фрагмент) · 3 говорящих (2 мужских + 1 женский голос)",
      assetPath = "samples/real_4_dela_podcast.wav",
      recommendedSpeakers = 0,
      recommendDenoise = false,
    ),
  )

  init {
    viewModelScope.launch {
      val saved = repository.loadAll()
      _uiState.update {
        it.copy(
          records = saved,
          activeRecord = saved.firstOrNull(),
        )
      }
    }
  }

  fun updateParams(transform: (ProcessingParams) -> ProcessingParams) {
    _uiState.update { it.copy(params = transform(it.params)) }
  }

  fun selectRecord(record: TranscriptRecord) {
    player.pause()
    _uiState.update {
      it.copy(
        activeRecord = record,
        selectedSpeakerFilter = null,
        searchQuery = "",
      )
    }
  }

  fun deleteRecord(record: TranscriptRecord) {
    viewModelScope.launch {
      if (_uiState.value.activeRecord?.id == record.id) {
        player.release()
      }
      repository.delete(record)
      val updated = repository.loadAll()
      _uiState.update {
        it.copy(
          records = updated,
          activeRecord = if (it.activeRecord?.id == record.id) updated.firstOrNull() else it.activeRecord,
          statusMessage = "Запись «${record.title}» удалена",
        )
      }
    }
  }

  fun setSearchQuery(q: String) {
    _uiState.update { it.copy(searchQuery = q) }
  }

  fun setSpeakerFilter(speakerId: String?) {
    _uiState.update {
      val next = if (it.selectedSpeakerFilter == speakerId) null else speakerId
      it.copy(selectedSpeakerFilter = next)
    }
  }

  fun clearMessages() {
    _uiState.update { it.copy(statusMessage = null, errorMessage = null) }
  }

  /** Импорт пользовательского аудиофайла из памяти Android-устройства. */
  fun importAudioUri(uri: Uri) {
    val context = getApplication<Application>()
    val fileName = resolveDisplayName(uri) ?: "Аудиозапись"
    val cleanTitle = fileName.substringBeforeLast('.').ifBlank { fileName }
    startPipeline(uri = uri, title = cleanTitle, params = _uiState.value.params)
  }

  /** Запуск обработки одного из встроенных тестовых диалогов котиков. */
  fun processBuiltinSample(sample: BuiltinSample) {
    viewModelScope.launch {
      val context = getApplication<Application>()
      val cacheFile = withContext(Dispatchers.IO) {
        val out = File(context.cacheDir, "${sample.id}.wav")
        context.assets.open(sample.assetPath).use { input ->
          out.outputStream().use { output -> input.copyTo(output) }
        }
        out
      }
      val effectiveParams = _uiState.value.params.copy(
        denoise = _uiState.value.params.denoise || sample.recommendDenoise
      )
      startPipeline(
        uri = Uri.fromFile(cacheFile),
        title = sample.title,
        params = effectiveParams,
      )
    }
  }

  /** Старт записи живого диалога с микрофона устройства. */
  fun startMicrophoneRecording() {
    runCatching {
      player.pause()
      val context = getApplication<Application>()
      val recDir = File(context.cacheDir, "mic").apply { mkdirs() }
      val target = File(recDir, "mic_${System.currentTimeMillis()}.wav")
      recorder.start(target)
    }.onFailure { err ->
      _uiState.update { it.copy(errorMessage = "Ошибка микрофона: ${err.localizedMessage}") }
    }
  }

  /** Остановка записи с микрофона и немедленный запуск локальной расшифровки. */
  fun stopMicrophoneAndProcess() {
    viewModelScope.launch {
      val file = recorder.stop() ?: return@launch
      val stamp = SimpleDateFormat("dd.MM HH:mm", Locale.forLanguageTag("ru")).format(Date())
      startPipeline(
        uri = Uri.fromFile(file),
        title = "Диктофон $stamp",
        params = _uiState.value.params,
      )
    }
  }

  private fun startPipeline(uri: Uri, title: String, params: ProcessingParams) {
    if (_uiState.value.isProcessing) return
    player.pause()
    processingJob = viewModelScope.launch {
      _uiState.update {
        it.copy(
          isProcessing = true,
          progress = PipelineProgress("Подготовка", "Инициализация конвейера…", 0.02f),
          errorMessage = null,
          statusMessage = null,
        )
      }
      runCatching {
        val record = pipeline.processAudioUri(uri, title, params) { prog ->
          _uiState.update { state -> state.copy(progress = prog) }
        }
        repository.save(record)
        val all = repository.loadAll()
        _uiState.update {
          it.copy(
            records = all,
            activeRecord = record,
            isProcessing = false,
            progress = null,
            selectedSpeakerFilter = null,
            searchQuery = "",
            statusMessage = "Расшифровано реплик: ${record.utterances.size}, участников: ${record.speakers.size} (RTF ${"%.2f".format(record.rtf)})",
          )
        }
      }.onFailure { err ->
        _uiState.update {
          it.copy(
            isProcessing = false,
            progress = null,
            errorMessage = "Ошибка обработки: ${err.localizedMessage ?: err.javaClass.simpleName}",
          )
        }
      }
    }
  }

  fun cancelProcessing() {
    processingJob?.cancel()
    processingJob = null
    _uiState.update {
      it.copy(
        isProcessing = false,
        progress = null,
        statusMessage = "Обработка отменена пользователем",
      )
    }
  }

  /** Переименование говорящего (например, `speaker_1` -> «Кот Борис»). В JSON-экспорте сохраняется `speaker_1`. */
  fun renameSpeaker(speakerId: String, newName: String) {
    val active = _uiState.value.activeRecord ?: return
    val updatedNames = active.speakerNames.toMutableMap()
    if (newName.isBlank()) {
      updatedNames.remove(speakerId)
    } else {
      updatedNames[speakerId] = newName.trim()
    }
    val updated = active.copy(speakerNames = updatedNames)
    saveUpdatedRecord(updated, "Имя участника обновлено")
  }

  /** Ручное редактирование текста и/или смена говорящего у конкретной реплики. */
  fun updateUtterance(utteranceId: Int, newText: String, newSpeakerId: String) {
    val active = _uiState.value.activeRecord ?: return
    val updatedList = active.utterances.map { u ->
      if (u.id == utteranceId) {
        u.copy(text = newText.trim(), speakerId = newSpeakerId, edited = true)
      } else {
        u
      }
    }
    saveUpdatedRecord(active.copy(utterances = updatedList), "Реплика #$utteranceId сохранена")
  }

  /** Повторная нейросетевая обработка выбранной реплики с шумоподавлением GTCRN (Ultra Pro Max). */
  fun reprocessSingleUtterance(utteranceId: Int) {
    val active = _uiState.value.activeRecord ?: return
    val target = active.utterances.find { it.id == utteranceId } ?: return
    viewModelScope.launch {
      _uiState.update { it.copy(reprocessUtteranceId = utteranceId) }
      runCatching {
        val (newText, newWords) = pipeline.reprocessSegment(
          audioPath = active.audioPath,
          startMs = target.startMs,
          endMs = target.endMs,
          denoise = true,
          numThreads = active.params.numThreads,
        )
        val prevText = active.utterances.firstOrNull { it.id == utteranceId - 1 }?.text
        val refined = llmEngine.refineUtteranceText(target.copy(text = newText), prevText)
        val updatedList = active.utterances.map { u ->
          if (u.id == utteranceId) {
            u.copy(
              text = refined.ifBlank { u.text },
              words = newWords.ifEmpty { u.words },
              edited = true,
            )
          } else {
            u
          }
        }
        val updated = active.copy(utterances = updatedList)
        saveUpdatedRecord(updated, "Фрагмент #$utteranceId повторно обработан (GTCRN + GigaAM v3 + LLM)")
      }.onFailure { err ->
        _uiState.update {
          it.copy(errorMessage = "Не удалось переобработать фрагмент: ${err.localizedMessage}")
        }
      }
      _uiState.update { it.copy(reprocessUtteranceId = null) }
    }
  }

  /** Генерация протокола заседания локальной LLM Qwen3-0.6B. */
  fun generateProtocolWithLlm() {
    val active = _uiState.value.activeRecord ?: return
    if (_uiState.value.isGeneratingProtocol) return
    viewModelScope.launch {
      _uiState.update { it.copy(isGeneratingProtocol = true, statusMessage = "Локальная LLM Qwen3-0.6B формирует протокол…") }
      runCatching {
        val finalProtocol = llmEngine.generateMeetingProtocol(active) { partial ->
          _uiState.update { state ->
            val cur = state.activeRecord ?: return@update state
            state.copy(activeRecord = cur.copy(protocol = partial))
          }
        }
        val current = _uiState.value.activeRecord ?: active
        val updated = current.copy(protocol = finalProtocol)
        saveUpdatedRecord(updated, "Протокол заседания сформирован локальной LLM")
      }.onFailure { err ->
        _uiState.update {
          it.copy(errorMessage = "Ошибка LLM: ${err.localizedMessage}")
        }
      }
      _uiState.update { it.copy(isGeneratingProtocol = false) }
    }
  }

  /** Запись экспортируемого файла в выбранный пользователем URI (системный диалог сохранения). */
  fun writeExportToUri(uri: Uri, format: ExportFormat) {
    val active = _uiState.value.activeRecord ?: return
    viewModelScope.launch {
      runCatching {
        val content = TranscriptExporter.export(active, format)
        withContext(Dispatchers.IO) {
          getApplication<Application>().contentResolver.openOutputStream(uri)?.use { out ->
            out.write(content.toByteArray(Charsets.UTF_8))
          } ?: error("Не удалось открыть поток для записи файла")
        }
        _uiState.update {
          it.copy(statusMessage = "Файл экспортирован в формате ${format.label}")
        }
      }.onFailure { err ->
        _uiState.update {
          it.copy(errorMessage = "Ошибка сохранения файла: ${err.localizedMessage}")
        }
      }
    }
  }

  /** Быстрое сохранение во внутреннюю папку Загрузки/exports и открытие системного меню «Поделиться». */
  fun shareExport(format: ExportFormat) {
    val active = _uiState.value.activeRecord ?: return
    viewModelScope.launch {
      runCatching {
        val context = getApplication<Application>()
        val content = TranscriptExporter.export(active, format)
        val exportDir = File(context.cacheDir, "exports").apply { mkdirs() }
        val safeName = active.title
          .replace(Regex("[^\\p{L}\\p{N}_-]+"), "_")
          .trim('_')
          .ifBlank { "transcript" }
        val outFile = File(exportDir, "${safeName}.${format.extension}")
        withContext(Dispatchers.IO) {
          outFile.writeText(content, Charsets.UTF_8)
        }
        val uri = FileProvider.getUriForFile(
          context,
          "${context.packageName}.fileprovider",
          outFile,
        )
        val sendIntent = Intent(Intent.ACTION_SEND).apply {
          type = format.mimeType
          putExtra(Intent.EXTRA_STREAM, uri)
          putExtra(Intent.EXTRA_SUBJECT, active.title)
          addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        val chooser = Intent.createChooser(sendIntent, "Экспорт стенограммы").apply {
          addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(chooser)
      }.onFailure { err ->
        _uiState.update {
          it.copy(errorMessage = "Не удалось поделиться файлом: ${err.localizedMessage}")
        }
      }
    }
  }

  private fun saveUpdatedRecord(updated: TranscriptRecord, message: String) {
    viewModelScope.launch {
      repository.save(updated)
      val all = repository.loadAll()
      _uiState.update {
        it.copy(
          records = all,
          activeRecord = updated,
          statusMessage = message,
        )
      }
    }
  }

  private fun resolveDisplayName(uri: Uri): String? {
    if (uri.scheme == "file") return uri.lastPathSegment
    val cr = getApplication<Application>().contentResolver
    return runCatching {
      cr.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
        if (c.moveToFirst()) c.getString(0) else null
      }
    }.getOrNull() ?: uri.lastPathSegment
  }

  override fun onCleared() {
    super.onCleared()
    player.release()
    llmEngine.close()
  }
}
