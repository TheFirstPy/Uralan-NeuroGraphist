package ru.hexteam.stenograf.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.FileDownload
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.People
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import ru.hexteam.stenograf.core.export.ExportFormat
import ru.hexteam.stenograf.core.export.TranscriptExporter
import ru.hexteam.stenograf.core.model.ProcessingParams
import ru.hexteam.stenograf.core.model.QualityPreset
import ru.hexteam.stenograf.core.model.TranscriptRecord
import ru.hexteam.stenograf.core.model.Utterance
import ru.hexteam.stenograf.core.pipeline.PipelineProgress
import ru.hexteam.stenograf.ui.theme.speakerAccentColor

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun StenografApp(vm: StenografViewModel = viewModel()) {
  val state by vm.uiState.collectAsState()
  val isRecording by vm.recorder.isRecording.collectAsState()
  val recElapsedMs by vm.recorder.elapsedMs.collectAsState()
  val recAmplitude by vm.recorder.amplitude.collectAsState()
  val isPlaying by vm.player.isPlaying.collectAsState()
  val playPositionMs by vm.player.positionMs.collectAsState()
  val playDurationMs by vm.player.durationMs.collectAsState()

  val context = LocalContext.current
  val snackbarHostState = remember { SnackbarHostState() }

  var selectedTab by rememberSaveable { mutableIntStateOf(0) }
  var showSettingsDialog by rememberSaveable { mutableStateOf(false) }
  var showHistorySheet by rememberSaveable { mutableStateOf(false) }
  var showSamplesSheet by rememberSaveable { mutableStateOf(false) }
  var showExportSheet by rememberSaveable { mutableStateOf(false) }
  var pendingSaveFormat by remember { mutableStateOf(ExportFormat.JSON_HACKATHON) }
  var editingUtterance by remember { mutableStateOf<Utterance?>(null) }
  var renamingSpeakerId by remember { mutableStateOf<String?>(null) }

  // Системный выбор аудиофайла из памяти устройства
  val openAudioLauncher = rememberLauncherForActivityResult(
    contract = ActivityResultContracts.OpenDocument()
  ) { uri ->
    if (uri != null) vm.importAudioUri(uri)
  }

  // Системное сохранение файла результата
  val createExportLauncher = rememberLauncherForActivityResult(
    contract = ActivityResultContracts.CreateDocument("*/*")
  ) { uri ->
    if (uri != null) vm.writeExportToUri(uri, pendingSaveFormat)
  }

  // Запрос разрешения на запись с микрофона
  val micPermissionLauncher = rememberLauncherForActivityResult(
    contract = ActivityResultContracts.RequestPermission()
  ) { granted ->
    if (granted) vm.startMicrophoneRecording()
  }

  LaunchedEffect(state.statusMessage, state.errorMessage) {
    val msg = state.errorMessage ?: state.statusMessage
    if (!msg.isNullOrBlank()) {
      snackbarHostState.showSnackbar(msg)
      vm.clearMessages()
    }
  }

  Scaffold(
    snackbarHost = { SnackbarHost(snackbarHostState) },
    topBar = {
      TopAppBar(
        title = {
          Row(verticalAlignment = Alignment.CenterVertically) {
            UralanLogoBadge(size = 36.dp)
            Spacer(Modifier.width(10.dp))
            Column {
              Text(
                text = "Уралан: Нейрографист",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
              )
              Text(
                text = "Офлайн ASR + Диаризация + LLM · Режим: ${state.params.qualityPreset.label}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
              )
            }
          }
        },
        actions = {
          if (state.records.isNotEmpty()) {
            IconButton(onClick = { showHistorySheet = true }) {
              Icon(Icons.Default.History, contentDescription = "История расшифровок")
            }
          }
          IconButton(onClick = { showSettingsDialog = true }) {
            Icon(Icons.Default.Settings, contentDescription = "Параметры обработки")
          }
        },
        colors = TopAppBarDefaults.topAppBarColors(
          containerColor = MaterialTheme.colorScheme.surface,
        ),
      )
    },
  ) { innerPadding ->
    Column(
      modifier = Modifier
        .fillMaxSize()
        .padding(innerPadding),
    ) {
      // Панель действий: Импорт файла, Реальные записи, Запись с микрофона, Экспорт
      ActionToolbar(
        isProcessing = state.isProcessing,
        isRecording = isRecording,
        recElapsedMs = recElapsedMs,
        recAmplitude = recAmplitude,
        hasActiveRecord = state.activeRecord != null,
        onImportClick = {
          openAudioLauncher.launch(arrayOf("audio/*", "application/ogg", "video/mp4"))
        },
        onSamplesClick = { showSamplesSheet = true },
        onMicToggle = {
          if (isRecording) {
            vm.stopMicrophoneAndProcess()
          } else {
            val granted = ContextCompat.checkSelfPermission(
              context,
              Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED
            if (granted) vm.startMicrophoneRecording()
            else micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
          }
        },
        onExportClick = { showExportSheet = true },
      )

      // Баннер прогресса локальной обработки
      AnimatedVisibility(visible = state.isProcessing && state.progress != null) {
        state.progress?.let { prog ->
          ProcessingBanner(
            progress = prog,
            params = state.params,
            onCancel = vm::cancelProcessing,
          )
        }
      }

      val active = state.activeRecord
      if (active == null) {
        EmptyWorkspaceCard(
          samples = vm.builtinSamples,
          isProcessing = state.isProcessing,
          onImportClick = {
            openAudioLauncher.launch(arrayOf("audio/*", "application/ogg", "video/mp4"))
          },
          onSampleSelected = { vm.processBuiltinSample(it) },
        )
      } else {
        // Плеер с таймлайном спикеров и переходом к репликам
        PlayerBar(
          record = active,
          isPlaying = isPlaying,
          positionMs = playPositionMs,
          durationMs = if (playDurationMs > 0) playDurationMs else active.durationMs,
          onTogglePlay = { vm.player.togglePlayPause(active.audioPath) },
          onSeek = { vm.player.seekTo(it) },
        )

        // Вкладки рабочего экрана: используем ScrollableTabRow, чтобы заголовки никогда не обрезались многоточием
        val tabs = listOf(
          "Диалог (${active.utterances.size})",
          "Участники (${active.speakers.size})",
          "ИИ-Протокол",
          "JSON и метрики",
        )
        ScrollableTabRow(
          selectedTabIndex = selectedTab,
          edgePadding = 8.dp,
        ) {
          tabs.forEachIndexed { idx, title ->
            Tab(
              selected = selectedTab == idx,
              onClick = { selectedTab = idx },
              text = {
                Text(
                  text = title,
                  maxLines = 1,
                  softWrap = false,
                  fontSize = 13.sp,
                  fontWeight = if (selectedTab == idx) FontWeight.SemiBold else FontWeight.Normal,
                )
              },
            )
          }
        }

        when (selectedTab) {
          0 -> DialogueTab(
            record = active,
            searchQuery = state.searchQuery,
            speakerFilter = state.selectedSpeakerFilter,
            currentPlayMs = if (isPlaying) playPositionMs else -1L,
            reprocessingId = state.reprocessUtteranceId,
            onSearchChange = vm::setSearchQuery,
            onSpeakerFilterClick = vm::setSpeakerFilter,
            onPlayUtterance = { u -> vm.player.playFrom(active.audioPath, u.startMs, u.endMs) },
            onEditUtterance = { u -> editingUtterance = u },
            onReprocessUtterance = { u -> vm.reprocessSingleUtterance(u.id) },
            onRenameSpeaker = { spk -> renamingSpeakerId = spk },
          )
          1 -> SpeakersTab(
            record = active,
            onRenameSpeaker = { spk -> renamingSpeakerId = spk },
            onFilterBySpeaker = { spk ->
              vm.setSpeakerFilter(spk)
              selectedTab = 0
            },
          )
          2 -> ProtocolLlmTab(
            record = active,
            isGenerating = state.isGeneratingProtocol,
            onGenerate = vm::generateProtocolWithLlm,
            onShareMarkdown = { vm.shareExport(ExportFormat.MARKDOWN) },
            onSeekToTimestamp = { ms ->
              vm.player.playFrom(active.audioPath, ms)
            },
          )
          3 -> JsonAndMetricsTab(
            record = active,
            onSaveJson = {
              pendingSaveFormat = ExportFormat.JSON_HACKATHON
              val base = active.title.replace(Regex("[^\\p{L}\\p{N}_-]+"), "_").trim('_').ifBlank { "transcript" }
              createExportLauncher.launch("$base.json")
            },
            onShareJson = { vm.shareExport(ExportFormat.JSON_HACKATHON) },
          )
        }
      }
    }
  }

  // Модальное окно выбора встроенных реальных аудиозаписей для быстрой проверки
  if (showSamplesSheet) {
    ModalBottomSheet(onDismissRequest = { showSamplesSheet = false }) {
      Column(
        modifier = Modifier
          .fillMaxWidth()
          .verticalScroll(rememberScrollState())
          .padding(horizontal = 20.dp, vertical = 8.dp),
      ) {
        Text(
          text = "Реальные тестовые аудиозаписи",
          style = MaterialTheme.typography.titleMedium,
          fontWeight = FontWeight.Bold,
        )
        Text(
          text = "Живые русскоязычные диалоги и разговоры для мгновенной проверки транскрибации и диаризации на устройстве",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
          modifier = Modifier.padding(top = 4.dp, bottom = 12.dp),
        )
        vm.builtinSamples.forEach { sample ->
          Card(
            modifier = Modifier
              .fillMaxWidth()
              .padding(vertical = 5.dp)
              .clickable {
                showSamplesSheet = false
                vm.processBuiltinSample(sample)
              },
            colors = CardDefaults.cardColors(
              containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
            ),
          ) {
            Row(
              modifier = Modifier.padding(14.dp),
              verticalAlignment = Alignment.CenterVertically,
            ) {
              Icon(
                Icons.Default.AudioFile,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(32.dp),
              )
              Spacer(Modifier.width(12.dp))
              Column(Modifier.weight(1f)) {
                Text(sample.title, fontWeight = FontWeight.SemiBold)
                Text(
                  sample.subtitle,
                  style = MaterialTheme.typography.bodySmall,
                  color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
              }
              Icon(Icons.Default.PlayArrow, contentDescription = "Запустить")
            }
          }
        }
        Spacer(Modifier.height(20.dp))
      }
    }
  }

  // Модальное окно экспорта файла
  if (showExportSheet && state.activeRecord != null) {
    val active = state.activeRecord!!
    ModalBottomSheet(onDismissRequest = { showExportSheet = false }) {
      Column(
        modifier = Modifier
          .fillMaxWidth()
          .verticalScroll(rememberScrollState())
          .padding(horizontal = 20.dp, vertical = 8.dp),
      ) {
        Text(
          text = "Экспорт расшифровки «${active.title}»",
          style = MaterialTheme.typography.titleMedium,
          fontWeight = FontWeight.Bold,
        )
        Spacer(Modifier.height(10.dp))
        ExportFormat.entries.forEach { fmt ->
          Card(
            modifier = Modifier
              .fillMaxWidth()
              .padding(vertical = 5.dp),
            colors = CardDefaults.cardColors(
              containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f),
            ),
          ) {
            Column(Modifier.padding(14.dp)) {
              Text(fmt.label, fontWeight = FontWeight.SemiBold)
              Text(
                fmt.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
              )
              Spacer(Modifier.height(8.dp))
              FlowRow(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
              ) {
                Button(
                  onClick = {
                    showExportSheet = false
                    pendingSaveFormat = fmt
                    val base = active.title
                      .replace(Regex("[^\\p{L}\\p{N}_-]+"), "_")
                      .trim('_')
                      .ifBlank { "transcript" }
                    createExportLauncher.launch("$base.${fmt.extension}")
                  },
                ) {
                  Icon(Icons.Default.FileDownload, contentDescription = null, modifier = Modifier.size(18.dp))
                  Spacer(Modifier.width(6.dp))
                  Text("Сохранить в файл")
                }
                OutlinedButton(
                  onClick = {
                    showExportSheet = false
                    vm.shareExport(fmt)
                  },
                ) {
                  Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(18.dp))
                  Spacer(Modifier.width(6.dp))
                  Text("Поделиться")
                }
              }
            }
          }
        }
        Spacer(Modifier.height(20.dp))
      }
    }
  }

  // Модальное окно истории расшифровок
  if (showHistorySheet) {
    ModalBottomSheet(onDismissRequest = { showHistorySheet = false }) {
      Column(
        modifier = Modifier
          .fillMaxWidth()
          .padding(horizontal = 20.dp, vertical = 8.dp),
      ) {
        Text(
          text = "Сохранённые расшифровки (${state.records.size})",
          style = MaterialTheme.typography.titleMedium,
          fontWeight = FontWeight.Bold,
        )
        Spacer(Modifier.height(10.dp))
        LazyColumn(modifier = Modifier.height(340.dp)) {
          items(state.records, key = { it.id }) { rec ->
            Card(
              modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp)
                .clickable {
                  showHistorySheet = false
                  vm.selectRecord(rec)
                },
            ) {
              Row(
                modifier = Modifier.padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
              ) {
                Column(Modifier.weight(1f)) {
                  Text(rec.title, fontWeight = FontWeight.SemiBold)
                  Text(
                    "Длительность: ${TranscriptExporter.formatTimeMs(rec.durationMs)} · Реплик: ${rec.utterances.size} · Спикеров: ${rec.speakers.size}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                  )
                }
                IconButton(onClick = { vm.deleteRecord(rec) }) {
                  Icon(Icons.Default.Delete, contentDescription = "Удалить")
                }
              }
            }
          }
        }
        Spacer(Modifier.height(16.dp))
      }
    }
  }

  // Диалог настроек конвейера
  if (showSettingsDialog) {
    SettingsDialog(
      params = state.params,
      onUpdate = vm::updateParams,
      onDismiss = { showSettingsDialog = false },
    )
  }

  // Диалог редактирования реплики
  editingUtterance?.let { utt ->
    val active = state.activeRecord
    if (active != null) {
      EditUtteranceDialog(
        utterance = utt,
        speakers = active.speakers,
        displayName = active::displayName,
        onDismiss = { editingUtterance = null },
        onSave = { newText, newSpk ->
          vm.updateUtterance(utt.id, newText, newSpk)
          editingUtterance = null
        },
      )
    }
  }

  // Диалог переименования спикера
  renamingSpeakerId?.let { spkId ->
    val active = state.activeRecord
    if (active != null) {
      RenameSpeakerDialog(
        speakerDisplayName = active.displayName(spkId),
        currentName = active.speakerNames[spkId] ?: "",
        onDismiss = { renamingSpeakerId = null },
        onSave = { newName ->
          vm.renameSpeaker(spkId, newName)
          renamingSpeakerId = null
        },
      )
    }
  }
}

@Composable
private fun ActionToolbar(
  isProcessing: Boolean,
  isRecording: Boolean,
  recElapsedMs: Long,
  recAmplitude: Float,
  hasActiveRecord: Boolean,
  onImportClick: () -> Unit,
  onSamplesClick: () -> Unit,
  onMicToggle: () -> Unit,
  onExportClick: () -> Unit,
) {
  Surface(
    tonalElevation = 2.dp,
    modifier = Modifier.fillMaxWidth(),
  ) {
    Column(modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
      val btnPadding = PaddingValues(horizontal = 11.dp, vertical = 6.dp)
      Row(
        modifier = Modifier
          .fillMaxWidth()
          .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically,
      ) {
        Button(
          onClick = onImportClick,
          enabled = !isProcessing && !isRecording,
          contentPadding = btnPadding,
        ) {
          Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(16.dp))
          Spacer(Modifier.width(5.dp))
          Text("Импорт", fontSize = 12.sp, maxLines = 1, softWrap = false)
        }

        FilledTonalButton(
          onClick = onSamplesClick,
          enabled = !isProcessing && !isRecording,
          contentPadding = btnPadding,
        ) {
          Icon(Icons.Default.GraphicEq, contentDescription = null, modifier = Modifier.size(16.dp))
          Spacer(Modifier.width(5.dp))
          Text("Примеры", fontSize = 12.sp, maxLines = 1, softWrap = false)
        }

        FilledTonalButton(
          onClick = onMicToggle,
          enabled = !isProcessing,
          contentPadding = btnPadding,
        ) {
          Icon(
            if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
            contentDescription = null,
            tint = if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(16.dp),
          )
          Spacer(Modifier.width(5.dp))
          Text(
            text = if (isRecording) "Стоп (${TranscriptExporter.formatTimeMs(recElapsedMs)})" else "Диктофон",
            fontSize = 12.sp,
            maxLines = 1,
            softWrap = false,
          )
        }

        if (hasActiveRecord) {
          OutlinedButton(
            onClick = onExportClick,
            enabled = !isProcessing,
            contentPadding = btnPadding,
          ) {
            Icon(Icons.Default.FileDownload, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(5.dp))
            Text("Экспорт", fontSize = 12.sp, maxLines = 1, softWrap = false)
          }
        }
      }

      if (isRecording) {
        Spacer(Modifier.height(6.dp))
        LinearProgressIndicator(
          progress = { (recAmplitude * 3.5f).coerceIn(0.05f, 1f) },
          modifier = Modifier
            .fillMaxWidth()
            .height(6.dp)
            .clip(RoundedCornerShape(3.dp)),
          color = MaterialTheme.colorScheme.error,
        )
      }
    }
  }
}

@Composable
private fun ProcessingBanner(
  progress: PipelineProgress,
  params: ProcessingParams,
  onCancel: () -> Unit,
) {
  Card(
    modifier = Modifier
      .fillMaxWidth()
      .padding(horizontal = 12.dp, vertical = 6.dp),
    colors = CardDefaults.cardColors(
      containerColor = MaterialTheme.colorScheme.primaryContainer,
    ),
  ) {
    Column(modifier = Modifier.padding(12.dp)) {
      Row(verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator(
          modifier = Modifier.size(22.dp),
          strokeWidth = 2.5.dp,
        )
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
          Text(
            text = progress.stage,
            fontWeight = FontWeight.Bold,
            style = MaterialTheme.typography.titleSmall,
          )
          Text(
            text = progress.detail,
            style = MaterialTheme.typography.bodySmall,
          )
        }
        Spacer(Modifier.width(6.dp))
        Text(
          text = "${(progress.fraction * 100).toInt()}%",
          fontWeight = FontWeight.Bold,
        )
        IconButton(onClick = onCancel) {
          Icon(Icons.Default.Close, contentDescription = "Отменить")
        }
      }
      Spacer(Modifier.height(6.dp))
      LinearProgressIndicator(
        progress = { progress.fraction.coerceIn(0f, 1f) },
        modifier = Modifier
          .fillMaxWidth()
          .height(6.dp)
          .clip(RoundedCornerShape(3.dp)),
      )
      Spacer(Modifier.height(4.dp))
      Text(
        text = "Режим: «${params.qualityPreset.label}» · Потоков CPU: ${params.numThreads}" +
          (if (params.denoise || params.qualityPreset == QualityPreset.HIGH) " · Шумоподавление GTCRN" else ""),
        style = MaterialTheme.typography.labelSmall,
      )
    }
  }
}

@Composable
fun UralanLogoBadge(
  modifier: Modifier = Modifier,
  size: Dp = 40.dp,
) {
  Canvas(modifier = modifier.size(size)) {
    val w = this.size.width
    val h = this.size.height

    // 1. Фон-значок: глубокий кобальтово-сапфировый градиент «Уралан»
    drawRoundRect(
      brush = Brush.linearGradient(
        colors = listOf(Color(0xFF07112B), Color(0xFF102A6B), Color(0xFF1D4ED8)),
        start = Offset.Zero,
        end = Offset(w, h),
      ),
      cornerRadius = CornerRadius(w * 0.24f, h * 0.24f),
    )

    // 2. Золотой солнечный ореол в верхней части (калмыцкое солнце «Уралан»)
    drawArc(
      color = Color(0xFFFBBF24),
      startAngle = 200f,
      sweepAngle = 140f,
      useCenter = false,
      topLeft = Offset(w * 0.18f, h * 0.16f),
      size = Size(w * 0.64f, h * 0.64f),
      style = Stroke(width = w * 0.065f),
    )

    // 3. Три звуковые волны разных спикеров (диаризация голосов)
    val barW = w * 0.065f
    drawRoundRect(
      color = Color(0xFF38BDF8),
      topLeft = Offset(w * 0.19f, h * 0.39f),
      size = Size(barW, h * 0.24f),
      cornerRadius = CornerRadius(barW / 2, barW / 2),
    )
    drawRoundRect(
      color = Color(0xFF34D399),
      topLeft = Offset(w * 0.29f, h * 0.31f),
      size = Size(barW, h * 0.40f),
      cornerRadius = CornerRadius(barW / 2, barW / 2),
    )
    drawRoundRect(
      color = Color(0xFFFBBF24),
      topLeft = Offset(w * 0.39f, h * 0.37f),
      size = Size(barW, h * 0.28f),
      cornerRadius = CornerRadius(barW / 2, barW / 2),
    )

    // 4. Стрела-шеврон «Уралан» («Вперёд!»)
    val chevron1 = Path().apply {
      moveTo(w * 0.49f, h * 0.31f)
      lineTo(w * 0.59f, h * 0.31f)
      lineTo(w * 0.74f, h * 0.50f)
      lineTo(w * 0.59f, h * 0.69f)
      lineTo(w * 0.49f, h * 0.69f)
      lineTo(w * 0.64f, h * 0.50f)
      close()
    }
    drawPath(path = chevron1, color = Color.White)

    val chevron2 = Path().apply {
      moveTo(w * 0.64f, h * 0.38f)
      lineTo(w * 0.71f, h * 0.38f)
      lineTo(w * 0.82f, h * 0.50f)
      lineTo(w * 0.71f, h * 0.62f)
      lineTo(w * 0.64f, h * 0.62f)
      lineTo(w * 0.74f, h * 0.50f)
      close()
    }
    drawPath(path = chevron2, color = Color(0xFFF59E0B))
  }
}

@Composable
private fun EmptyWorkspaceCard(
  samples: List<BuiltinSample>,
  isProcessing: Boolean,
  onImportClick: () -> Unit,
  onSampleSelected: (BuiltinSample) -> Unit,
) {
  Column(
    modifier = Modifier
      .fillMaxSize()
      .verticalScroll(rememberScrollState())
      .padding(16.dp),
    verticalArrangement = Arrangement.spacedBy(12.dp),
  ) {
    Card(
      colors = CardDefaults.cardColors(
        containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.55f),
      ),
    ) {
      Column(Modifier.padding(16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
          UralanLogoBadge(size = 46.dp)
          Spacer(Modifier.width(12.dp))
          Column {
            Text(
              text = "Уралан: Нейрографист",
              style = MaterialTheme.typography.titleMedium,
              fontWeight = FontWeight.Bold,
            )
            Text(
              text = "«Уралан» (с калмыцкого — «Вперёд!») · Офлайн-расшифровка и диаризация",
              style = MaterialTheme.typography.labelMedium,
              color = MaterialTheme.colorScheme.primary,
            )
          }
        }
        Spacer(Modifier.height(8.dp))
        Text(
          text = "Все этапы выполняются автономно на Android-устройстве без интернета:\n" +
            "• Распознавание речи и пунктуации: GigaAM v3 RNN-T (SberDevices, int8)\n" +
            "• Диаризация говорящих: pyannote segmentation 3.0 + 3D-Speaker Cam++\n" +
            "• Фильтрация пауз и шума: Silero VAD + нейросеть GTCRN\n" +
            "• Генерация протокола совещания: локальная LLM Qwen3-0.6B (LiteRT-LM)",
          style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(12.dp))
        Button(
          onClick = onImportClick,
          enabled = !isProcessing,
          modifier = Modifier.fillMaxWidth(),
        ) {
          Icon(Icons.Default.FolderOpen, contentDescription = null)
          Spacer(Modifier.width(8.dp))
          Text("Импортировать аудиозапись с устройства")
        }
      }
    }

    Text(
      text = "Или запустите проверку на реальных записях диалогов:",
      style = MaterialTheme.typography.titleSmall,
      fontWeight = FontWeight.SemiBold,
    )

    samples.forEach { sample ->
      Card(
        modifier = Modifier
          .fillMaxWidth()
          .clickable(enabled = !isProcessing) { onSampleSelected(sample) },
      ) {
        Row(
          modifier = Modifier.padding(14.dp),
          verticalAlignment = Alignment.CenterVertically,
        ) {
          Icon(
            Icons.Default.AudioFile,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(28.dp),
          )
          Spacer(Modifier.width(12.dp))
          Column(Modifier.weight(1f)) {
            Text(sample.title, fontWeight = FontWeight.SemiBold)
            Text(
              sample.subtitle,
              style = MaterialTheme.typography.bodySmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
          }
          Icon(Icons.Default.PlayArrow, contentDescription = "Запустить")
        }
      }
    }
  }
}

@Composable
private fun PlayerBar(
  record: TranscriptRecord,
  isPlaying: Boolean,
  positionMs: Long,
  durationMs: Long,
  onTogglePlay: () -> Unit,
  onSeek: (Long) -> Unit,
) {
  Surface(
    tonalElevation = 1.dp,
    modifier = Modifier.fillMaxWidth(),
  ) {
    Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)) {
      Row(verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = onTogglePlay, modifier = Modifier.size(36.dp)) {
          Icon(
            if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
            contentDescription = if (isPlaying) "Пауза" else "Воспроизвести",
            tint = MaterialTheme.colorScheme.primary,
          )
        }
        Spacer(Modifier.width(6.dp))
        Column(Modifier.weight(1f)) {
          Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Text(
              text = record.title,
              style = MaterialTheme.typography.labelLarge,
              fontWeight = FontWeight.SemiBold,
              maxLines = 1,
              overflow = TextOverflow.Ellipsis,
              modifier = Modifier.weight(1f),
            )
            Spacer(Modifier.width(8.dp))
            Text(
              text = "${TranscriptExporter.formatTimeMs(positionMs)} / ${TranscriptExporter.formatTimeMs(durationMs)}",
              style = MaterialTheme.typography.labelSmall,
              fontFamily = FontFamily.Monospace,
              maxLines = 1,
              softWrap = false,
            )
          }
          Slider(
            value = if (durationMs > 0) (positionMs.toFloat() / durationMs).coerceIn(0f, 1f) else 0f,
            onValueChange = { frac -> onSeek((frac * durationMs).toLong()) },
            modifier = Modifier.height(22.dp),
          )
        }
      }

      // Цветовая шкала распределения реплик спикеров по времени
      SpeakerTimelineBar(record = record, durationMs = durationMs)
    }
  }
}

@Composable
private fun SpeakerTimelineBar(record: TranscriptRecord, durationMs: Long) {
  if (durationMs <= 0 || record.utterances.isEmpty()) return
  Box(
    modifier = Modifier
      .fillMaxWidth()
      .height(5.dp)
      .clip(RoundedCornerShape(3.dp))
      .background(MaterialTheme.colorScheme.surfaceVariant),
  ) {
    Row(Modifier.fillMaxSize()) {
      var cursor = 0L
      for (u in record.utterances) {
        val gap = (u.startMs - cursor).coerceAtLeast(0L)
        if (gap > 0) {
          Spacer(Modifier.weight((gap.toFloat() / durationMs).coerceAtLeast(0.001f)))
        }
        val span = (u.endMs - u.startMs).coerceAtLeast(80L)
        Box(
          modifier = Modifier
            .weight((span.toFloat() / durationMs).coerceAtLeast(0.002f))
            .height(5.dp)
            .background(speakerAccentColor(u.speakerId)),
        )
        cursor = u.endMs
      }
      val tail = (durationMs - cursor).coerceAtLeast(0L)
      if (tail > 0) {
        Spacer(Modifier.weight((tail.toFloat() / durationMs).coerceAtLeast(0.001f)))
      }
    }
  }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DialogueTab(
  record: TranscriptRecord,
  searchQuery: String,
  speakerFilter: String?,
  currentPlayMs: Long,
  reprocessingId: Int?,
  onSearchChange: (String) -> Unit,
  onSpeakerFilterClick: (String?) -> Unit,
  onPlayUtterance: (Utterance) -> Unit,
  onEditUtterance: (Utterance) -> Unit,
  onReprocessUtterance: (Utterance) -> Unit,
  onRenameSpeaker: (String) -> Unit,
) {
  Column(Modifier.fillMaxSize()) {
    // Компактная панель поиска и однострочная горизонтальная полоса фильтра по спикерам,
    // чтобы при любом числе участников диалог всегда занимал основную часть экрана.
    Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)) {
      OutlinedTextField(
        value = searchQuery,
        onValueChange = onSearchChange,
        placeholder = { Text("Поиск по тексту реплик…", fontSize = 13.sp) },
        leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, modifier = Modifier.size(18.dp)) },
        trailingIcon = {
          if (searchQuery.isNotEmpty()) {
            IconButton(onClick = { onSearchChange("") }) {
              Icon(Icons.Default.Close, contentDescription = "Очистить", modifier = Modifier.size(18.dp))
            }
          }
        },
        singleLine = true,
        textStyle = MaterialTheme.typography.bodyMedium,
        modifier = Modifier.fillMaxWidth(),
      )
      Spacer(Modifier.height(4.dp))
      Row(
        modifier = Modifier
          .fillMaxWidth()
          .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically,
      ) {
        FilterChip(
          selected = speakerFilter == null,
          onClick = { onSpeakerFilterClick(null) },
          label = { Text("Все (${record.utterances.size})", fontSize = 12.sp, maxLines = 1, softWrap = false) },
        )
        record.speakers.forEach { spk ->
          FilterChip(
            selected = speakerFilter == spk,
            onClick = { onSpeakerFilterClick(spk) },
            label = {
              Text(
                text = record.displayName(spk),
                fontSize = 12.sp,
                maxLines = 1,
                softWrap = false,
              )
            },
          )
        }
      }
    }

    val filtered = remember(record.utterances, searchQuery, speakerFilter) {
      record.utterances.filter { u ->
        val matchesSpk = speakerFilter == null || u.speakerId == speakerFilter
        matchesSpk && (searchQuery.isBlank() || u.text.contains(searchQuery, ignoreCase = true))
      }
    }

    LazyColumn(
      modifier = Modifier
        .fillMaxWidth()
        .weight(1f),
      contentPadding = PaddingValues(start = 12.dp, end = 12.dp, top = 4.dp, bottom = 28.dp),
      verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
      items(filtered, key = { it.id }) { utt ->
        val isActive = currentPlayMs in utt.startMs..utt.endMs
        UtteranceCard(
          utterance = utt,
          speakerDisplayName = record.displayName(utt.speakerId),
          isActivePlaying = isActive,
          isReprocessing = reprocessingId == utt.id,
          onPlayClick = { onPlayUtterance(utt) },
          onEditClick = { onEditUtterance(utt) },
          onReprocessClick = { onReprocessUtterance(utt) },
          onSpeakerClick = { onRenameSpeaker(utt.speakerId) },
        )
      }
    }
  }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun UtteranceCard(
  utterance: Utterance,
  speakerDisplayName: String,
  isActivePlaying: Boolean,
  isReprocessing: Boolean,
  onPlayClick: () -> Unit,
  onEditClick: () -> Unit,
  onReprocessClick: () -> Unit,
  onSpeakerClick: () -> Unit,
) {
  val accent = speakerAccentColor(utterance.speakerId)
  val borderMod = if (isActivePlaying) {
    Modifier.border(2.dp, accent, RoundedCornerShape(12.dp))
  } else {
    Modifier
  }

  Card(
    modifier = Modifier
      .fillMaxWidth()
      .then(borderMod)
      .clickable { onPlayClick() },
    shape = RoundedCornerShape(12.dp),
    colors = CardDefaults.cardColors(
      containerColor = if (isActivePlaying) accent.copy(alpha = 0.12f)
      else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
    ),
  ) {
    Column(modifier = Modifier.padding(12.dp)) {
      Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth(),
      ) {
        Box(
          modifier = Modifier
            .size(10.dp)
            .clip(CircleShape)
            .background(accent),
        )
        Spacer(Modifier.width(8.dp))
        Text(
          text = speakerDisplayName,
          color = accent,
          fontWeight = FontWeight.Bold,
          style = MaterialTheme.typography.labelLarge,
          maxLines = 1,
          overflow = TextOverflow.Ellipsis,
          modifier = Modifier
            .weight(1f, fill = false)
            .clickable { onSpeakerClick() },
        )
        Spacer(Modifier.weight(1f))
        Spacer(Modifier.width(8.dp))
        Text(
          text = "${TranscriptExporter.formatTimeMs(utterance.startMs)} – ${TranscriptExporter.formatTimeMs(utterance.endMs)}",
          style = MaterialTheme.typography.labelSmall,
          fontFamily = FontFamily.Monospace,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
          maxLines = 1,
          softWrap = false,
        )
      }

      Spacer(Modifier.height(6.dp))
      Text(
        text = utterance.text,
        style = MaterialTheme.typography.bodyMedium,
      )

      Spacer(Modifier.height(6.dp))
      FlowRow(
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
      ) {
        AssistChip(
          onClick = onPlayClick,
          label = { Text("Прослушать", fontSize = 11.sp, maxLines = 1, softWrap = false) },
          leadingIcon = {
            Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(15.dp))
          },
        )
        AssistChip(
          onClick = onEditClick,
          label = { Text("Изменить", fontSize = 11.sp, maxLines = 1, softWrap = false) },
          leadingIcon = {
            Icon(Icons.Default.Edit, contentDescription = null, modifier = Modifier.size(15.dp))
          },
        )
        AssistChip(
          onClick = onReprocessClick,
          enabled = !isReprocessing,
          label = {
            Text(
              if (isReprocessing) "Переобработка…" else "Переобработать",
              fontSize = 11.sp,
              maxLines = 1,
              softWrap = false,
            )
          },
          leadingIcon = {
            if (isReprocessing) {
              CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp)
            } else {
              Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(15.dp))
            }
          },
        )
      }
    }
  }
}

@Composable
private fun SpeakersTab(
  record: TranscriptRecord,
  onRenameSpeaker: (String) -> Unit,
  onFilterBySpeaker: (String) -> Unit,
) {
  LazyColumn(
    modifier = Modifier.fillMaxSize(),
    contentPadding = PaddingValues(start = 14.dp, end = 14.dp, top = 12.dp, bottom = 28.dp),
    verticalArrangement = Arrangement.spacedBy(10.dp),
  ) {
    items(record.speakers) { spk ->
      val turns = record.utterances.filter { it.speakerId == spk }
      val totalSpeakMs = turns.sumOf { (it.endMs - it.startMs).coerceAtLeast(0L) }
      val share = if (record.durationMs > 0) (totalSpeakMs.toFloat() / record.durationMs).coerceIn(0f, 1f) else 0f
      val wordCount = turns.sumOf { it.text.split(Regex("\\s+")).count { w -> w.isNotBlank() } }
      val accent = speakerAccentColor(spk)

      Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
          Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
              modifier = Modifier
                .size(14.dp)
                .clip(CircleShape)
                .background(accent),
            )
            Spacer(Modifier.width(8.dp))
            Column(Modifier.weight(1f)) {
              Text(
                text = record.displayName(spk),
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleSmall,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
              )
              Text(
                text = "Доля речи в диалоге: ${(share * 100).toInt()}%",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
              )
            }
            OutlinedButton(
              onClick = { onRenameSpeaker(spk) },
              contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
            ) {
              Icon(Icons.Default.Edit, contentDescription = null, modifier = Modifier.size(15.dp))
              Spacer(Modifier.width(4.dp))
              Text("Имя", fontSize = 12.sp, maxLines = 1, softWrap = false)
            }
            Spacer(Modifier.width(6.dp))
            FilledTonalButton(
              onClick = { onFilterBySpeaker(spk) },
              contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
            ) {
              Icon(Icons.Default.People, contentDescription = null, modifier = Modifier.size(15.dp))
              Spacer(Modifier.width(4.dp))
              Text("Реплики", fontSize = 12.sp, maxLines = 1, softWrap = false)
            }
          }
          Spacer(Modifier.height(8.dp))
          Text(
            text = "Реплик: ${turns.size} · Слов: $wordCount · Эфирное время: ${TranscriptExporter.formatTimeMs(totalSpeakMs)}",
            style = MaterialTheme.typography.bodySmall,
          )
          Spacer(Modifier.height(6.dp))
          LinearProgressIndicator(
            progress = { share },
            color = accent,
            modifier = Modifier
              .fillMaxWidth()
              .height(6.dp)
              .clip(RoundedCornerShape(3.dp)),
          )
        }
      }
    }
  }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ProtocolLlmTab(
  record: TranscriptRecord,
  isGenerating: Boolean,
  onGenerate: () -> Unit,
  onShareMarkdown: () -> Unit,
  onSeekToTimestamp: (Long) -> Unit,
) {
  Column(
    modifier = Modifier
      .fillMaxSize()
      .verticalScroll(rememberScrollState())
      .padding(start = 14.dp, end = 14.dp, top = 12.dp, bottom = 28.dp),
    verticalArrangement = Arrangement.spacedBy(12.dp),
  ) {
    Card(
      colors = CardDefaults.cardColors(
        containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f),
      ),
    ) {
      Column(Modifier.padding(14.dp)) {
        Text(
          text = "Локальная LLM Qwen3-0.6B (LiteRT-LM)",
          style = MaterialTheme.typography.titleSmall,
          fontWeight = FontWeight.Bold,
        )
        Text(
          text = "Формирует структурированный протокол совещания, главные тезисы, вопросы и поручения непосредственно на устройстве. Нажмите на любой таймкод ▶ [MM:SS.d] в протоколе, чтобы сразу прослушать этот фрагмент.",
          style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(10.dp))
        FlowRow(
          horizontalArrangement = Arrangement.spacedBy(10.dp),
          verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
          Button(
            onClick = onGenerate,
            enabled = !isGenerating,
          ) {
            if (isGenerating) {
              CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
            } else {
              Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(18.dp))
            }
            Spacer(Modifier.width(6.dp))
            Text(if (record.protocol.isNullOrBlank()) "Сформировать протокол" else "Обновить протокол")
          }
          if (!record.protocol.isNullOrBlank()) {
            OutlinedButton(onClick = onShareMarkdown) {
              Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(18.dp))
              Spacer(Modifier.width(6.dp))
              Text("Экспорт .md")
            }
          }
        }
      }
    }

    if (!record.protocol.isNullOrBlank()) {
      Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
          FormattedMarkdownProtocol(
            markdown = record.protocol,
            onSeekToTimestamp = onSeekToTimestamp,
          )
        }
      }
    }
  }
}

/**
 * Компонент чистого форматированного отображения протокола совещания (Markdown -> нативная типографика Compose):
 * убирает служебные символы `#`, `###`, `**`, `` ` `` и превращает таймкоды `[MM:SS.d]` в интерактивные
 * кликабельные кнопки воспроизведения (Idea #1).
 */
@Composable
private fun FormattedMarkdownProtocol(
  markdown: String,
  onSeekToTimestamp: (Long) -> Unit,
) {
  val lines = remember(markdown) { markdown.lines() }
  val primaryColor = MaterialTheme.colorScheme.primary
  val codeBg = MaterialTheme.colorScheme.surfaceVariant

  Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
    lines.forEach { rawLine ->
      val trimmed = rawLine.trim()
      when {
        trimmed.isEmpty() -> {
          Spacer(Modifier.height(2.dp))
        }
        trimmed == "---" || trimmed == "***" -> {
          HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
        }
        trimmed.startsWith("#") -> {
          val level = trimmed.takeWhile { it == '#' }.length
          val headingText = trimmed.dropWhile { it == '#' }.trim().replace("**", "")
          if (headingText.isNotEmpty()) {
            Spacer(Modifier.height(4.dp))
            Text(
              text = headingText,
              style = if (level <= 2) MaterialTheme.typography.titleMedium else MaterialTheme.typography.titleSmall,
              fontWeight = FontWeight.Bold,
              color = primaryColor,
            )
            HorizontalDivider(
              modifier = Modifier.padding(top = 2.dp),
              color = primaryColor.copy(alpha = 0.2f),
            )
          }
        }
        trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("• ") -> {
          val body = trimmed.substring(2).trim()
          val tsMs = TranscriptExporter.extractFirstTimestampMs(body)
          // Если в начале тезиса стоит `[MM:SS.d]`, выделяем его в кликабельную пилюлю с иконкой Play
          val bodyWithoutLeadingTs = body
            .replaceFirst(Regex("^`?\\[\\d{1,3}:\\d{2}(?:\\.\\d)?\\]`?\\s*"), "")
            .trim()
          val rowMod = if (tsMs != null) {
            Modifier
              .fillMaxWidth()
              .clip(RoundedCornerShape(8.dp))
              .clickable { onSeekToTimestamp(tsMs) }
              .padding(vertical = 2.dp)
          } else {
            Modifier.fillMaxWidth()
          }
          Row(
            modifier = rowMod,
            verticalAlignment = Alignment.Top,
          ) {
            if (tsMs != null && bodyWithoutLeadingTs.length < body.length) {
              Surface(
                color = primaryColor.copy(alpha = 0.14f),
                shape = RoundedCornerShape(6.dp),
                modifier = Modifier
                  .padding(end = 8.dp, top = 2.dp)
                  .clickable { onSeekToTimestamp(tsMs) },
              ) {
                Row(
                  verticalAlignment = Alignment.CenterVertically,
                  modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                ) {
                  Icon(
                    Icons.Default.PlayArrow,
                    contentDescription = "Прослушать",
                    tint = primaryColor,
                    modifier = Modifier.size(13.dp),
                  )
                  Spacer(Modifier.width(2.dp))
                  Text(
                    text = TranscriptExporter.formatTimeMs(tsMs),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryColor,
                    maxLines = 1,
                    softWrap = false,
                  )
                }
              }
              Text(
                text = rememberFormattedInlineMarkdown(bodyWithoutLeadingTs, codeBg),
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f),
              )
            } else {
              Text(
                text = "•",
                fontWeight = FontWeight.Bold,
                color = primaryColor,
                modifier = Modifier.padding(end = 8.dp, top = 1.dp),
              )
              Text(
                text = rememberFormattedInlineMarkdown(body, codeBg),
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f),
              )
            }
          }
        }
        trimmed.matches(Regex("^\\d+\\.\\s+.*")) -> {
          val numPart = trimmed.substringBefore(".") + "."
          val body = trimmed.substringAfter(".").trim()
          val tsMs = TranscriptExporter.extractFirstTimestampMs(body)
          Row(
            modifier = Modifier
              .fillMaxWidth()
              .then(if (tsMs != null) Modifier.clickable { onSeekToTimestamp(tsMs) } else Modifier),
            verticalAlignment = Alignment.Top,
          ) {
            Text(
              text = numPart,
              fontWeight = FontWeight.Bold,
              color = primaryColor,
              modifier = Modifier.padding(end = 8.dp),
            )
            Text(
              text = rememberFormattedInlineMarkdown(body, codeBg),
              style = MaterialTheme.typography.bodyMedium,
              modifier = Modifier.weight(1f),
            )
          }
        }
        else -> {
          Text(
            text = rememberFormattedInlineMarkdown(trimmed, codeBg),
            style = MaterialTheme.typography.bodyMedium,
          )
        }
      }
    }
  }
}

@Composable
private fun rememberFormattedInlineMarkdown(text: String, codeBg: Color): AnnotatedString {
  return remember(text, codeBg) {
    parseInlineMarkdown(text, codeBg)
  }
}

private fun parseInlineMarkdown(text: String, codeBg: Color): AnnotatedString = buildAnnotatedString {
  val cleaned = text.replace(Regex("^#{1,6}\\s*"), "")
  var i = 0
  while (i < cleaned.length) {
    when {
      cleaned.startsWith("**", i) -> {
        val end = cleaned.indexOf("**", i + 2)
        if (end > i + 2) {
          withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
            append(cleaned.substring(i + 2, end))
          }
          i = end + 2
        } else {
          i += 2
        }
      }
      cleaned[i] == '`' -> {
        val end = cleaned.indexOf('`', i + 1)
        if (end > i + 1) {
          withStyle(
            SpanStyle(
              fontFamily = FontFamily.Monospace,
              background = codeBg,
              fontWeight = FontWeight.Medium,
            )
          ) {
            append(" " + cleaned.substring(i + 1, end) + " ")
          }
          i = end + 1
        } else {
          i++
        }
      }
      cleaned[i] == '*' && (i + 1 < cleaned.length && cleaned[i + 1] != ' ') -> {
        val end = cleaned.indexOf('*', i + 1)
        if (end > i + 1) {
          withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
            append(cleaned.substring(i + 1, end))
          }
          i = end + 1
        } else {
          append(cleaned[i])
          i++
        }
      }
      else -> {
        append(cleaned[i])
        i++
      }
    }
  }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun JsonAndMetricsTab(
  record: TranscriptRecord,
  onSaveJson: () -> Unit,
  onShareJson: () -> Unit,
) {
  val jsonText = remember(record.utterances) {
    TranscriptExporter.toHackathonJson(record.utterances)
  }

  Column(
    modifier = Modifier
      .fillMaxSize()
      .verticalScroll(rememberScrollState())
      .padding(start = 14.dp, end = 14.dp, top = 12.dp, bottom = 28.dp),
    verticalArrangement = Arrangement.spacedBy(12.dp),
  ) {
    Card(Modifier.fillMaxWidth()) {
      Column(Modifier.padding(14.dp)) {
        Text(
          text = "Метрики производительности на устройстве",
          style = MaterialTheme.typography.titleSmall,
          fontWeight = FontWeight.Bold,
        )
        Spacer(Modifier.height(6.dp))
        val t = record.timings
        Text(
          text = "• Режим обработки: ${record.params.qualityPreset.label}\n" +
            "• Длительность аудио: ${TranscriptExporter.formatTimeMs(record.durationMs)} (${record.durationMs} мс)\n" +
            "• Полное время обработки: ${t.totalMs} мс (RTF = ${"%.3f".format(record.rtf)})\n" +
            "• Декодирование и ресемплинг: ${t.decodeMs} мс\n" +
            "• DSP / Шумоподавление: ${t.preprocessMs} мс\n" +
            "• Диаризация и VAD: ${t.diarizationMs} мс\n" +
            "• Распознавание речи (GigaAM v3 RNN-T): ${t.asrMs} мс\n" +
            "• Пословное выравнивание: ${t.alignMs} мс",
          style = MaterialTheme.typography.bodySmall,
          fontFamily = FontFamily.Monospace,
        )
      }
    }

    FlowRow(
      horizontalArrangement = Arrangement.spacedBy(10.dp),
      verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
      Button(onClick = onSaveJson) {
        Icon(Icons.Default.Code, contentDescription = null, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(6.dp))
        Text("Сохранить JSON в файл")
      }
      OutlinedButton(onClick = onShareJson) {
        Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(6.dp))
        Text("Поделиться JSON")
      }
    }

    Card(
      modifier = Modifier.fillMaxWidth(),
      colors = CardDefaults.cardColors(
        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
      ),
    ) {
      Column(Modifier.padding(12.dp)) {
        Text(
          text = "Выходной JSON (формат задания Nuclear IT Hack, стр. 15):",
          style = MaterialTheme.typography.labelMedium,
          fontWeight = FontWeight.Bold,
        )
        Spacer(Modifier.height(6.dp))
        Text(
          text = jsonText,
          fontFamily = FontFamily.Monospace,
          fontSize = 11.sp,
        )
      }
    }
  }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun SettingsDialog(
  params: ProcessingParams,
  onUpdate: ((ProcessingParams) -> ProcessingParams) -> Unit,
  onDismiss: () -> Unit,
) {
  AlertDialog(
    onDismissRequest = onDismiss,
    title = { Text("Настройки обработки", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold) },
    text = {
      Column(
        modifier = Modifier
          .fillMaxWidth()
          .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp),
      ) {
        Text(
          text = "Режим скорости и качества:",
          fontWeight = FontWeight.SemiBold,
          style = MaterialTheme.typography.bodyMedium,
        )

        QualityPreset.entries.forEach { preset ->
          val selected = params.qualityPreset == preset
          Card(
            modifier = Modifier
              .fillMaxWidth()
              .clickable {
                onUpdate {
                  it.copy(
                    qualityPreset = preset,
                    denoise = if (preset == QualityPreset.HIGH) true else if (preset == QualityPreset.FAST) false else it.denoise,
                  )
                }
              },
            colors = CardDefaults.cardColors(
              containerColor = if (selected) {
                MaterialTheme.colorScheme.primaryContainer
              } else {
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)
              },
            ),
          ) {
            Row(
              modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
              verticalAlignment = Alignment.CenterVertically,
            ) {
              RadioButton(
                selected = selected,
                onClick = {
                  onUpdate {
                    it.copy(
                      qualityPreset = preset,
                      denoise = if (preset == QualityPreset.HIGH) true else if (preset == QualityPreset.FAST) false else it.denoise,
                    )
                  }
                },
              )
              Spacer(Modifier.width(4.dp))
              Column(Modifier.weight(1f)) {
                Text(
                  text = preset.label,
                  fontWeight = FontWeight.Bold,
                  style = MaterialTheme.typography.bodyMedium,
                )
                Text(
                  text = preset.subtitle,
                  style = MaterialTheme.typography.bodySmall,
                  color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
              }
            }
          }
        }

        HorizontalDivider()

        Text(
          text = "Количество участников разговора: " +
            (if (params.numSpeakers == 0) "Автоопределение" else "${params.numSpeakers}"),
          fontWeight = FontWeight.SemiBold,
          style = MaterialTheme.typography.bodyMedium,
        )
        FlowRow(
          horizontalArrangement = Arrangement.spacedBy(6.dp),
          verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
          listOf(0, 2, 3, 4, 5).forEach { n ->
            FilterChip(
              selected = params.numSpeakers == n,
              onClick = { onUpdate { it.copy(numSpeakers = n) } },
              label = { Text(if (n == 0) "Авто" else "$n") },
            )
          }
        }

        HorizontalDivider()

        Text(
          text = "Порог объединения голосов: ${"%.2f".format(params.clusterThreshold)}",
          fontWeight = FontWeight.SemiBold,
          style = MaterialTheme.typography.bodyMedium,
        )
        Text(
          text = "Меньше (0.35–0.46) — чётче разделяет голоса; больше (0.55–0.70) — сильнее объединяет в одного спикера",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Slider(
          value = params.clusterThreshold,
          onValueChange = { v -> onUpdate { it.copy(clusterThreshold = v) } },
          valueRange = 0.30f..0.75f,
        )

        HorizontalDivider()

        Row(
          modifier = Modifier.fillMaxWidth(),
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.SpaceBetween,
        ) {
          Column(Modifier.weight(1f).padding(end = 8.dp)) {
            Text(
              "Шумоподавление GTCRN",
              fontWeight = FontWeight.SemiBold,
              style = MaterialTheme.typography.bodyMedium,
            )
            Text(
              "Очистка сильного фонового шума перед распознаванием",
              style = MaterialTheme.typography.bodySmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
          }
          Switch(
            checked = params.denoise || params.qualityPreset == QualityPreset.HIGH,
            enabled = params.qualityPreset != QualityPreset.FAST,
            onCheckedChange = { checked -> onUpdate { it.copy(denoise = checked) } },
          )
        }

        HorizontalDivider()

        Text(
          text = "Потоков процессора (CPU): ${params.numThreads}",
          fontWeight = FontWeight.SemiBold,
          style = MaterialTheme.typography.bodyMedium,
        )
        FlowRow(
          horizontalArrangement = Arrangement.spacedBy(6.dp),
          verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
          listOf(2, 4, 6, 8).forEach { th ->
            FilterChip(
              selected = params.numThreads == th,
              onClick = { onUpdate { it.copy(numThreads = th) } },
              label = { Text("$th") },
            )
          }
        }
      }
    },
    confirmButton = {
      TextButton(onClick = onDismiss) { Text("Готово") }
    },
  )
}

@Composable
private fun EditUtteranceDialog(
  utterance: Utterance,
  speakers: List<String>,
  displayName: (String) -> String,
  onDismiss: () -> Unit,
  onSave: (String, String) -> Unit,
) {
  var text by remember(utterance) { mutableStateOf(utterance.text) }
  var spk by remember(utterance) { mutableStateOf(utterance.speakerId) }

  AlertDialog(
    onDismissRequest = onDismiss,
    title = { Text("Редактирование реплики #${utterance.id}") },
    text = {
      Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(
          text = "Временной интервал: ${TranscriptExporter.formatTimeMs(utterance.startMs)} – ${TranscriptExporter.formatTimeMs(utterance.endMs)}",
          style = MaterialTheme.typography.labelSmall,
        )
        OutlinedTextField(
          value = text,
          onValueChange = { text = it },
          label = { Text("Текст реплики") },
          modifier = Modifier.fillMaxWidth(),
        )
        Text("Говорящий:", style = MaterialTheme.typography.labelMedium)
        Row(
          modifier = Modifier.horizontalScroll(rememberScrollState()),
          horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
          val allSpeakers = (speakers + listOf("speaker_${speakers.size + 1}")).distinct()
          allSpeakers.forEach { candidate ->
            FilterChip(
              selected = spk == candidate,
              onClick = { spk = candidate },
              label = { Text(displayName(candidate), maxLines = 1, softWrap = false) },
            )
          }
        }
      }
    },
    dismissButton = {
      TextButton(onClick = onDismiss) { Text("Отмена") }
    },
    confirmButton = {
      Button(onClick = { onSave(text, spk) }) { Text("Сохранить") }
    },
  )
}

@Composable
private fun RenameSpeakerDialog(
  speakerDisplayName: String,
  currentName: String,
  onDismiss: () -> Unit,
  onSave: (String) -> Unit,
) {
  var name by remember(speakerDisplayName) { mutableStateOf(currentName) }
  AlertDialog(
    onDismissRequest = onDismiss,
    title = { Text("Переименовать: $speakerDisplayName") },
    text = {
      Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
          "Введите имя участника для отображения в диалоге и протоколе совещания:",
          style = MaterialTheme.typography.bodySmall,
        )
        OutlinedTextField(
          value = name,
          onValueChange = { name = it },
          placeholder = { Text("Например: Александр") },
          singleLine = true,
          modifier = Modifier.fillMaxWidth(),
        )
      }
    },
    dismissButton = {
      TextButton(onClick = onDismiss) { Text("Отмена") }
    },
    confirmButton = {
      Button(onClick = { onSave(name) }) { Text("Сохранить") }
    },
  )
}
